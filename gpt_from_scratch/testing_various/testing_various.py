"""
Pure Titans-based Language Modeling Training with Memory As Context (MAC),
Gradient Checkpointing, and Mixed Precision Training.

This script implements a transformer-based language model that is augmented
with a neural long-term memory module using the Memory as Context (MAC) approach.
Key features include:
  - A persistent memory module (learned memory tokens) that is prepended to each
    input segment.
  - Gradient checkpointing for memory efficiency.
  - Mixed precision training via torch.cuda.amp.
  - Periodic training logs via TensorBoard.
  - Autoregressive text generation (inference).
"""

import datetime
import math
import os
import optuna
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tensorboard import program

from SinkGD import SinkGD

# ------------------ Hyperparameters ------------------
maximum_context_length = 1024  # Number of text tokens per training example.
max_inference_tokens = 1024

embedding_dimensions = 384  # Must be divisible by head_count.
head_count = 8
transformer_layers = 8
memory_tokens = 16  # Number of persistent memory tokens for MAC.

if embedding_dimensions % head_count != 0:
    raise ValueError("embedding_dimensions must be divisible by head_count.")

num_epochs = 20
parallel_batch_size = 5

# Training hyperparams.
learning_rate = 3e-4
dropout = 0.2
learning_rate_decay = 0.90
percent_for_plateau = 0.02
evaluation_patience = 0 # must be < num_epochs to use reducelronplateau

# ------------------ TensorBoard Setup ------------------
# Create a unique log directory based on the current date and time.
run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tracking_address = os.path.join("runs_titans", run_time)

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', tracking_address])
url = tb.launch()
print(f"TensorBoard listening on {url}")

# ------------------ Dataset Preparation ------------------
with open('tiny-shakespeare.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print(f"Length of dataset in characters: {len(text)}")
unique_chars = sorted(list(set(text)))
vocabulary_size = len(unique_chars)
print(f"{vocabulary_size} unique characters in dataset:\n{''.join(unique_chars)}")

# Simple character-level tokenization.
chartointegers = {ch: i for i, ch in enumerate(unique_chars)}
encode = lambda s: [chartointegers[c] for c in s]
integertochars = {i: ch for i, ch in enumerate(unique_chars)}
decode = lambda l: ''.join([integertochars[i] for i in l])

# Split dataset: 90% training, 10% validation.
train_percent = 0.9
text_as_tensor = torch.tensor(encode(text), dtype=torch.long)
training_index = int(train_percent * len(text_as_tensor))
training_data = text_as_tensor[:training_index]
validation_data = text_as_tensor[training_index:]

# ---- rev up that gpu -----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(8181)

# ------------------ Batch Functions ------------------
def get_training_batches():
    """
    Splits the training data into non-overlapping segments of length `maximum_context_length`.
    Shuffles the segments at the start of each epoch.
    """
    # Determine how many full segments we can extract.
    n = len(training_data) - maximum_context_length - 1
    # Create starting indices with a stride equal to maximum_context_length.
    indices = list(range(0, n, maximum_context_length))
    random.shuffle(indices)
    for i in range(0, len(indices), parallel_batch_size):
        batch_indices = indices[i:i+parallel_batch_size]
        contexts = torch.stack([training_data[idx:idx+maximum_context_length] for idx in batch_indices]).to(device)
        targets  = torch.stack([training_data[idx+1:idx+maximum_context_length+1] for idx in batch_indices]).to(device)
        yield contexts, targets

def get_validation_batches():
    """
    Splits the validation data into non-overlapping segments.
    """
    n = len(validation_data) - maximum_context_length - 1
    indices = list(range(0, n, maximum_context_length))
    # No shuffling for validation.
    for i in range(0, len(indices), parallel_batch_size):
        batch_indices = indices[i:i+parallel_batch_size]
        contexts = torch.stack([validation_data[idx:idx+maximum_context_length] for idx in batch_indices]).to(device)
        targets  = torch.stack([validation_data[idx+1:idx+maximum_context_length+1] for idx in batch_indices]).to(device)
        yield contexts, targets

# ------------------ Transformer Components ------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dimensions, head_size, bias=False)
        self.query = nn.Linear(embedding_dimensions, head_size, bias=False)
        self.value = nn.Linear(embedding_dimensions, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        keys = self.key(x)      # (B, T, head_size)
        queries = self.query(x) # (B, T, head_size)
        values = self.value(x)  # (B, T, head_size)
        weights = torch.matmul(queries, keys.transpose(-2, -1)) * (keys.shape[-1] ** -0.5)  # (B, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        weights = weights.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, values)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_dimensions)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, embedding_dimensions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dimensions, 4 * embedding_dimensions),
            nn.ReLU(),
            nn.Linear(4 * embedding_dimensions, embedding_dimensions),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# TitansBlock implements the Memory As Context (MAC) architecture.
class TitansBlock(nn.Module):
    def __init__(self, embedding_dimensions, persistent_memory, n_heads=head_count):
        super().__init__()
        self.persistent_memory = persistent_memory  # Shared memory tokens (a Parameter)
        head_size = embedding_dimensions // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(embedding_dimensions)
        self.ln1 = nn.LayerNorm(embedding_dimensions)
        self.ln2 = nn.LayerNorm(embedding_dimensions)
    def forward(self, x):
        B, T, D = x.shape
        M = self.persistent_memory.shape[0]  # Number of memory tokens.
        mem = self.persistent_memory.unsqueeze(0).expand(B, M, D)  # (B, M, D)
        x_aug = torch.cat([mem, x], dim=1)  # (B, M+T, D)
        x_norm = self.ln1(x_aug)
        attn_out = self.sa(x_norm)  # (B, M+T, D)
        x_aug = x_aug + attn_out    # Residual connection.
        x_text = x_aug[:, M:, :]  # (B, T, D)
        x_text = x_text + self.ffwd(self.ln2(x_text))
        return x_text

# Wrap the block with gradient checkpointing.
class CheckpointedTitansBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    def forward(self, x):
        return checkpoint.checkpoint(self.block, x, use_reentrant=False)

# TitansLMModel uses the MAC architecture.
class TitansLMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dimensions, transformer_layers,
                 maximum_context_length, memory_tokens, dropout=dropout, n_heads=head_count):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dimensions)
        self.pos_embedding = nn.Parameter(torch.zeros(1, maximum_context_length, embedding_dimensions))
        self.dropout = nn.Dropout(dropout)
        self.persistent_memory = nn.Parameter(torch.randn(memory_tokens, embedding_dimensions))
        self.blocks = nn.ModuleList([
            CheckpointedTitansBlock(TitansBlock(embedding_dimensions, self.persistent_memory, n_heads=n_heads))
            for _ in range(transformer_layers)
        ])
        self.ln_f = nn.LayerNorm(embedding_dimensions)
        self.head = nn.Linear(embedding_dimensions, vocab_size, bias=False)
    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)  # (B, T, D)
        pos_emb = self.pos_embedding[:, :T, :]   # (1, T, D)
        x = token_emb + pos_emb
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

# ------------------ Training Function ------------------
def train_model(model, num_epochs=num_epochs):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=learning_rate_decay,
        patience=evaluation_patience,
        threshold=percent_for_plateau,
        threshold_mode='rel',
    )
    scaler = GradScaler()  # for mixed precision training
    writer = SummaryWriter(log_dir=tracking_address)
    
    previous_epoch_loss = None  # to track loss decay epoch-over-epoch
    start_time = time.time()
    global_step = 0
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        batch_count = 0
        print(f"\nStarting epoch {epoch}/{num_epochs}")
        
        # Training: iterate over all batches for the current epoch.
        for contexts, targets in get_training_batches():
            optimizer.zero_grad()
            with autocast():
                logits = model(contexts)  # (B, T, vocab_size)
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            
            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
        
        # Compute average training loss for this epoch.
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
        
        # Validation: evaluate the average loss over all validation batches.
        model.eval()
        val_losses = []
        for val_contexts, val_targets in get_validation_batches():
            with torch.no_grad():
                with autocast():
                    val_logits = model(val_contexts)
                    B_val, T_val, V_val = val_logits.shape
                    val_loss = F.cross_entropy(val_logits.view(B_val * T_val, V_val),
                                               val_targets.view(B_val * T_val))
                    val_losses.append(val_loss.item())
        model.train()
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        
        # Update the scheduler once per epoch using the average validation loss.
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Estimate the remaining time based on elapsed time and epochs.
        elapsed = time.time() - start_time
        avg_epoch_time = elapsed / epoch
        remaining_epochs = num_epochs - epoch
        eta = remaining_epochs * avg_epoch_time
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        
        # Log epoch summary.
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Epoch {epoch} Summary: "
              f"Train Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"LR: {current_lr:.6f}, ETA: {eta_str}")
        writer.add_scalar("Loss/Train_Epoch", avg_epoch_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)
        
        # Display percentage loss decay compared to the previous epoch.
        if previous_epoch_loss is not None:
            loss_decay_percent = ((previous_epoch_loss - avg_val_loss) / previous_epoch_loss) * 100.0
            print(f"Epoch {epoch} val loss decay: {loss_decay_percent:.2f}%")
        else:
            print("No previous epoch to compare for loss decay.")
        previous_epoch_loss = avg_val_loss
    
    writer.close()


# ------------------ Inference Function ------------------
@torch.no_grad()
def generate_text(model, idx, num_new_tokens = max_inference_tokens, temperature=0.8):
    model.eval()
    for _ in range(num_new_tokens):
        idx_cond = idx[:, -maximum_context_length:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx

#search for best hyperparams @ current scale...
def objective(trial):


# ------------------ Main ------------------
if __name__ == '__main__':
    model = TitansLMModel(
        vocabulary_size,
        embedding_dimensions,
        transformer_layers,
        maximum_context_length,
        memory_tokens,
        dropout=dropout,
        n_heads=head_count
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"TitansLMModel has {total_params/1e6:.2f}M parameters")
    
    train_model(model, num_epochs=num_epochs)
    
    # Inference: generate text starting from a random training context.
    start_idx = random.randint(0, len(training_data) - maximum_context_length - 1)
    # Get a context segment (shape: [1, maximum_context_length])
    context = training_data[start_idx:start_idx+maximum_context_length].unsqueeze(0).to(device)
    # Generate additional tokens (e.g., 256 tokens) appended to the context.
    generated = generate_text(model, context)
    # The full generated sequence has shape [1, maximum_context_length + 256]
    full_sequence = generated[0].tolist()

    # Split the sequence: the provided context and the generated text.
    provided_context_tokens = full_sequence[:maximum_context_length]
    generated_tokens = full_sequence[maximum_context_length:]

    provided_context_text = decode(provided_context_tokens)
    generated_text_only = decode(generated_tokens)

    # Print the output with an indicator between the context and generated text.
    print("Provided Context:")
    print(provided_context_text)
    print("\n=== GENERATED TEXT BELOW ===")
    print(generated_text_only)
