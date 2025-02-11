import datetime
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

# Import the model from models.py
from models import TitansLMModel
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

# Training hyperparameters.
learning_rate = 3e-4
optimizer_iters = 5
dropout = 0.2
learning_rate_decay = 0.90
percent_for_plateau = 0.02
evaluation_patience = 0  # must be < num_epochs to use ReduceLROnPlateau

# ------------------ TensorBoard Setup ------------------
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

# Set device and seed.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(8181)

# ------------------ Batch Functions ------------------
def get_training_batches():
    n = len(training_data) - maximum_context_length - 1
    indices = list(range(0, n, maximum_context_length))
    random.shuffle(indices)
    for i in range(0, len(indices), parallel_batch_size):
        batch_indices = indices[i:i+parallel_batch_size]
        contexts = torch.stack(
            [training_data[idx:idx+maximum_context_length] for idx in batch_indices]
        ).to(device)
        targets = torch.stack(
            [training_data[idx+1:idx+maximum_context_length+1] for idx in batch_indices]
        ).to(device)
        yield contexts, targets

def get_validation_batches():
    n = len(validation_data) - maximum_context_length - 1
    indices = list(range(0, n, maximum_context_length))
    for i in range(0, len(indices), parallel_batch_size):
        batch_indices = indices[i:i+parallel_batch_size]
        contexts = torch.stack(
            [validation_data[idx:idx+maximum_context_length] for idx in batch_indices]
        ).to(device)
        targets = torch.stack(
            [validation_data[idx+1:idx+maximum_context_length+1] for idx in batch_indices]
        ).to(device)
        yield contexts, targets

# ------------------ Training Function ------------------
def train_model(model, num_epochs=num_epochs):
    model.train()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = SinkGD(model.parameters(), lr = learning_rate, num_iterations = optimizer_iters)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    writer = SummaryWriter(log_dir=tracking_address)
    
    previous_epoch_loss = None
    start_time = time.time()
    global_step = 0
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        batch_count = 0
        print(f"\nStarting epoch {epoch}/{num_epochs}")
        
        for contexts, targets in get_training_batches():
            optimizer.zero_grad()
            with autocast():
                logits = model(contexts)  # (B, T, vocab_size)
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
        
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
        
        # Validation
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
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        elapsed = time.time() - start_time
        avg_epoch_time = elapsed / epoch
        remaining_epochs = num_epochs - epoch
        eta = remaining_epochs * avg_epoch_time
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Epoch {epoch} Summary: Train Loss: {avg_epoch_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}, ETA: {eta_str}")
        writer.add_scalar("Loss/Train_Epoch", avg_epoch_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)
        
        if previous_epoch_loss is not None:
            loss_decay_percent = ((previous_epoch_loss - avg_val_loss) / previous_epoch_loss) * 100.0
            print(f"Epoch {epoch} val loss decay: {loss_decay_percent:.2f}%")
        else:
            print("No previous epoch to compare for loss decay.")
        previous_epoch_loss = avg_val_loss
    
    writer.close()

# ------------------ Inference Function ------------------
@torch.no_grad()
def generate_text(model, idx, num_new_tokens=max_inference_tokens, temperature=0.8):
    model.eval()
    for _ in range(num_new_tokens):
        idx_cond = idx[:, -maximum_context_length:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx

# ------------------ Main Script ------------------
if __name__ == '__main__':
    model = TitansLMModel(
        vocabulary_size,
        embedding_dimensions,
        transformer_layers,
        maximum_context_length,
        memory_tokens,
        dropout_rate=dropout,
        num_heads=head_count
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"TitansLMModel has {total_params/1e6:.2f}M parameters")
    
    train_model(model, num_epochs=num_epochs)
    
    # Inference: Generate text starting from a random training context.
    start_idx = random.randint(0, len(training_data) - maximum_context_length - 1)
    context = training_data[start_idx:start_idx+maximum_context_length].unsqueeze(0).to(device)
    generated = generate_text(model, context)
    full_sequence = generated[0].tolist()
    
    provided_context_tokens = full_sequence[:maximum_context_length]
    generated_tokens = full_sequence[maximum_context_length:]
    
    provided_context_text = decode(provided_context_tokens)
    generated_text_only = decode(generated_tokens)
    
    print("Provided Context:")
    print(provided_context_text)
    print("\n=== GENERATED TEXT BELOW ===")
    print(generated_text_only)
