import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# ------------------ Model Components ------------------

class Head(nn.Module):
    def __init__(self, embedding_dim, head_size, dropout_rate=0.2):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        B, T, _ = x.shape
        keys = self.key(x)      # (B, T, head_size)
        queries = self.query(x) # (B, T, head_size)
        values = self.value(x)  # (B, T, head_size)
        # Scale dot-product attention.
        weights = torch.matmul(queries, keys.transpose(-2, -1)) * (keys.shape[-1] ** -0.5)  # (B, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        weights = weights.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, values)  # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_rate=0.2):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList(
            [Head(embedding_dim, head_size, dropout_rate) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Concatenate outputs from all heads.
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return self.net(x)

# TitansBlock implements the Memory As Context (MAC) architecture.
class TitansBlock(nn.Module):
    def __init__(self, embedding_dim, persistent_memory, num_heads, dropout_rate=0.2):
        super().__init__()
        self.persistent_memory = persistent_memory  # Shared persistent memory tokens.
        self.sa = MultiHeadAttention(embedding_dim, num_heads, dropout_rate)
        self.ffwd = FeedForward(embedding_dim, dropout_rate)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        B, T, D = x.shape
        M = self.persistent_memory.shape[0]  # Number of memory tokens.
        # Expand persistent memory for the batch.
        mem = self.persistent_memory.unsqueeze(0).expand(B, M, D)  # (B, M, D)
        # Prepend memory to the input.
        x_aug = torch.cat([mem, x], dim=1)  # (B, M+T, D)
        x_norm = self.ln1(x_aug)
        attn_out = self.sa(x_norm)
        x_aug = x_aug + attn_out    # Residual connection.
        # Discard memory tokens for the feedforward step.
        x_text = x_aug[:, M:, :]   # (B, T, D)
        x_text = x_text + self.ffwd(self.ln2(x_text))
        return x_text

# Wrap the block with gradient checkpointing for memory efficiency.
class CheckpointedTitansBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    
    def forward(self, x):
        return checkpoint.checkpoint(self.block, x, use_reentrant=False)

# Main language model using the MAC architecture.
class TitansLMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, max_context_length,
                 memory_tokens, dropout_rate=0.2, num_heads=8):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_context_length, embedding_dim))
        self.dropout = nn.Dropout(dropout_rate)
        # Initialize persistent memory tokens.
        self.persistent_memory = nn.Parameter(torch.randn(memory_tokens, embedding_dim))
        # Create a stack of transformer blocks (with gradient checkpointing).
        self.blocks = nn.ModuleList([
            CheckpointedTitansBlock(
                TitansBlock(embedding_dim, self.persistent_memory, num_heads, dropout_rate)
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size, bias=False)
    
    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)   # (B, T, embedding_dim)
        pos_emb = self.pos_embedding[:, :T, :]    # (1, T, embedding_dim)
        x = token_emb + pos_emb
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits
