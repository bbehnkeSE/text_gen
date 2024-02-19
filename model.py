import torch
import torch.nn                   as nn
from   torch.nn import functional as F


class Head(nn.Module):
    """
    One head of self-attention
    """
    def __init__(self, head_size, n_embed, block_size, dropout, masked=False):
        super().__init__()
        self.masked  = masked
        self.key     = nn.Linear(n_embed, head_size, bias=False)
        self.query   = nn.Linear(n_embed, head_size, bias=False)
        self.value   = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, C)
        q = self.query(x)  # (B, T, C)

        # Compute scaled attention scores
        W = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, C) @ (B, C, T) = (B, T, T)
        if(self.masked):
            W = W.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        W = F.softmax(W, dim=-1)  # (B, T, T)
        W = self.dropout(W)

        # Perform weighted aggregation of the values
        V = self.value(x)  # (B, T, C)
        out = W @ V  # (B, T, T) @ (B, T, C) = (B, T, C)

        return out
    

class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention
    """
    def __init__(self, num_heads, head_size, n_embed, block_size, dropout, masked=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout, masked) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concat over channel dim
        out = self.projection(out)
        out = self.dropout(out)
        
        return out
    

class FeedForward(nn.Module):
    """
    Linear layer followed by non-linearity
    """
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),  # Projection layer going back into residual pathway
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_embed, num_heads, head_size, block_size, dropout, masked=False):
        super().__init__()
        self.sa  = MultiHeadAttention(num_heads, head_size, n_embed, block_size, dropout, masked)
        self.ff  = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # (B, T, C)
        x = x + self.ff(self.ln2(x))  # (B, T, C)  # Allows each token to "think" on the data from SA

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, head_size, num_heads, n_layer, dropout, device, masked=True):
        super().__init__()
        self.block_size = block_size
        self.device     = device

        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head                  = nn.Linear(n_embed, vocab_size)
        self.ln_f                     = nn.LayerNorm(n_embed)  # Final layer normalization
        self.blocks                   = nn.Sequential(*[Block(n_embed, num_heads, head_size, block_size, dropout, masked) 
                                                        for _ in range(n_layer)])

    def forward(self, idx, targets=None):
        B, T                = idx.shape

        # idx and targets are both (B, T) tensors of integers
        token_embeddings    = self.token_embedding_table(idx)  # (B, T, C)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=self.device))  # (T, C)
        x                   = token_embeddings + position_embeddings  # (B, T, C)
        x                   = self.blocks(x)  # (B, T, C)
        x                   = self.ln_f(x)  # (B, T, C)
        logits              = self.lm_head(x)  # (B, T, vocab_size)

        # PyTorch wants C in the second dimension, so we stretch the logits into
        # a 2D tensor and the targets into a 1D tensor
        if targets is None:
            loss    = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss    = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond     = idx[:, -self.block_size:]  # idx can't be > block_size or pos_embed will go out of scope
            logits, _    = self(idx_cond)  # Get predictions
            logits       = logits[:, -1, :]  # Focus on the last time step  # Becomes (B, C)
            probs        = F.softmax(logits, dim=-1)  # (B, C)
            idx_next     = torch.multinomial(probs, num_samples=1)  # (B, 1)  # Sample from distrobution
            idx          = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)  # Append sampled index to the running sequence
        
        return idx
    

class CharTokenizer():
    def __init__(self, char_list):
        self.char_list  = char_list
        self.vocab_size = len(char_list)
        self.stoi = {ch:i for i, ch in enumerate(char_list)}
        self.itos = {i:ch for i, ch in enumerate(char_list)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, s):
        return "".join([self.itos[i] for i in s])