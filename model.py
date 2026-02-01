from torch.nn.functional import dropout

from utils import SinusoidalPositionalEncoding

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # embedding size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # matrix of emb_size/3*emb_size -> WQ WK WV
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # B - batch size
        # T - sequence length
        # d_model - embdedding size
        B, T, _ = x.shape

        # x of size emb_dim
        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)

        # Split into Q, K, V
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D) H - heads, T - seq. length, D - head dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (B, H, T, T)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()  # (T, T)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, H, T, D)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.out_proj(out)





class NativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        T = x.size(1)

        # True = masked
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return out



class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        attn_type="custom",
        dropout=0.1,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if attn_type == "custom":
            self.attn = CustomMultiHeadAttention(d_model, n_heads, dropout)
        elif attn_type == "native":
            self.attn = NativeMultiHeadAttention(d_model, n_heads, dropout)
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            self.dropout
        )

    # pre-layer norm - different form original paper, supposedly works better
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        max_seq_len=512,
        dropout=0.1,
        attn_type="custom",
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = SinusoidalPositionalEncoding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                attn_type=attn_type,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        B, T = input_ids.shape

        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x += self.pos_emb(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits



def language_modeling_loss(logits, targets):
    """
    logits: (B, T, vocab_size)
    targets: (B, T)
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1), label_smoothing=0.1

    )
