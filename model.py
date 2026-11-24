import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) per codificare distanze relative senza inquinare token_emb."""

    def __init__(self, dim: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "d_head deve essere pari per RoPE"
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (max_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_len, dim)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self, seq_len: int, dtype: torch.dtype, device: torch.device):
        if seq_len > self.cos_cached.size(2):
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(device=device, dtype=dtype),
            self.sin_cached[:, :, :seq_len, :].to(device=device, dtype=dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_len: int = 320,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve essere divisibile per n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.d_head, max_len=max_len)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Applica RoPE solo a Q e K (non a V)
        cos, sin = self.rope(T, dtype=q.dtype, device=q.device)
        q, k = apply_rope(q, k, cos, sin)

        if not return_attention:
            sdpa_mask = (
                (attention_mask == 1).unsqueeze(1).unsqueeze(2)
                if attention_mask is not None
                else None
            )
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
            attn_probs = None
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
            if attention_mask is not None:
                key_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(key_mask, torch.finfo(scores.dtype).min)
            attn_probs = F.softmax(scores, dim=-1)
            attn_dropped = self.attn_dropout(attn_probs)
            out = torch.matmul(attn_dropped, v)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.resid_dropout(self.out_proj(out))

        return out, attn_probs


class TransformerBlock(nn.Module):
    """Blocco Transformer Pre-LN standard con RoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_len: int = 320,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model, n_heads, max_len=max_len, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_probs = self.attn(
            self.ln1(x),
            attention_mask=attention_mask,
            return_attention=return_attention,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_probs


class LogicTransformer(nn.Module):

    def __init__(
        self,
        vocab_size: int = 66,
        max_len: int = 320,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1024,
        num_classes: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_layers = n_layers

        # Nessun pos_emb assoluto: usiamo RoPE in ogni layer
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, max_len=max_len, dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        resid_std = 0.02 / math.sqrt(2 * self.n_layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Scala i pesi di uscita residui per stabilizzare reti profonde from-scratch
        for layer in self.layers:
            nn.init.normal_(layer.attn.out_proj.weight, mean=0.0, std=resid_std)
            nn.init.normal_(layer.ffn[3].weight, mean=0.0, std=resid_std)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb_dropout(self.token_emb(input_ids))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
    ) -> Dict[str, object]:
        x = self.embed(input_ids)

        hidden_states: List[torch.Tensor] = [x] if return_hidden_states else []
        attentions: List[torch.Tensor] = []

        for layer in self.layers:
            x, attn_probs = layer(
                x,
                attention_mask=attention_mask,
                return_attention=return_attentions,
            )
            if return_hidden_states:
                hidden_states.append(x)
            if return_attentions and attn_probs is not None:
                attentions.append(attn_probs)

        x_norm = self.final_ln(x)
        cls_repr = x_norm[:, 0, :]
        logits = self.classifier(cls_repr)

        out: Dict[str, object] = {"logits": logits}
        if return_hidden_states:
            out["hidden_states"] = hidden_states
        if return_attentions:
            out["attentions"] = attentions
        return out

    def forward_from_layer(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_layer: int = 4,
    ) -> torch.Tensor:
        x = hidden
        for layer in self.layers[start_layer:]:
            x, _ = layer(x, attention_mask=attention_mask, return_attention=False)
        x_norm = self.final_ln(x)
        return self.classifier(x_norm[:, 0, :])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LogicTransformer()
    print(f"Parametri totali: {count_parameters(model):,}")

    model.eval()
    B, T = 4, 64
    dummy_ids = torch.randint(0, 66, (B, T))
    dummy_mask = torch.ones((B, T), dtype=torch.long)
    dummy_mask[:, -10:] = 0

    with torch.no_grad():
        out_fast = model(dummy_ids, dummy_mask, return_attentions=False)
        out = model(
            dummy_ids,
            dummy_mask,
            return_hidden_states=True,
            return_attentions=True,
        )
        logits = out["logits"]
        h_states = out["hidden_states"]
        attns = out["attentions"]

        stitched_logits = model.forward_from_layer(
            h_states[4], dummy_mask, start_layer=4
        )
        diff = (logits - stitched_logits).abs().max().item()
        sdpa_diff = (logits - out_fast["logits"]).abs().max().item()

    print(f"Logits shape: {logits.shape}")
    print(f"Numero hidden states: {len(h_states)} (emb + {model.n_layers} layers)")
    print(f"Attentions shape (layer 0): {attns[0].shape}")
    print(f"Differenza max forward vs forward_from_layer(4): {diff:.2e}")
    print(f"Differenza max SDPA vs Manuale: {sdpa_diff:.2e}")