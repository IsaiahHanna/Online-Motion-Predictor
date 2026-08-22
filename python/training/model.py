""""
model.py -  IntentPredictor Base Model
Online Human Intent Predictor with Adaptive Learning

Architecture: small temporal transformer
    Input  : [B, W=30, 99]  - 30 frames x 33 joints x 3 coords
    Output : [B, K=15, 99]  - 15 predicted future frames

Design Decisions
----------------
* Small on purpose (hidden=128, 4 layers): fast inference is the goal,
  not benchmark SOTA. A < 5ms inference time beats a slower larger model
  for the real-time C++ pipeline that follows.

* Learned positional encoding (nn.Parameter) rather than fixed sinusoidal:
  Torchscript handles it cleanly and it adds no compute overhead.

  
* Last-timestep pooling (x[:, -1, :]): the final encoder token sees the 
  full causal history via self-attention, making it a natural summary.

* LoRAAdapter stub included: validates the adapter insertion logic in Python
  before replicating it in C++.

"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# LoRA Adapter (Python reference implementation for later phases)
# ---------------------------------------------------------------------------

class LoRAAdapter(nn.Module):
    """"
    Bottleneck residual adapter: x -> W_down -> GELU -> W_up -> residual.

    Initialized so W_up = 0, meaning the adapter starts as an exact
    identity (zero residual). The base model's behavior is unchanged
    at the start of online adaptation.

    Parameter count per adapter:
        hidden_dim * r   (W_down)
      + r * hidden_dim   (W_up)
      = 2 * hidden_dim * r

    For hidden=128, r=4: 1024 params per adapter
    """

    def __init__(self, hidden_dim: int, rank: int = 4):
        super().__init__()
        self.W_down = nn.Linear(hidden_dim, rank, bias=False)
        self.W_up   = nn.Linear(rank, hidden_dim, bias=False)

        # Standard LoRA init: W_down ~ N(0, 1/sqrt(rank)), w_up = 0
        nn.init.kaiming_uniform_(self.W_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_up.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, hidden] -> residual same shape
        return self.W_up(F.gelu(self.W_down(x)))
    
    def reset(self) -> None:
        """Reset to identity (zero residual) - used when switching subjects"""
        nn.init.zeros_(self.W_up.weight)


# ---------------------------------------------------------------------------
# Adapted Encoder Layer (wraps standard TransformerEncoderLayer + adapters)
# ---------------------------------------------------------------------------

class AdaptedEncoderLayer(nn.Module):
    """
    Wraps a nn.TransformerEncoderLayer and adds two LoRA adapters:
        - after the seelf-attention sublayer output
        - after the feedforward sublayer output
    
    Used in later phases to validate adapter behavior in Python before
    replicating the insertion logic in C++/libtorch.
    """

    def __init__(self, base_layer: nn.TransformerEncoderLayer, rank:int = 4):
        super().__init__()
        hidden_dim = base_layer.self_attn.embed_dim
        self.base = base_layer
        self.attn_adapter = LoRAAdapter(hidden_dim, rank)
        self.ff_adapter   = LoRAAdapter(hidden_dim, rank)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        # Replicate TransformerEncoderLayer internals to insert adapters
        # at the right points (after each sublayer, before layer norm).
        # We call the base layer directly to avoid double-applying norms.
        x = src

        # Self-attention sublayer
        attn_out = self.base.self_attn(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )[0]
        attn_out = attn_out + self.attn_adapter(attn_out)  # <- adapter
        x = self.base.norm1(x + self.base.dropout(attn_out))

        # Feedforward sublayer
        ff_out = self.base.linear2(
            self.base.dropout(self.base.activation(self.base.linear1(x)))
        )
        ff_out = ff_out + self.ff_adapter(ff_out)
        x = self.base.norm2(x + self.base.dropout(ff_out))

        return x


# ---------------------------------------------------------------------------
# Base Model
# ---------------------------------------------------------------------------

class IntentPredictor(nn.Module):
    """
    Temporal transformer for human motion prediction.

    Input  : x - [B, W, in_dim]  (W=30, in_dim=99)
    Output :     [B, K, in_dim]  (K=15)

    TorchScript-compatible: no dynamic control flow on tensor dims,
    all shapes are statically inferable. Verified with torch.jit.script().
    """

    def __init__(
        self, 
        in_dim:    int = 99,
        hidden:    int = 128,
        n_heads:   int = 4,
        n_layers:  int = 4,
        W:         int = 30,
        K:         int = 15,
        dropout:   float = 0.1
    ):
        super().__init__()
        self.K      = K
        self.in_dim = in_dim
        self.hidden = hidden

        # Input projection: 99 -> hidden
        self.proj = nn.Linear(in_dim, hidden)

        # Learned positional encoding over W time steps
        self.pos = nn.Parameter(torch.randn(1, W, hidden) * 0.02)

        # Transformer encoder stack
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,    # [B, T, D] convention throughout
            norm_first=False     # post-LN (standard)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Output head: hidden -> K * in_dim
        self.head = nn.Linear(hidden, K * in_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """ Xavier init for projection and head; leave encoder at default."""
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _encode_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input sequence and return the final-timestep hidden vector.

        x      : [B, W, 99]
        returns: [B, hidden]
        """
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        return x[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, W, 99]
        returns : [B, K, 99]
        """
        # Project + add positional encoding
        x = self.proj(x) + self.pos            # [B, W, hidden]

        # Encode full sequence
        x = self.encoder(x)                    # [B, W, hidden]

        # Pool at the final timestep - sees full history via attention
        x = x[:, -1, :]                        # [B, hidden]

        # Decode to K future frames
        x = self.head(x)                       # [B, K * in_dim]
        return x.view(-1, self.K, self.in_dim) # [B, K, 99]

    @torch.jit.export
    def forward_with_hidden(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode_hidden(x)

        pred = self.head(hidden)
        pred = pred.view(-1, self.K, self.in_dim)

        return pred, hidden
        
    
    # ---------------------------------------------------------------------------
    # Adapter insertion (prep for later phases)
    # ---------------------------------------------------------------------------

    def insert_adapters(self, n_adapt: int = 2, rank: int = 4) -> List[LoRAAdapter]:
        """
        Replace the last *n_adapt* encoder layers with AdaptedEncoderLayer
        wrappers. Freezes all base parameters first.

        Call this AFTER loading a trained checkpoint and BEFORE online deployment.
        Returns a flat list of all LoRAAdapter modules so you can pass their 
        parameters directly to an optimizer.

        Example 
        -------
        model.load_state_dict(torch.load('checkpoint.pt'))
        adapters = model.insert_adapters(n_adapt=2, rank=4)
        optimizer = torch.optim.Adam(
            [p for a in adapters for p in a.parameters()], lr=1e-4
        )
        """
        # Freeze all base weights
        for p in self.parameters():
            p.requires_grad = False
        
        layers: List[nn.Module] = list(self.encoder.layers)
        n = len(layers)
        if n_adapt > n:
            raise ValueError(f"n_adapt={n_adapt} > n_layers={n}")
        
        adapter_list: List[LoRAAdapter] = []
        for i in range(n - n_adapt, n):
            adapted = AdaptedEncoderLayer(layers[i], rank = rank)
            # Unfreeze only the adapter parameters
            for p in adapted.attn_adapter.parameters():
                p.requires_grad = True
            for p in adapted.ff_adapter.parameters():
                p.requires_grad = True
            layers[i] = adapted
            adapter_list.extend([adapted.attn_adapter, adapted.ff_adapter])
        
        # Re-assign the encoder's layer list 
        self.encoder.layers = nn.ModuleList(layers)
        return adapter_list
    
    def freeze_base(self) -> None:
        """Freeze everything; call before insert_adapters if you want explicit control"""
        for p in self.parameters():
            p.requires_grad = False
    
    def count_parameters(self) -> dict:
        """Return a dict with total, trainable, and frozen param counts"""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total":     total,
            "trainable": trainable,
            "frozen":    total - trainable
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def build_model(cfg: dict) -> IntentPredictor:
    """
    Build an IntentPredictor from a config dict.
    Matches the keys used in configs/train.yaml.

    cfg keys (all optional, fall back to defaults):
        in_dim, hidden, n_heads, n_layers, W, K, dropout
    """
    return IntentPredictor(
        in_dim=cfg.get("in_dim", 99),
        hidden=cfg.get("hidden", 128),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 4),
        W=cfg.get("W", 30),
        K=cfg.get("K", 15),
        dropout=cfg.get("dropout", 0.1)
    )


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly to sanity-check)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("IntentPredictor - architecture self-test")
    print("=" * 55)

    model = IntentPredictor(hidden=128, n_layers=4)
    model.eval()

    # Forward pass
    x    = torch.randn(4,30,99)
    pred = model(x)
    assert pred.shape == (4, 15, 99), f"Unexpected output shape: {pred.shape}"
    print(f"Forward pass:     {x.shape} -> {pred.shape}")

    # Parameter count
    counts = model.count_parameters()
    print(f"Total params:    {counts['total']:,}")
    print(f"Trainable:       {counts['trainable']:,}")

    # TorchScript export
    scripted = torch.jit.script(model)
    pred_scripted = scripted(x)
    diff = (pred - pred_scripted).abs().max().item()
    assert diff < 1e-5, f"TorchScript output differs: max_diff = {diff}"
    print(f"TorchScript:   max output diff vs eager = {diff:.2e}")

    # Adapter insertion
    model.train()
    adapters = model.insert_adapters(n_adapt=2, rank=4)
    counts_adapted = model.count_parameters()
    print(f"\nAfter insert_adapters(n_adapt=2, rank=4):")
    print(f"    Trainable params: {counts_adapted['trainable']}:,"
          f"({counts_adapted['trainable']/counts_adapted['total']*100:.2f}% of total)")
    print(f"    Adapter Modules:  {len(adapters)}")

    # Verify only adapter params have gradients
    base_grad = [p for n,p in model.named_parameters()
                 if p.requires_grad and "adapter" not in n]
    assert len(base_grad) == 0, f"{len(base_grad)} non-adapter params still trainable"
    print(f"    Base weights frozen.")

    # Gradient flows through adapters
    x2   = torch.randn(2, 30, 99)
    out  = model(x2)
    loss = out.mean()
    loss.backward()
    adapter_has_grad = all(
        p.grad is not None for a in adapters for p in a.parameters()
    )
    assert adapter_has_grad, "Adapter parameters have no gradient"
    print(f"    Adapter grads flow successful.")

    # Adapter reset
    for a in adapters:
        a.reset()
    print(f"    Adapter reset passes.")

    print("\n All self-tests passed.")