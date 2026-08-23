"""
export_adapted_model.py — Attention-Layer Adapter Export
Online Human Intent Predictor with Adaptive Learning
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model import IntentPredictor, build_model


# ---------------------------------------------------------------------------
# LoRAAdapter
# ---------------------------------------------------------------------------
class LoRAAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int):
        super().__init__()
        self.W_down = nn.Linear(hidden_dim, rank, bias=False)
        self.W_up   = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(self.W_down.weight)
        nn.init.zeros_(self.W_up.weight)
        self.rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.W_down(x))
        return self.W_up(h)


# ---------------------------------------------------------------------------
# AdaptedEncoderLayer
# ---------------------------------------------------------------------------
class AdaptedEncoderLayer(nn.Module):
    def __init__(self, base_layer: nn.TransformerEncoderLayer, rank: int):
        super().__init__()
        self.self_attn  = base_layer.self_attn
        self.linear1    = base_layer.linear1
        self.linear2    = base_layer.linear2
        self.norm1      = base_layer.norm1
        self.norm2      = base_layer.norm2
        self.dropout    = base_layer.dropout
        self.activation = base_layer.activation
        hidden_dim = base_layer.self_attn.embed_dim
        self.adapter_attn = LoRAAdapter(hidden_dim, rank)
        self.adapter_ff   = LoRAAdapter(hidden_dim, rank)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        attn_out = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]
        x = self.norm1(src + self.dropout(attn_out) + self.adapter_attn(attn_out))

        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_out) + self.adapter_ff(ff_out))
        return x


# ---------------------------------------------------------------------------
# AdaptedIntentPredictor
# ---------------------------------------------------------------------------
class AdaptedIntentPredictor(nn.Module):
    def __init__(self, base_model: IntentPredictor, rank: int, n_adapt: int):
        super().__init__()
        self.proj    = base_model.proj
        self.pos     = base_model.pos
        self.encoder = base_model.encoder
        self.head    = base_model.head
        self.K       = base_model.K
        self.in_dim  = base_model.in_dim

        layers = list(self.encoder.layers)
        n = len(layers)
        for i in range(n - n_adapt, n):
            layers[i] = AdaptedEncoderLayer(layers[i], rank)
        self.encoder.layers = nn.ModuleList(layers)

        for p in self.parameters():
            p.requires_grad = False
        for name, p in self.named_parameters():
            if "adapter" in name:
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.head(x)
        return x.view(-1, self.K, self.in_dim)

    def forward_with_hidden(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        hidden = x[:, -1, :]           # [B, hidden_dim]
        pred   = self.head(hidden)
        pred   = pred.view(-1, self.K, self.in_dim)
        return pred, hidden

    def count_adapter_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export(args: argparse.Namespace) -> None:
    device = torch.device("cpu")

    # 1. Load base checkpoint
    cfg = {
        "in_dim":   99,
        "hidden":   args.hidden,
        "n_heads":  4,
        "n_layers": 4,
        "W":        30,
        "K":        15,
        "dropout":  0.0,
    }
    base_model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    base_model.load_state_dict(state)
    base_model.eval()

    # 2. Wrap with adapters
    adapted = AdaptedIntentPredictor(base_model, args.rank, args.n_adapt)
    adapted.eval()

    # 3. Verify adapter parameter setup
    total_params   = sum(p.numel() for p in adapted.parameters())
    adapter_params = adapted.count_adapter_params()
    adapter_pct    = adapter_params / total_params * 100

    print(f"Total params:   {total_params:,}")
    print(f"Adapter params: {adapter_params:,}  ({adapter_pct:.3f}% of total)")
    assert adapter_pct < 2.0, f"Adapter params {adapter_pct:.3f}% exceed 2% budget"

    non_adapter_trainable = [
        name for name, p in adapted.named_parameters()
        if p.requires_grad and "adapter" not in name
    ]
    assert len(non_adapter_trainable) == 0, \
        f"Non-adapter params are trainable: {non_adapter_trainable}"
    print("Adapter names:")
    for name, p in adapted.named_parameters():
        if p.requires_grad:
            print(f"  {name}  ({p.numel()} params)")

    # 4. Identity residual check
    x_test = torch.randn(1, 30, 99)
    with torch.no_grad():
        out_base    = base_model(x_test)
        out_adapted = adapted(x_test)

    max_diff = (out_base - out_adapted).abs().max().item()
    assert max_diff < 1e-5, f"Identity check failed: max_diff={max_diff:.2e}"
    print(f"Identity check passed (max_diff={max_diff:.2e})")

    # 5. TorchScript export
    scripted = torch.jit.script(adapted)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved TorchScript to {output_path}  ({size_mb:.1f} MB)")

    # 6. Round-trip verification
    reloaded   = torch.jit.load(str(output_path))
    out_script = reloaded(x_test)
    rt_diff = (out_base - out_script).abs().max().item()
    assert rt_diff < 1e-4, f"Round-trip failed: max_diff={rt_diff:.2e}"
    print(f"Round-trip verified (max_diff={rt_diff:.2e})")

    # 7. Save test I/O for C++ verification
    test_io_path = Path(args.test_io)
    test_io_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x_test, "y": out_script}, test_io_path)
    print(f"Test I/O saved to {test_io_path}")

    # 8. Summary
    print("\n" + "=" * 55)
    print("EXPORT COMPLETE")
    print("=" * 55)
    print(f"  Rank r          : {args.rank}")
    print(f"  Layers adapted  : {args.n_adapt}")
    print(f"  Adapter params  : {adapter_params:,}  ({adapter_pct:.3f}%)")
    print(f"  Output          : {output_path}")
    print(f"  Test I/O        : {test_io_path}")
    print("=" * 55)
    print("\nNext: scp adapted_model.pt and adapted_test_io.pt to the VM")
    print("Then rebuild and run with adapted_model.pt instead of intent_model.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export IntentPredictor with attention-layer LoRA adapters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default="models/checkpoints/best.pt")
    p.add_argument("--output",     default="models/torchscript/adapted_model.pt")
    p.add_argument("--test_io",    default="models/torchscript/adapted_test_io.pt")
    p.add_argument("--rank",    type=int, default=4)
    p.add_argument("--n_adapt", type=int, default=2)
    p.add_argument("--hidden",  type=int, default=128)
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())