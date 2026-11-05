"""
saccade - Faithfulness Benchmark (AUC-Del & AUC-Ins)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluates Deletion and Insertion curves comparing intrinsic Saccade attention
ranking against random token masking.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from saccade import SaccadeViT
from saccade.benchmarks.faithfulness import compute_faithfulness_curves


def main():
    print("=" * 65)
    print("saccade // Faithfulness Evaluation Benchmark")
    print("Deletion vs Insertion Area Under Curve (AUC)")
    print("=" * 65)

    model = SaccadeViT.from_pretrained(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=10,
        top_k_patches=16,
        bottleneck_layer=6,
    )
    model.eval()

    # Synthetic input
    x = torch.randn(1, 3, 224, 224)

    print("\nComputing Deletion and Insertion curves across token budget steps...")
    curves = compute_faithfulness_curves(model, x, num_steps=12)

    print(f"\n--- Faithfulness Results ---")
    print(f"Base Target Class: {curves['target_class']}")
    print(f"Base Probability: {curves['base_prob']:.4f}")
    print(f"AUC Deletion (Lower is better): {curves['auc_deletion']:.4f}")
    print(f"AUC Insertion (Higher is better): {curves['auc_insertion']:.4f}")

    # Plotting
    assets_dir = Path(__file__).parent.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig_path = assets_dir / "faithfulness_curves.png"

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0B0E14")
    ax.set_facecolor("#12161F")

    norm_steps = curves["norm_steps"]
    del_probs = curves["deletion_probs"]
    ins_probs = curves["insertion_probs"]

    ax.plot(norm_steps, del_probs, label=f"Deletion Curve (AUC={curves['auc_deletion']:.3f})", color="#FF3366", linewidth=2.5, marker="o")
    ax.plot(norm_steps, ins_probs, label=f"Insertion Curve (AUC={curves['auc_insertion']:.3f})", color="#00FFCC", linewidth=2.5, marker="s")

    ax.set_xlabel("Fraction of Patches Masked / Inserted", color="white", fontsize=11)
    ax.set_ylabel("Target Class Probability", color="white", fontsize=11)
    ax.set_title("Faithfulness Benchmark: Deletion & Insertion Curves", color="white", fontsize=13, pad=12)
    ax.grid(True, linestyle="--", alpha=0.2, color="#8E9BAE")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#222938")
    ax.legend(facecolor="#12161F", edgecolor="#222938", labelcolor="white", fontsize=10)

    plt.tight_layout()
    fig.savefig(str(fig_path), facecolor=fig.get_facecolor(), dpi=200)
    print(f"\nSaved Faithfulness Curve plot to: {fig_path}")


if __name__ == "__main__":
    main()

