import pytest
import torch
from saccade.models import SaccadeViT
from saccade.benchmarks.faithfulness import verify_causal_gradient_isolation, compute_faithfulness_curves


def test_verify_causal_gradient_isolation():
    model = SaccadeViT.from_pretrained(
        "native",
        img_size=64,
        patch_size=16,
        num_classes=4,
        top_k_patches=4,
        bottleneck_layer=1,
        embed_dim=32,
        depth=3,
        num_heads=2
    )
    x = torch.randn(1, 3, 64, 64)
    res = verify_causal_gradient_isolation(model, x, top_k=4)

    assert res["active_patches_count"] == 4
    assert res["discarded_patches_count"] == 12  # 16 total - 4
    assert res["exact_zero_leakage_guarantee"] is True
    assert res["max_discarded_patch_grad_norm"] == 0.0


def test_compute_faithfulness_curves():
    model = SaccadeViT.from_pretrained(
        "native",
        img_size=64,
        patch_size=16,
        num_classes=3,
        top_k_patches=4,
        bottleneck_layer=1,
        embed_dim=32,
        depth=3,
        num_heads=2
    )
    x = torch.randn(1, 3, 64, 64)
    curves = compute_faithfulness_curves(model, x, num_steps=5)

    assert "auc_deletion" in curves
    assert "auc_insertion" in curves
    assert len(curves["deletion_probs"]) == len(curves["steps"])
    assert len(curves["insertion_probs"]) == len(curves["steps"])

