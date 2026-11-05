import pytest
import torch
from saccade.models import SaccadeViT, CleanViT


def test_clean_vit_instantiation():
    model = CleanViT(
        img_size=64,
        patch_size=16,
        in_chans=3,
        num_classes=10,
        embed_dim=32,
        depth=2,
        num_heads=2
    )
    x = torch.randn(2, 3, 64, 64)
    # Patch tokens: (64/16)*(64/16) = 16 patches
    tokens = model.patch_embed(x).flatten(2).transpose(1, 2)
    assert tokens.shape == (2, 16, 32)


def test_saccade_vit_native_forward():
    model = SaccadeViT.from_pretrained(
        "native",
        img_size=64,
        patch_size=16,
        num_classes=5,
        top_k_patches=4,
        bottleneck_layer=1,
        embed_dim=32,
        depth=3,
        num_heads=2
    )
    x = torch.randn(2, 3, 64, 64)
    logits, meta = model(x, return_saccades=True)
    assert logits.shape == (2, 5)
    assert meta.patch_indices.shape == (2, 4)
    assert meta.grid_size == (4, 4)
    assert len(meta.bounding_boxes) == 2


def test_saccade_vit_dynamic_budget():
    model = SaccadeViT.from_pretrained(
        "native",
        img_size=64,
        patch_size=16,
        num_classes=5,
        top_k_patches=4,
        bottleneck_layer=1,
        embed_dim=32,
        depth=3,
        num_heads=2
    )
    x = torch.randn(1, 3, 64, 64)
    # Override K to 8
    logits, meta = model(x, return_saccades=True, override_top_k=8)
    assert meta.patch_indices.shape == (1, 8)
    assert meta.top_k == 8


def test_saccade_vit_sparsity_trajectory():
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
    traj = model.compute_sparsity_trajectory(x, k_values=[2, 4, 8, 16])
    assert len(traj) == 4
    for pt in traj:
        assert "k" in pt
        assert "top1_class" in pt
        assert "top1_prob" in pt


def test_saccade_vit_counterfactual_ablations():
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
    cf = model.compute_counterfactual_ablations(x, top_k=4)
    assert "base_class" in cf
    assert len(cf["active_patches"]) == 4
    assert len(cf["ablations"]) == 4
    for p_idx in cf["active_patches"]:
        assert p_idx in cf["ablations"]
        assert "delta_logit" in cf["ablations"][p_idx]

