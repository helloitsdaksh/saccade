import pytest
import torch
from saccade.bottleneck import (
    compute_patch_importance,
    patch_indices_to_bboxes,
    SaccadeBottleneck,
    SaccadeMetadata,
)


def test_compute_patch_importance_col_sum():
    # Batch=1, heads=1, tokens=5 (1 CLS + 4 spatial patches)
    # Patch 3 gets high attention from all spatial tokens
    attn = torch.zeros(1, 1, 5, 5)
    # Set col 3 (which corresponds to spatial patch 2 in spatial indices [1, 2, 3, 4])
    attn[0, 0, 1:, 3] = 0.8
    attn[0, 0, 1:, 1] = 0.2

    importance = compute_patch_importance(attn, has_cls_token=True)
    assert importance.shape == (1, 4)
    # Patch 2 (index 2 in 0..3) should have highest column sum
    assert torch.argmax(importance[0]).item() == 2


def test_patch_indices_to_bboxes():
    grid_size = (14, 14)
    # Indices 0 (top-left: row 0, col 0) and 195 (bottom-right: row 13, col 13)
    patch_indices = torch.tensor([[0, 195]])
    bboxes = patch_indices_to_bboxes(patch_indices, grid_size)
    assert len(bboxes) == 1
    assert len(bboxes[0]) == 2
    
    # Check top-left box
    y0, x0, y1, x1 = bboxes[0][0]
    assert y0 == 0.0 and x0 == 0.0
    assert pytest.approx(y1) == 1.0 / 14.0
    assert pytest.approx(x1) == 1.0 / 14.0

    # Check bottom-right box
    y0, x0, y1, x1 = bboxes[0][1]
    assert pytest.approx(y0) == 13.0 / 14.0
    assert pytest.approx(x0) == 13.0 / 14.0
    assert pytest.approx(y1) == 1.0
    assert pytest.approx(x1) == 1.0


def test_saccade_bottleneck_forward():
    bottleneck = SaccadeBottleneck(top_k=4, grid_size=(4, 4), has_cls_token=True, layer_idx=3)
    # Tokens: [B=2, S=17, D=32] (1 CLS + 16 patches)
    tokens = torch.randn(2, 17, 32)
    # Uniform attention
    attn = torch.ones(2, 17, 17) / 17.0

    pruned_tokens, meta = bottleneck(tokens, attn, return_meta=True)
    # Pruned tokens should retain CLS + top-4 patches = 5 tokens
    assert pruned_tokens.shape == (2, 5, 32)
    assert isinstance(meta, SaccadeMetadata)
    assert meta.patch_indices.shape == (2, 4)
    assert meta.top_k == 4
    assert meta.layer_idx == 3

