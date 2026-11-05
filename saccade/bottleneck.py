"""
saccade.bottleneck
~~~~~~~~~~~~~~~~~~

Core intrinsic attention bottleneck modules implementing column-sum
patch-importance voting and hard top-K spatial pruning.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SaccadeMetadata:
    """Metadata detailing the causal spatial saccade bottleneck decision.
    
    Attributes:
        patch_indices: Selected patch indices of shape [B, K] in [0, N-1].
        patch_importance: Full voting scores for all N spatial patches [B, N].
        top_k: Number of preserved patches K.
        grid_size: Tuple (H_patches, W_patches).
        bounding_boxes: List of bounding boxes [ymin, xmin, ymax, xmax] in [0, 1] per batch item.
        layer_idx: Index of transformer block where the bottleneck was applied.
        attention_matrix: Optional averaged self-attention matrix [B, S, S].
    """
    patch_indices: torch.Tensor
    patch_importance: torch.Tensor
    top_k: int
    grid_size: Tuple[int, int]
    bounding_boxes: List[List[Tuple[float, float, float, float]]]
    layer_idx: int
    attention_matrix: Optional[torch.Tensor] = None

    def to(self, device: Union[str, torch.device]) -> "SaccadeMetadata":
        return SaccadeMetadata(
            patch_indices=self.patch_indices.to(device),
            patch_importance=self.patch_importance.to(device),
            top_k=self.top_k,
            grid_size=self.grid_size,
            bounding_boxes=self.bounding_boxes,
            layer_idx=self.layer_idx,
            attention_matrix=self.attention_matrix.to(device) if self.attention_matrix is not None else None,
        )


def compute_patch_importance(
    attention_matrix: torch.Tensor,
    has_cls_token: bool = True,
    cls_weight: float = 0.0,
) -> torch.Tensor:
    """Computes patch importance voting scores from multi-head self-attention.
    
    In David Ha's formulation (Tang et al., 2020), the column sum of the spatial
    self-attention affinity matrix represents the total 'votes' cast by all
    spatial patches for patch j:
    
        p_j = \\sum_{i=1}^N A_{i, j}
    
    Optionally, a weighted combination with [CLS]->patch attention can be blended:
        p_j = (1 - \\alpha) * \\sum_{i} A_{i, j} + \\alpha * A_{0, j}
    
    Args:
        attention_matrix: Tensor of shape [B, S, S] or [B, H, S, S] where S is sequence length.
        has_cls_token: Whether index 0 corresponds to a [CLS] token.
        cls_weight: Float in [0, 1], weight given to [CLS] query attention row vs column sum.
    
    Returns:
        patch_importance: Tensor of shape [B, N] where N is number of spatial patches.
    """
    # If heads dimension is present [B, H, S, S], average across heads
    if attention_matrix.ndim == 4:
        attn = attention_matrix.mean(dim=1)  # [B, S, S]
    elif attention_matrix.ndim == 3:
        attn = attention_matrix
    else:
        raise ValueError(f"Expected attention matrix with 3 or 4 dimensions, got {attention_matrix.shape}")

    if has_cls_token:
        # Spatial-to-spatial submatrix: rows [1:], cols [1:]
        spatial_attn = attn[:, 1:, 1:]  # [B, N, N]
        # Column sum over querying patches: sum over rows i
        col_sum_votes = spatial_attn.sum(dim=1)  # [B, N]
        
        if cls_weight > 0.0:
            # CLS query attention: row 0 attending to spatial patches [1:]
            cls_votes = attn[:, 0, 1:]  # [B, N]
            importance = (1.0 - cls_weight) * col_sum_votes + cls_weight * (cls_votes * spatial_attn.shape[1])
        else:
            importance = col_sum_votes
    else:
        # No CLS token: purely spatial tokens
        importance = attn.sum(dim=1)  # [B, N]

    return importance


def patch_indices_to_bboxes(
    patch_indices: torch.Tensor,
    grid_size: Tuple[int, int],
) -> List[List[Tuple[float, float, float, float]]]:
    """Converts 1D patch indices to normalized [ymin, xmin, ymax, xmax] bounding boxes.
    
    Args:
        patch_indices: Tensor of shape [B, K] containing spatial patch indices in [0, N-1].
        grid_size: Tuple of (H_patches, W_patches).
        
    Returns:
        List of length B, where each element is a list of K bounding boxes:
        [(ymin, xmin, ymax, xmax), ...] normalized to [0, 1].
    """
    h_p, w_p = grid_size
    batch_size = patch_indices.shape[0]
    indices_np = patch_indices.detach().cpu().numpy()
    
    batch_bboxes = []
    for b in range(batch_size):
        item_bboxes = []
        for idx in indices_np[b]:
            r = int(idx // w_p)
            c = int(idx % w_p)
            ymin = r / float(h_p)
            ymax = (r + 1) / float(h_p)
            xmin = c / float(w_p)
            xmax = (c + 1) / float(w_p)
            item_bboxes.append((ymin, xmin, ymax, xmax))
        batch_bboxes.append(item_bboxes)
    return batch_bboxes


class SaccadeBottleneck(nn.Module):
    """Drop-in hard Top-K attention information bottleneck.
    
    This module filters the sequence of transformer tokens based on the self-attention
    voting scores of a probing layer. Only the top-K attended patches are routed
    downstream, discarding all other visual tokens.
    
    Attributes:
        top_k: Default visual token budget K.
        has_cls_token: Whether the model sequences prepend a [CLS] token.
        grid_size: Spatial grid dimensions (H_patches, W_patches).
        layer_idx: Identifying index of the layer this bottleneck monitors.
        differentiable_eval: If True, uses straight-through estimator mask during training.
    """

    def __init__(
        self,
        top_k: int = 16,
        grid_size: Tuple[int, int] = (14, 14),
        has_cls_token: bool = True,
        layer_idx: int = 6,
        cls_weight: float = 0.0,
        sort_spatially: bool = False,
    ):
        super().__init__()
        self.top_k = top_k
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1]
        self.has_cls_token = has_cls_token
        self.layer_idx = layer_idx
        self.cls_weight = cls_weight
        self.sort_spatially = sort_spatially

    def forward(
        self,
        tokens: torch.Tensor,
        attention_matrix: torch.Tensor,
        override_top_k: Optional[int] = None,
        return_meta: bool = True,
    ) -> Tuple[torch.Tensor, Optional[SaccadeMetadata]]:
        """Filters tokens to top-K patches based on attention importance.
        
        Args:
            tokens: Sequence tensor of shape [B, S, D].
                    If has_cls_token=True, S = 1 + N. Otherwise S = N.
            attention_matrix: Attention weights of shape [B, H, S, S] or [B, S, S].
            override_top_k: Optional integer to override self.top_k dynamically.
            return_meta: Whether to compute and return SaccadeMetadata.
            
        Returns:
            pruned_tokens: Filtered tokens of shape [B, K + (1 if cls else 0), D].
            metadata: SaccadeMetadata if return_meta=True else None.
        """
        B, S, D = tokens.shape
        K = override_top_k if override_top_k is not None else self.top_k
        N = self.num_patches
        
        # Clamp K to valid range [1, N]
        K = max(1, min(K, N))

        # 1. Compute patch importance vector p in R^{B x N}
        importance = compute_patch_importance(
            attention_matrix=attention_matrix,
            has_cls_token=self.has_cls_token,
            cls_weight=self.cls_weight,
        )  # [B, N]

        # 2. Select top-K patch indices
        # topk returns values [B, K] and indices [B, K] in [0, N-1]
        topk_scores, topk_indices = torch.topk(importance, k=K, dim=-1, largest=True, sorted=True)

        if self.sort_spatially:
            # Sort indices along sequence position to preserve relative spatial raster order
            topk_indices, _ = torch.sort(topk_indices, dim=-1)

        # 3. Extract tokens
        if self.has_cls_token:
            cls_token = tokens[:, :1, :]  # [B, 1, D]
            spatial_tokens = tokens[:, 1:, :]  # [B, N, D]
            
            # Gather top-K spatial tokens: topk_indices expanded to [B, K, D]
            gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            selected_spatial_tokens = torch.gather(spatial_tokens, dim=1, index=gather_indices)  # [B, K, D]
            
            # Downstream tokens: [CLS] concatenated with selected top-K spatial tokens
            pruned_tokens = torch.cat([cls_token, selected_spatial_tokens], dim=1)  # [B, 1 + K, D]
        else:
            gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            pruned_tokens = torch.gather(tokens, dim=1, index=gather_indices)  # [B, K, D]

        meta = None
        if return_meta:
            bboxes = patch_indices_to_bboxes(topk_indices, self.grid_size)
            meta = SaccadeMetadata(
                patch_indices=topk_indices,
                patch_importance=importance,
                top_k=K,
                grid_size=self.grid_size,
                bounding_boxes=bboxes,
                layer_idx=self.layer_idx,
                attention_matrix=attention_matrix if attention_matrix.ndim == 3 else attention_matrix.mean(dim=1),
            )

        return pruned_tokens, meta

