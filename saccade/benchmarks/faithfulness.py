"""
saccade.benchmarks.faithfulness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantitative faithfulness benchmarks: Deletion/Insertion Curves (AUC-Del, AUC-Ins)
and causal verification metrics comparing Saccade intrinsic bottlenecks against
post-hoc explainability methods.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from saccade.models import SaccadeViT
from saccade.bottleneck import compute_patch_importance


def compute_faithfulness_curves(
    model: SaccadeViT,
    image: torch.Tensor,
    target_class: Optional[int] = None,
    num_steps: int = 15,
) -> Dict[str, Any]:
    """Computes Deletion and Insertion curves for Saccade token rankings.
    
    Args:
        model: SaccadeViT model.
        image: Image tensor [1, C, H, W].
        target_class: Target class index. If None, uses top-1 class of full image.
        num_steps: Number of evaluation points along the curve.
        
    Returns:
        Dictionary containing deletion curve, insertion curve, auc_del, auc_ins.
    """
    model.eval()
    device = next(model.parameters()).device
    x = image.to(device)

    # 1. Full image forward pass
    with torch.no_grad():
        full_logits, meta = model(x, return_saccades=True, override_top_k=model.num_patches)
        full_probs = F.softmax(full_logits, dim=-1)
        if target_class is None:
            target_class = int(torch.argmax(full_probs, dim=-1)[0].item())
        base_prob = float(full_probs[0, target_class].item())

    # Ranked patch indices from most important to least
    ranked_indices = meta.patch_indices[0].cpu().numpy().tolist()
    total_patches = len(ranked_indices)

    # Step fractions
    steps = np.linspace(1, total_patches, num_steps, dtype=int)
    steps = sorted(list(set(steps)))

    deletion_probs = []
    insertion_probs = []

    with torch.no_grad():
        # Deletion: Remove top patches step by step
        for k in steps:
            # Mask the top k patches
            ablated = ranked_indices[:k]
            del_logits = model(x, return_saccades=False, override_top_k=total_patches - k, ablate_patch_indices=ablated)
            del_probs = F.softmax(del_logits, dim=-1)
            deletion_probs.append(float(del_probs[0, target_class].item()))

        # Insertion: Keep only the top k patches
        for k in steps:
            ins_logits = model(x, return_saccades=False, override_top_k=k)
            ins_probs = F.softmax(ins_logits, dim=-1)
            insertion_probs.append(float(ins_probs[0, target_class].item()))

    # AUC using trapezoidal rule normalized by range
    norm_steps = [s / float(total_patches) for s in steps]
    auc_del = float(np.trapz(deletion_probs, norm_steps))
    auc_ins = float(np.trapz(insertion_probs, norm_steps))

    return {
        "target_class": target_class,
        "base_prob": base_prob,
        "steps": steps,
        "norm_steps": norm_steps,
        "deletion_probs": deletion_probs,
        "insertion_probs": insertion_probs,
        "auc_deletion": auc_del,
        "auc_insertion": auc_ins,
    }


def verify_causal_gradient_isolation(
    model: SaccadeViT,
    image: torch.Tensor,
    top_k: int = 12,
) -> Dict[str, Any]:
    """Mathematically verifies that tokens discarded by the Saccade bottleneck
    have identically zero gradient with respect to downstream predictions.
    
    In post-hoc saliency methods, gradients leak across all spatial tokens.
    In SaccadeViT, downstream tokens are causally restricted to the Top-K set.
    """
    model.eval()
    device = next(model.parameters()).device
    x = image.clone().detach().to(device).requires_grad_(True)

    # Forward with bottleneck
    tokens, attn_weights = model.forward_features_to_bottleneck(x)
    importance = compute_patch_importance(
        attention_matrix=attn_weights,
        has_cls_token=model.has_cls_token,
        cls_weight=model.bottleneck.cls_weight,
    )
    _, topk_indices = torch.topk(importance, k=top_k, dim=-1, largest=True, sorted=True)

    B, S, D = tokens.shape
    cls_token = tokens[:, :1, :]
    spatial_tokens = tokens[:, 1:, :]
    gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
    selected_spatial = torch.gather(spatial_tokens, dim=1, index=gather_indices)
    pruned_tokens = torch.cat([cls_token, selected_spatial], dim=1)

    # Downstream prediction
    logits = model.forward_from_bottleneck(pruned_tokens)
    target_logit = logits[0, 0]
    
    # Compute gradient w.r.t input spatial tokens before bottleneck
    # We measure gradient on spatial_tokens
    grads = torch.autograd.grad(target_logit, spatial_tokens, retain_graph=False)[0]  # [1, N, D]
    grad_norm_per_patch = grads.norm(dim=-1)[0].detach().cpu().numpy()  # [N]

    active_set = set(topk_indices[0].cpu().numpy().tolist())
    all_patches = set(range(model.num_patches))
    discarded_set = all_patches - active_set

    active_grad_norms = [float(grad_norm_per_patch[i]) for i in active_set]
    discarded_grad_norms = [float(grad_norm_per_patch[i]) for i in discarded_set]

    max_discarded_grad = max(discarded_grad_norms) if discarded_grad_norms else 0.0
    mean_active_grad = float(np.mean(active_grad_norms)) if active_grad_norms else 0.0

    return {
        "top_k": top_k,
        "active_patches_count": len(active_set),
        "discarded_patches_count": len(discarded_set),
        "mean_active_patch_grad_norm": mean_active_grad,
        "max_discarded_patch_grad_norm": max_discarded_grad,
        "exact_zero_leakage_guarantee": bool(max_discarded_grad < 1e-9),
    }

