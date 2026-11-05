"""
saccade.visualization
~~~~~~~~~~~~~~~~~~~~~

Visualization utilities for spatial attention saccades, patch overlays,
scanpaths, and comparative interpretability heatmaps.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from saccade.bottleneck import SaccadeMetadata


def render_saccade_overlay(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    meta: SaccadeMetadata,
    dim_factor: float = 0.25,
    border_color: str = "#00FFCC",
    draw_scanpath: bool = True,
    scanpath_color: str = "#FF007F",
    show_ranks: bool = True,
) -> Image.Image:
    """Renders the image with non-selected patches dimmed out and active patches highlighted.
    
    Args:
        image: PIL Image, numpy array [H, W, 3], or Torch tensor [3, H, W] in [0, 1].
        meta: SaccadeMetadata from SaccadeViT forward pass.
        dim_factor: Brightness multiplier for unselected background patches (0.0 to 1.0).
        border_color: Hex or RGBA string for patch borders.
        draw_scanpath: Whether to draw connecting lines following attention rank order.
        scanpath_color: Color of the scanpath line.
        show_ranks: Whether to label patches with their importance ranking (1 to K).
        
    Returns:
        PIL.Image containing the rendered visualization.
    """
    # 1. Convert input to PIL Image
    if isinstance(image, torch.Tensor):
        img_t = image.detach().cpu()
        if img_t.ndim == 4:
            img_t = img_t[0]
        if img_t.shape[0] == 3 or img_t.shape[0] == 1:
            img_t = img_t.permute(1, 2, 0)
        img_np = (img_t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)
        base_img = Image.fromarray(img_np)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        base_img = Image.fromarray(image)
    else:
        base_img = image.convert("RGB")

    w, h = base_img.size
    h_p, w_p = meta.grid_size
    patch_w = w / float(w_p)
    patch_h = h / float(h_p)

    # 2. Create dimmed background
    dimmed_img = Image.eval(base_img, lambda x: int(x * dim_factor))

    # 3. Create composite where active patches keep full brightness
    composite = dimmed_img.copy()
    
    # Active patch indices for batch item 0
    active_indices = set(meta.patch_indices[0].cpu().numpy().tolist())
    ranking = {idx: rank + 1 for rank, idx in enumerate(meta.patch_indices[0].cpu().numpy().tolist())}

    centers = []
    for idx in meta.patch_indices[0].cpu().numpy().tolist():
        r = int(idx // w_p)
        c = int(idx % w_p)
        x0 = int(c * patch_w)
        y0 = int(r * patch_h)
        x1 = int(round((c + 1) * patch_w))
        y1 = int(round((r + 1) * patch_h))

        # Paste original crisp patch region
        patch_crop = base_img.crop((x0, y0, x1, y1))
        composite.paste(patch_crop, (x0, y0))
        centers.append(( (x0 + x1) / 2.0, (y0 + y1) / 2.0 ))

    # 4. Draw annotations (borders, scanpaths, rank text)
    draw = ImageDraw.Draw(composite, "RGBA")

    # Draw scanpaths
    if draw_scanpath and len(centers) > 1:
        for i in range(len(centers) - 1):
            p1 = centers[i]
            p2 = centers[i + 1]
            draw.line([p1, p2], fill=scanpath_color, width=2)

    # Draw patch borders & rank tags
    for rank_idx, idx in enumerate(meta.patch_indices[0].cpu().numpy().tolist()):
        r = int(idx // w_p)
        c = int(idx % w_p)
        x0 = int(c * patch_w)
        y0 = int(r * patch_h)
        x1 = int(round((c + 1) * patch_w))
        y1 = int(round((r + 1) * patch_h))

        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=border_color, width=2)
        
        if show_ranks:
            rank_label = f"#{rank_idx + 1}"
            draw.rectangle([x0, y0, x0 + 22, y0 + 14], fill=(0, 0, 0, 200))
            draw.text((x0 + 2, y0 + 1), rank_label, fill=(255, 255, 255, 255))

    return composite


def plot_saccade_breakdown(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    meta: SaccadeMetadata,
    class_names: Optional[List[str]] = None,
    logits: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Creates a 3-panel figure: Original image, Saccade Patch Selection, and Patch Importance Heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0F1117")

    # Panel 1: Original
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim == 4:
            t = t[0]
        if t.shape[0] == 3 or t.shape[0] == 1:
            t = t.permute(1, 2, 0)
        img_np = (t.numpy()).clip(0, 1)
    elif isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB")) / 255.0
    else:
        img_np = image if image.max() <= 1.0 else image / 255.0

    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", color="white", fontsize=12, pad=10)
    axes[0].axis("off")

    # Panel 2: Saccade Bottleneck Overlay
    overlay_pil = render_saccade_overlay(image, meta, dim_factor=0.25)
    axes[1].imshow(overlay_pil)
    title_str = f"Saccade Bottleneck (K={meta.top_k}/{meta.grid_size[0]*meta.grid_size[1]})"
    if logits is not None and class_names is not None:
        pred_idx = int(torch.argmax(logits, dim=-1)[0].item())
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
        title_str += f"\nPred: {pred_name}"
    axes[1].set_title(title_str, color="#00FFCC", fontsize=12, pad=10)
    axes[1].axis("off")

    # Panel 3: Full Attention Voting Heatmap
    importance = meta.patch_importance[0].cpu().numpy().reshape(meta.grid_size)
    im = axes[2].imshow(importance, cmap="magma", interpolation="nearest")
    axes[2].set_title(r"Patch Importance $p_j = \sum_i A_{i,j}$", color="white", fontsize=12, pad=10)
    axes[2].axis("off")
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
    return fig

