"""
saccade.exporter
~~~~~~~~~~~~~~~~

Generates zero-dependency, self-contained interactive HTML/Canvas artifacts
in the visual research style of David Ha (@hardmaru) and otoro.net.
"""

from __future__ import annotations
import base64
import io
import json
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from saccade.models import SaccadeViT
from saccade.bottleneck import compute_patch_importance


def image_to_base64(image: Union[Image.Image, torch.Tensor, np.ndarray]) -> str:
    """Converts an image to a base64-encoded PNG data URI."""
    if isinstance(image, torch.Tensor):
        img_t = image.detach().cpu()
        if img_t.ndim == 4:
            img_t = img_t[0]
        if img_t.shape[0] == 3 or img_t.shape[0] == 1:
            img_t = img_t.permute(1, 2, 0)
        img_np = (img_t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)
        pil_img = Image.fromarray(img_np)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image)
    else:
        pil_img = image.convert("RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def export_interactive_html(
    model: SaccadeViT,
    image: Union[Image.Image, torch.Tensor],
    output_path: Union[str, Path] = "saccade_viewer.html",
    class_names: Optional[List[str]] = None,
    default_top_k: int = 12,
    title: str = "saccade // Intrinsically Interpretable Vision Transformer",
    description: Optional[str] = None,
) -> str:
    """Exports a David Ha / otoro.net style standalone interactive web artifact.
    
    The generated HTML file embeds the image, precomputed patch attention votes,
    sparsity trajectory (confidence vs K), and counterfactual ablation deltas.
    Users can drag a slider to vary visual budget K from 1 to N, click patches
    to ablate them, and inspect causal shifts in real time.
    
    Args:
        model: SaccadeViT instance.
        image: Input image (PIL.Image or [1, C, H, W] Torch tensor).
        output_path: Path to save the self-contained HTML file.
        class_names: Optional mapping from class index to human-readable string.
        default_top_k: Initial visual budget slider position.
        title: Page and header title.
        description: Optional subtext or case-study context.
        
    Returns:
        Absolute file path to the generated HTML file.
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. Prepare image tensor
    expected_size = getattr(model.backbone, "img_size", None)
    if expected_size is None:
        if hasattr(model.backbone, "patch_embed") and hasattr(model.backbone.patch_embed, "img_size"):
            expected_size = model.backbone.patch_embed.img_size
    if expected_size is None:
        expected_size = (224, 224)
    elif isinstance(expected_size, int):
        expected_size = (expected_size, expected_size)

    if isinstance(image, Image.Image):
        # Resize to model expected input size
        img_resized = image.resize(expected_size, Image.Resampling.BILINEAR)
        img_np = np.array(img_resized.convert("RGB"), dtype=np.float32) / 255.0
        # Standard ViT normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_np - mean) / std
        x_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
        base64_data = image_to_base64(img_resized)
    elif isinstance(image, torch.Tensor):
        x_tensor = image.to(device)
        if x_tensor.ndim == 3:
            x_tensor = x_tensor.unsqueeze(0)
        base64_data = image_to_base64(x_tensor)
    else:
        raise TypeError("Expected PIL Image or Torch Tensor.")

    # 2. Extract features and attention at bottleneck layer
    with torch.no_grad():
        tokens, attn_weights = model.forward_features_to_bottleneck(x_tensor)
        importance = compute_patch_importance(
            attention_matrix=attn_weights,
            has_cls_token=model.has_cls_token,
            cls_weight=model.bottleneck.cls_weight,
        )[0]  # [N]

    num_patches = model.num_patches
    grid_h, grid_w = model.grid_size
    sorted_importance, sorted_indices = torch.sort(importance, descending=True)
    all_ranked_indices = sorted_indices.cpu().numpy().tolist()
    importance_scores = importance.cpu().numpy().tolist()

    # 3. Precompute trajectory for K values from 1 to num_patches
    # To keep payload compact and fast, compute K for dense early steps and subsampled later
    dense_k = list(range(1, min(33, num_patches + 1)))
    coarse_k = list(range(36, num_patches + 1, 8))
    if num_patches not in coarse_k and num_patches not in dense_k:
        coarse_k.append(num_patches)
    all_k = sorted(list(set(dense_k + coarse_k)))

    trajectory = {}
    with torch.no_grad():
        B, S, D = tokens.shape
        cls_tok = tokens[:, :1, :] if model.has_cls_token else None
        spatial_toks = tokens[:, 1:, :] if model.has_cls_token else tokens

        for k_val in all_k:
            k_indices = torch.tensor(all_ranked_indices[:k_val], device=device).unsqueeze(0)
            gather_indices = k_indices.unsqueeze(-1).expand(-1, -1, D)
            sel_toks = torch.gather(spatial_toks, dim=1, index=gather_indices)
            down_toks = torch.cat([cls_tok, sel_toks], dim=1) if model.has_cls_token else sel_toks

            logits = model.forward_from_bottleneck(down_toks)[0]
            probs = F.softmax(logits, dim=-1)
            
            top_vals, top_inds = torch.topk(probs, k=min(5, logits.shape[-1]))
            top_preds = [
                {
                    "class_idx": int(top_inds[i].item()),
                    "class_name": class_names[int(top_inds[i].item())] if class_names and int(top_inds[i].item()) < len(class_names) else f"Class {int(top_inds[i].item())}",
                    "prob": float(top_vals[i].item()),
                    "logit": float(logits[top_inds[i]].item()),
                }
                for i in range(len(top_inds))
            ]

            trajectory[k_val] = {
                "top_preds": top_preds,
                "top1_class": top_preds[0]["class_name"],
                "top1_prob": top_preds[0]["prob"],
            }

    # 4. Precompute counterfactual single-patch ablations for top-24 patches at default_top_k
    ablations = {}
    base_preds = trajectory.get(default_top_k, trajectory[all_k[0]])
    base_top1_idx = base_preds["top_preds"][0]["class_idx"]
    base_top1_logit = base_preds["top_preds"][0]["logit"]

    eval_top_k = min(default_top_k, num_patches)
    top_eval_indices = all_ranked_indices[:eval_top_k]

    with torch.no_grad():
        for p_idx in top_eval_indices:
            # Mask out this patch
            ablated_k_indices = [idx for idx in all_ranked_indices if idx != p_idx][:eval_top_k]
            k_indices = torch.tensor(ablated_k_indices, device=device).unsqueeze(0)
            gather_indices = k_indices.unsqueeze(-1).expand(-1, -1, D)
            sel_toks = torch.gather(spatial_toks, dim=1, index=gather_indices)
            down_toks = torch.cat([cls_tok, sel_toks], dim=1) if model.has_cls_token else sel_toks

            logits = model.forward_from_bottleneck(down_toks)[0]
            probs = F.softmax(logits, dim=-1)
            ablated_top1_logit = float(logits[base_top1_idx].item())
            ablated_top1_prob = float(probs[base_top1_idx].item())
            delta_logit = base_top1_logit - ablated_top1_logit

            ablations[p_idx] = {
                "delta_logit": delta_logit,
                "ablated_prob": ablated_top1_prob,
                "base_prob": base_preds["top_preds"][0]["prob"],
            }

    # Assemble embedded JSON payload
    payload = {
        "title": title,
        "grid_size": [grid_h, grid_w],
        "num_patches": num_patches,
        "default_top_k": default_top_k,
        "all_ranked_indices": all_ranked_indices,
        "importance_scores": importance_scores,
        "trajectory": trajectory,
        "ablations": ablations,
        "available_k": all_k,
        "bottleneck_layer": model.bottleneck_layer,
    }

    html_content = _build_html_template(
        base64_image=base64_data,
        payload=payload,
        title=title,
        description=description or "Zero-dependency causal interpretability artifact. Slide to expand the visual budget or click patches to ablate them counterfactually."
    )

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html_content, encoding="utf-8")
    return str(out_file)


def _build_html_template(base64_image: str, payload: Dict[str, Any], title: str, description: str) -> str:
    """Builds the single-file, zero-dependency HTML/CSS/Canvas artifact."""
    payload_json = json.dumps(payload)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #090B10;
      --card-bg: #12161F;
      --border: #222938;
      --accent: #00FFCC;
      --accent-dim: rgba(0, 255, 204, 0.15);
      --pink: #FF007F;
      --danger: #FF3366;
      --text: #F0F4F8;
      --text-muted: #8E9BAE;
      --font-mono: 'SF Mono', Monaco, Consolas, 'Fira Code', monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem;
      -webkit-font-smoothing: antialiased;
    }}
    header {{
      text-align: center;
      max-width: 800px;
      margin-bottom: 2rem;
    }}
    .badge {{
      display: inline-block;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      color: var(--accent);
      background: var(--accent-dim);
      border: 1px solid var(--accent);
      border-radius: 9999px;
      padding: 0.2rem 0.75rem;
      margin-bottom: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    h1 {{
      font-size: 1.85rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 0.5rem;
      color: #FFFFFF;
    }}
    p.desc {{
      color: var(--text-muted);
      font-size: 0.95rem;
      line-height: 1.5;
    }}
    .container {{
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 1.5rem;
      max-width: 1060px;
      width: 100%;
    }}
    @media (max-width: 860px) {{
      .container {{
        grid-template-columns: 1fr;
      }}
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }}
    .canvas-container {{
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .canvas-wrapper {{
      position: relative;
      width: 448px;
      height: 448px;
      max-width: 100%;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
      cursor: crosshair;
      user-select: none;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .canvas-hints {{
      display: flex;
      justify-content: space-between;
      width: 100%;
      max-width: 448px;
      margin-top: 0.6rem;
      font-size: 0.75rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }}
    .controls {{
      margin-top: 1.25rem;
      width: 100%;
      max-width: 448px;
    }}
    .slider-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.4rem;
      font-size: 0.85rem;
      font-family: var(--font-mono);
    }}
    .budget-val {{
      color: var(--accent);
      font-weight: 700;
    }}
    input[type=range] {{
      width: 100%;
      height: 6px;
      background: var(--border);
      border-radius: 3px;
      outline: none;
      -webkit-appearance: none;
    }}
    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--accent);
      cursor: pointer;
      box-shadow: 0 0 10px var(--accent);
    }}
    .toggles {{
      display: flex;
      gap: 1rem;
      margin-top: 0.75rem;
      font-size: 0.8rem;
    }}
    .toggle-label {{
      display: flex;
      align-items: center;
      gap: 0.35rem;
      color: var(--text-muted);
      cursor: pointer;
    }}
    .toggle-label input {{
      accent-color: var(--accent);
    }}
    .sidebar {{
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }}
    .section-title {{
      font-family: var(--font-mono);
      font-size: 0.8rem;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.75rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .pred-row {{
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
      margin-bottom: 0.75rem;
    }}
    .pred-header {{
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
    }}
    .pred-name {{
      font-weight: 600;
      color: #FFF;
    }}
    .pred-pct {{
      font-family: var(--font-mono);
      color: var(--accent);
    }}
    .pred-bar-bg {{
      width: 100%;
      height: 8px;
      background: var(--border);
      border-radius: 4px;
      overflow: hidden;
    }}
    .pred-bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, #00BFA5, var(--accent));
      border-radius: 4px;
      transition: width 0.25s ease;
    }}
    .causal-box {{
      background: rgba(255, 51, 102, 0.08);
      border: 1px dashed var(--danger);
      border-radius: 8px;
      padding: 0.85rem;
      font-size: 0.8rem;
    }}
    .causal-box.active {{
      background: rgba(255, 51, 102, 0.15);
    }}
    .causal-title {{
      color: var(--danger);
      font-weight: 700;
      font-family: var(--font-mono);
      margin-bottom: 0.35rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .causal-stats {{
      font-family: var(--font-mono);
      color: var(--text);
      line-height: 1.4;
    }}
    .math-note {{
      background: rgba(255, 255, 255, 0.03);
      border-radius: 8px;
      padding: 0.85rem;
      font-size: 0.75rem;
      color: var(--text-muted);
      line-height: 1.45;
      font-family: var(--font-mono);
    }}
    .reset-btn {{
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.7rem;
      cursor: pointer;
      font-family: var(--font-mono);
    }}
    .reset-btn:hover {{
      color: var(--accent);
      border-color: var(--accent);
    }}
    footer {{
      margin-top: 2.5rem;
      text-align: center;
      font-size: 0.8rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }}
    footer a {{
      color: var(--accent);
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <header>
    <div class="badge">David Ha Research Philosophy // Saccade Vision</div>
    <h1>{title}</h1>
    <p class="desc">{description}</p>
  </header>

  <div class="container">
    <!-- Visual Canvas Card -->
    <div class="card canvas-container">
      <div class="canvas-wrapper">
        <canvas id="saccadeCanvas" width="448" height="448"></canvas>
      </div>

      <div class="canvas-hints">
        <span id="hoverInfo">Hover over a patch to inspect</span>
        <span>Click active patch to ablate</span>
      </div>

      <div class="controls">
        <div class="slider-row">
          <span>Visual Patch Budget (K):</span>
          <span><span class="budget-val" id="kDisplay">12</span> / <span id="nDisplay">196</span> (<span id="ratioDisplay">6.1%</span> context)</span>
        </div>
        <input type="range" id="kSlider" min="1" max="196" value="12">

        <div class="toggles">
          <label class="toggle-label">
            <input type="checkbox" id="toggleScanpath" checked> Scanpath Arcs
          </label>
          <label class="toggle-label">
            <input type="checkbox" id="toggleRanks" checked> Rank Labels
          </label>
          <label class="toggle-label">
            <input type="checkbox" id="toggleDim" checked> Dim Background
          </label>
        </div>
      </div>
    </div>

    <!-- Causal Inspection Sidebar -->
    <div class="sidebar">
      <div class="card">
        <div class="section-title">
          <span>Model Prediction</span>
          <span style="font-size: 0.7rem; color: var(--text-muted);">Layer {payload.get("bottleneck_layer", 6)} Bottleneck</span>
        </div>
        <div id="predictionsList"></div>
      </div>

      <div class="card">
        <div class="section-title">
          <span>Counterfactual Ablation</span>
          <button class="reset-btn" id="resetAblations">Reset All</button>
        </div>
        <div class="causal-box" id="causalBox">
          <div class="causal-title">
            <span>●</span> Causal Perturbation
          </div>
          <div class="causal-stats" id="causalStats">
            Click any active patch in the image to knock it out and observe the causal logit shift.
          </div>
        </div>
      </div>

      <div class="card math-note">
        <strong style="color: #FFF;">Intrinsic Faithfulness:</strong><br>
        Patch importance is computed via attention column sum voting:<br>
        <code style="color: var(--accent);">p_j = Σ_i A_{{i,j}}</code><br><br>
        All downstream layers receive strictly the Top-K tokens. Non-selected patches are physically excluded from the compute graph:
        <code style="color: var(--pink);">∂ŷ/∂x_m = 0 ∀ m ∉ K</code>
      </div>
    </div>
  </div>

  <footer>
    Built with <a href="https://github.com/helloitsdaksh/saccade" target="_blank">saccade-ai</a> // Intrinsic Attention Bottlenecks
  </footer>

  <script>
    const DATA = {payload_json};
    const IMG_SRC = "{base64_image}";

    // State
    let currentK = DATA.default_top_k;
    let ablatedPatches = new Set();
    let hoveredPatch = null;
    let showScanpath = true;
    let showRanks = true;
    let dimBackground = true;

    // Elements
    const canvas = document.getElementById('saccadeCanvas');
    const ctx = canvas.getContext('2d');
    const kSlider = document.getElementById('kSlider');
    const kDisplay = document.getElementById('kDisplay');
    const nDisplay = document.getElementById('nDisplay');
    const ratioDisplay = document.getElementById('ratioDisplay');
    const hoverInfo = document.getElementById('hoverInfo');
    const predictionsList = document.getElementById('predictionsList');
    const causalStats = document.getElementById('causalStats');
    const causalBox = document.getElementById('causalBox');
    const resetAblationsBtn = document.getElementById('resetAblations');
    const toggleScanpath = document.getElementById('toggleScanpath');
    const toggleRanks = document.getElementById('toggleRanks');
    const toggleDim = document.getElementById('toggleDim');

    // Setup slider limits
    const numPatches = DATA.num_patches;
    kSlider.max = numPatches;
    kSlider.value = currentK;
    nDisplay.textContent = numPatches;

    // Image loading
    const baseImg = new Image();
    baseImg.src = IMG_SRC;
    baseImg.onload = () => {{
      render();
      updateUI();
    }};

    // Get closest trajectory entry
    function getTrajectoryForK(k) {{
      const available = DATA.available_k;
      let closest = available[0];
      let minDiff = Math.abs(k - closest);
      for (const cand of available) {{
        const diff = Math.abs(k - cand);
        if (diff < minDiff) {{
          minDiff = diff;
          closest = cand;
        }}
      }}
      return DATA.trajectory[closest];
    }}

    function getActivePatches() {{
      const list = DATA.all_ranked_indices.slice(0, currentK);
      return list.filter(idx => !ablatedPatches.has(idx));
    }}

    function render() {{
      const w = canvas.width;
      const h = canvas.height;
      const [gh, gw] = DATA.grid_size;
      const patchW = w / gw;
      const patchH = h / gh;

      ctx.clearRect(0, 0, w, h);

      // 1. Draw base image
      if (dimBackground) {{
        ctx.save();
        ctx.drawImage(baseImg, 0, 0, w, h);
        ctx.fillStyle = 'rgba(9, 11, 16, 0.78)';
        ctx.fillRect(0, 0, w, h);
        ctx.restore();
      }} else {{
        ctx.drawImage(baseImg, 0, 0, w, h);
      }}

      const activeList = getActivePatches();
      const activeSet = new Set(activeList);
      const centers = [];

      // 2. Render active patches crisp and highlighted
      for (let rank = 0; rank < activeList.length; rank++) {{
        const idx = activeList[rank];
        const r = Math.floor(idx / gw);
        const c = idx % gw;
        const x = c * patchW;
        const y = r * patchH;

        // Draw crisp patch
        ctx.drawImage(baseImg, x, y, patchW, patchH, x, y, patchW, patchH);

        // Border
        ctx.strokeStyle = '#00FFCC';
        ctx.lineWidth = (hoveredPatch === idx) ? 3 : 1.5;
        ctx.strokeRect(x, y, patchW, patchH);

        // Rank label
        if (showRanks) {{
          ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
          ctx.fillRect(x, y, 22, 14);
          ctx.fillStyle = '#00FFCC';
          ctx.font = '10px var(--font-mono)';
          ctx.fillText('#' + (rank + 1), x + 3, y + 11);
        }}

        centers.push({{ x: x + patchW / 2, y: y + patchH / 2 }});
      }}

      // 3. Render ablated patches with red cross-hatch
      ablatedPatches.forEach(idx => {{
        const r = Math.floor(idx / gw);
        const c = idx % gw;
        const x = c * patchW;
        const y = r * patchH;

        ctx.fillStyle = 'rgba(255, 51, 102, 0.35)';
        ctx.fillRect(x, y, patchW, patchH);
        ctx.strokeStyle = '#FF3366';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x, y, patchW, patchH);

        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + patchW, y + patchH);
        ctx.moveTo(x + patchW, y);
        ctx.lineTo(x, y + patchH);
        ctx.stroke();
      }});

      // 4. Render scanpath arcs
      if (showScanpath && centers.length > 1) {{
        ctx.save();
        ctx.strokeStyle = 'rgba(255, 0, 127, 0.75)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(centers[0].x, centers[0].y);
        for (let i = 1; i < centers.length; i++) {{
          ctx.lineTo(centers[i].x, centers[i].y);
        }}
        ctx.stroke();

        // Draw node circles
        centers.forEach((pt, i) => {{
          ctx.fillStyle = (i === 0) ? '#00FFCC' : '#FF007F';
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, (i === 0) ? 4.5 : 3, 0, Math.PI * 2);
          ctx.fill();
        }});
        ctx.restore();
      }}
    }}

    function updateUI() {{
      kDisplay.textContent = currentK;
      const pct = ((currentK / numPatches) * 100).toFixed(1);
      ratioDisplay.textContent = pct + '%';

      // Update predictions
      const traj = getTrajectoryForK(currentK);
      if (traj && traj.top_preds) {{
        predictionsList.innerHTML = traj.top_preds.map((pred, i) => {{
          const probPct = (pred.prob * 100).toFixed(1);
          return `
            <div class="pred-row">
              <div class="pred-header">
                <span class="pred-name">${{pred.class_name}}</span>
                <span class="pred-pct">${{probPct}}%</span>
              </div>
              <div class="pred-bar-bg">
                <div class="pred-bar-fill" style="width: ${{probPct}}%;"></div>
              </div>
            </div>
          `;
        }}).join('');
      }}

      // Update ablation info
      if (ablatedPatches.size > 0) {{
        causalBox.classList.add('active');
        let totalDelta = 0;
        let details = [];
        ablatedPatches.forEach(idx => {{
          const ab = DATA.ablations[idx];
          if (ab) {{
            totalDelta += ab.delta_logit;
            details.push(`Patch #${{DATA.all_ranked_indices.indexOf(idx) + 1}} (Δ Logit: -${{ab.delta_logit.toFixed(2)}})`);
          }}
        }});
        causalStats.innerHTML = `
          <strong>${{ablatedPatches.size}} patch(es) causally knocked out:</strong><br>
          ${{details.join('<br>')}}<br><br>
          <span style="color: var(--danger);">Cumulative Target Logit Penalty: -${{totalDelta.toFixed(2)}}</span>
        `;
      }} else {{
        causalBox.classList.remove('active');
        causalStats.textContent = 'Click any active patch in the image to knock it out and observe the causal logit shift.';
      }}
    }}

    // Events
    kSlider.addEventListener('input', (e) => {{
      currentK = parseInt(e.target.value, 10);
      render();
      updateUI();
    }});

    toggleScanpath.addEventListener('change', (e) => {{
      showScanpath = e.target.checked;
      render();
    }});

    toggleRanks.addEventListener('change', (e) => {{
      showRanks = e.target.checked;
      render();
    }});

    toggleDim.addEventListener('change', (e) => {{
      dimBackground = e.target.checked;
      render();
    }});

    resetAblationsBtn.addEventListener('click', () => {{
      ablatedPatches.clear();
      render();
      updateUI();
    }});

    // Canvas click & hover
    function getPatchFromEvent(e) {{
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const clientX = (e.clientX - rect.left) * scaleX;
      const clientY = (e.clientY - rect.top) * scaleY;
      const [gh, gw] = DATA.grid_size;
      const patchW = canvas.width / gw;
      const patchH = canvas.height / gh;
      const col = Math.floor(clientX / patchW);
      const row = Math.floor(clientY / patchH);
      if (col >= 0 && col < gw && row >= 0 && row < gh) {{
        return row * gw + col;
      }}
      return null;
    }}

    canvas.addEventListener('mousemove', (e) => {{
      const patchIdx = getPatchFromEvent(e);
      hoveredPatch = patchIdx;
      if (patchIdx !== null) {{
        const rank = DATA.all_ranked_indices.indexOf(patchIdx) + 1;
        const score = DATA.importance_scores[patchIdx].toFixed(3);
        const isActive = getActivePatches().includes(patchIdx);
        const isAblated = ablatedPatches.has(patchIdx);
        const status = isAblated ? ' [ABLATED]' : (isActive ? ' [ACTIVE]' : ' [PRUNED]');
        hoverInfo.textContent = `Patch #${{rank}} (idx: ${{patchIdx}}, vote: ${{score}})${{status}}`;
      }} else {{
        hoverInfo.textContent = 'Hover over a patch to inspect';
      }}
      render();
    }});

    canvas.addEventListener('mouseleave', () => {{
      hoveredPatch = null;
      hoverInfo.textContent = 'Hover over a patch to inspect';
      render();
    }});

    canvas.addEventListener('click', (e) => {{
      const patchIdx = getPatchFromEvent(e);
      if (patchIdx === null) return;

      const topPatches = new Set(DATA.all_ranked_indices.slice(0, currentK));
      if (topPatches.has(patchIdx)) {{
        if (ablatedPatches.has(patchIdx)) {{
          ablatedPatches.delete(patchIdx);
        }} else {{
          ablatedPatches.add(patchIdx);
        }}
        render();
        updateUI();
      }}
    }});
  </script>
</body>
</html>
"""
