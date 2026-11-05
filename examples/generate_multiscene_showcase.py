"""
examples/generate_multiscene_showcase.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates an intuitive, clutter-free Side-by-Side interactive showcase using
pretrained DeiT-Small on real photos. 

Solves visual overlap by displaying:
1. Clean Original Photo (Left)
2. What the Model Actually Sees (Right, isolated patches on dark background)
3. Extracted Patch Filmstrip (Bottom, magnified individual patch crops with rank & votes)
"""

import sys
import json
import base64
import io
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ViT_B_16_Weights

sys.path.insert(0, str(Path(__file__).parent.parent))

from saccade.models import SaccadeViT
from saccade.bottleneck import compute_patch_importance


def image_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def main():
    print("=" * 65)
    print("Building Intuitive Side-by-Side Showcase (Pretrained ViT)...")
    print("=" * 65)

    labels = ViT_B_16_Weights.DEFAULT.meta["categories"]

    print("Loading pretrained deit_small_patch16_224...")
    model = SaccadeViT.from_pretrained(
        "deit_small_patch16_224",
        pretrained=True,
        num_classes=1000,
        bottleneck_layer=6,
    )
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    samples_dir = Path(__file__).parent.parent / "assets" / "samples"
    scenes_meta = [
        {
            "id": "panda",
            "name": "🐼 Giant Panda",
            "file": "giant_panda.jpg",
            "target_label": "giant panda",
            "hint": "Look at the right canvas: with only 12 patches (6% of pixels), the model discards the bamboo and background, seeing only the eyes, ears, and snout—yet predicts 'giant panda' with 81%+ confidence!",
        },
        {
            "id": "cat",
            "name": "🐱 Tabby Cat",
            "file": "tabby_cat.jpg",
            "target_label": "tabby",
            "hint": "The model isolates the whiskers, eyes, and ear tips. Notice how surrounding sand and rocks receive zero visual tokens.",
        },
        {
            "id": "dog",
            "name": "🐕 Dog / Samoyed",
            "file": "golden_retriever.jpg",
            "target_label": "Samoyed",
            "hint": "Attention concentrates on the smiling mouth and eyes. Disabling the snout patch immediately shifts class distribution.",
        },
    ]

    transform = T.Compose([
        T.Resize((224, 224), Image.Resampling.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    scenes_data = {}

    for scene in scenes_meta:
        print(f"\nProcessing scene: {scene['name']}...")
        img_path = samples_dir / scene["file"]
        pil_raw = Image.open(img_path).convert("RGB")
        pil_224 = pil_raw.resize((224, 224), Image.Resampling.BILINEAR)
        b64 = image_to_base64(pil_224)

        x = transform(pil_raw).unsqueeze(0).to(device)

        with torch.no_grad():
            tokens, attn_weights = model.forward_features_to_bottleneck(x)
            importance = compute_patch_importance(
                attention_matrix=attn_weights,
                has_cls_token=model.has_cls_token,
                cls_weight=0.0
            )[0]

            sorted_scores, sorted_indices = torch.sort(importance, descending=True)
            ranked_indices = sorted_indices.cpu().numpy().tolist()
            importance_scores = importance.cpu().numpy().tolist()

            k_steps = sorted(list(set([1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 196])))
            trajectory = {}

            B, S, D = tokens.shape
            cls_tok = tokens[:, :1, :]
            spatial_toks = tokens[:, 1:, :]

            for k in k_steps:
                k_sel = torch.tensor(ranked_indices[:k], device=device).unsqueeze(0)
                gather_indices = k_sel.unsqueeze(-1).expand(-1, -1, D)
                sel_toks = torch.gather(spatial_toks, dim=1, index=gather_indices)
                down_toks = torch.cat([cls_tok, sel_toks], dim=1)

                logits = model.forward_from_bottleneck(down_toks)[0]
                probs = F.softmax(logits, dim=-1)

                top_vals, top_inds = torch.topk(probs, k=5)
                top_preds = [
                    {
                        "class_idx": int(top_inds[i].item()),
                        "class_name": labels[int(top_inds[i].item())],
                        "prob": float(top_vals[i].item()),
                        "logit": float(logits[top_inds[i]].item()),
                    }
                    for i in range(len(top_inds))
                ]

                trajectory[k] = {
                    "top_preds": top_preds,
                    "top1_class": top_preds[0]["class_name"],
                    "top1_prob": top_preds[0]["prob"],
                }

            # Precompute counterfactual ablations for top-24 patches at default K=12
            default_k = 12
            base_pred = trajectory[default_k]
            base_class_idx = base_pred["top_preds"][0]["class_idx"]
            base_class_logit = base_pred["top_preds"][0]["logit"]

            ablations = {}
            for p_idx in ranked_indices[:default_k]:
                ablated_indices = [idx for idx in ranked_indices if idx != p_idx][:default_k]
                k_sel = torch.tensor(ablated_indices, device=device).unsqueeze(0)
                gather_indices = k_sel.unsqueeze(-1).expand(-1, -1, D)
                sel_toks = torch.gather(spatial_toks, dim=1, index=gather_indices)
                down_toks = torch.cat([cls_tok, sel_toks], dim=1)

                logits = model.forward_from_bottleneck(down_toks)[0]
                probs = F.softmax(logits, dim=-1)
                ablated_logit = float(logits[base_class_idx].item())
                ablated_prob = float(probs[base_class_idx].item())
                delta = base_class_logit - ablated_logit

                ablations[p_idx] = {
                    "delta_logit": delta,
                    "ablated_prob": ablated_prob,
                    "base_prob": base_pred["top_preds"][0]["prob"],
                }

        scenes_data[scene["id"]] = {
            "meta": scene,
            "image_b64": b64,
            "ranked_indices": ranked_indices,
            "importance_scores": importance_scores,
            "trajectory": trajectory,
            "ablations": ablations,
            "available_k": k_steps,
        }
        print(f"  ✓ {scene['name']}: Top-1 at K=12 is '{trajectory[12]['top1_class']}' ({trajectory[12]['top1_prob']*100:.1f}%)")

    html_path = Path(__file__).parent.parent / "assets" / "proper_model_interactive.html"
    print(f"\nWriting interactive showcase to {html_path}...")
    html_content = generate_showcase_html(scenes_data)
    html_path.write_text(html_content, encoding="utf-8")
    print(f"✓ Showcase successfully written! ({html_path.stat().st_size / 1024:.1f} KB)")


def generate_showcase_html(scenes_data: dict) -> str:
    scenes_json = json.dumps(scenes_data)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>saccade // Side-by-Side Interpretable Vision Transformer</title>
  <style>
    :root {{
      --bg: #080A0F;
      --card: #10141E;
      --border: #1E2638;
      --accent: #00FFCC;
      --accent-glow: rgba(0, 255, 204, 0.25);
      --danger: #FF3366;
      --text: #F1F5F9;
      --text-muted: #8E9BAE;
      --mono: 'SF Mono', Consolas, 'Fira Code', monospace;
      --sans: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 1.5rem 1rem 3rem 1rem;
      -webkit-font-smoothing: antialiased;
    }}
    header {{
      text-align: center;
      max-width: 900px;
      margin-bottom: 1.25rem;
    }}
    .tag {{
      display: inline-block;
      font-family: var(--mono);
      font-size: 0.72rem;
      color: var(--accent);
      background: rgba(0, 255, 204, 0.1);
      border: 1px solid var(--accent);
      padding: 0.2rem 0.75rem;
      border-radius: 9999px;
      margin-bottom: 0.6rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    h1 {{
      font-size: 1.85rem;
      font-weight: 700;
      color: #FFF;
      letter-spacing: -0.02em;
      margin-bottom: 0.4rem;
    }}
    .intro {{
      color: var(--text-muted);
      font-size: 0.92rem;
      line-height: 1.5;
    }}

    /* Scene Switcher Tabs */
    .tabs-bar {{
      display: flex;
      gap: 0.75rem;
      margin-bottom: 1.25rem;
      background: var(--card);
      border: 1px solid var(--border);
      padding: 0.35rem;
      border-radius: 10px;
    }}
    .tab-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 0.9rem;
      font-weight: 600;
      padding: 0.5rem 1.25rem;
      border-radius: 7px;
      cursor: pointer;
      transition: all 0.2s ease;
      font-family: var(--sans);
    }}
    .tab-btn:hover {{
      color: #FFF;
      background: rgba(255, 255, 255, 0.05);
    }}
    .tab-btn.active {{
      background: var(--accent);
      color: #080A0F;
      box-shadow: 0 0 12px var(--accent-glow);
    }}

    .main-grid {{
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      max-width: 1060px;
      width: 100%;
    }}

    /* Top Row: Side by Side Canvases + Predictions */
    .top-row {{
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 1.25rem;
      width: 100%;
    }}
    @media (max-width: 980px) {{
      .top-row {{ grid-template-columns: 1fr; }}
    }}

    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }}

    /* Dual Canvas Display */
    .dual-canvas-container {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      width: 100%;
    }}
    @media (max-width: 650px) {{
      .dual-canvas-container {{ grid-template-columns: 1fr; }}
    }}

    .canvas-panel {{
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .canvas-title {{
      font-family: var(--mono);
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .canvas-title.active-title {{
      color: var(--accent);
    }}

    .canvas-wrapper {{
      position: relative;
      width: 280px;
      height: 280px;
      max-width: 100%;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
      user-select: none;
      background: #000;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}

    /* Controls Bar */
    .controls-card {{
      margin-top: 1rem;
      background: rgba(255,255,255,0.02);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.85rem 1rem;
    }}
    .slider-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.4rem;
      font-size: 0.82rem;
      font-family: var(--mono);
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

    /* Filmstrip (Extracted Patches) */
    .filmstrip-card {{
      width: 100%;
    }}
    .filmstrip-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.75rem;
    }}
    .filmstrip-title {{
      font-family: var(--mono);
      font-size: 0.82rem;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .filmstrip-subtitle {{
      font-size: 0.75rem;
      color: var(--text-muted);
    }}
    .filmstrip-scroll {{
      display: flex;
      gap: 0.75rem;
      overflow-x: auto;
      padding-bottom: 0.6rem;
      scroll-behavior: smooth;
    }}
    .filmstrip-scroll::-webkit-scrollbar {{
      height: 6px;
    }}
    .filmstrip-scroll::-webkit-scrollbar-thumb {{
      background: var(--border);
      border-radius: 3px;
    }}
    .patch-card {{
      flex: 0 0 auto;
      display: flex;
      flex-direction: column;
      align-items: center;
      background: #090C12;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.5rem;
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
    }}
    .patch-card:hover {{
      border-color: var(--accent);
      transform: translateY(-2px);
    }}
    .patch-card.ablated {{
      border-color: var(--danger);
      background: rgba(255, 51, 102, 0.12);
    }}
    .patch-crop-canvas {{
      width: 54px;
      height: 54px;
      border-radius: 4px;
      border: 1px solid var(--border);
      display: block;
      margin-bottom: 0.35rem;
    }}
    .patch-badge {{
      font-family: var(--mono);
      font-size: 0.7rem;
      color: var(--accent);
      font-weight: 700;
    }}
    .patch-card.ablated .patch-badge {{
      color: var(--danger);
    }}
    .patch-action {{
      font-size: 0.65rem;
      color: var(--text-muted);
      margin-top: 0.2rem;
      font-family: var(--mono);
    }}

    /* Sidebar Predictions */
    .sidebar {{
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }}
    .section-title {{
      font-family: var(--mono);
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
      gap: 0.25rem;
      margin-bottom: 0.65rem;
    }}
    .pred-header {{
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
    }}
    .pred-name {{ font-weight: 600; color: #FFF; }}
    .pred-pct {{ font-family: var(--mono); color: var(--accent); }}
    .pred-bar-bg {{
      width: 100%;
      height: 7px;
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
      font-family: var(--mono);
      margin-bottom: 0.35rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .causal-stats {{
      font-family: var(--mono);
      color: var(--text);
      line-height: 1.4;
    }}

    .explanation-callout {{
      background: rgba(0, 255, 204, 0.05);
      border-left: 3px solid var(--accent);
      border-radius: 0 8px 8px 0;
      padding: 0.75rem 0.85rem;
      font-size: 0.78rem;
      color: var(--text);
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <header>
    <div class="tag">Pretrained Vision Transformer (DeiT-Small // ImageNet-1K)</div>
    <h1>saccade: Side-by-Side Causal Attention</h1>
    <p class="intro">
      No overlapping clutter. Compare the pristine original photo side-by-side with 
      <strong>what the transformer's attention bottleneck actually sees</strong>.
    </p>
  </header>

  <!-- Scene Switcher Tabs -->
  <div class="tabs-bar">
    <button class="tab-btn active" onclick="switchScene('panda')">🐼 Giant Panda</button>
    <button class="tab-btn" onclick="switchScene('cat')">🐱 Tabby Cat</button>
    <button class="tab-btn" onclick="switchScene('dog')">🐕 Dog / Samoyed</button>
  </div>

  <div class="main-grid">
    <!-- Top Row: Dual Canvases + Predictions -->
    <div class="top-row">
      <div class="card">
        <div class="dual-canvas-container">
          <!-- Left: Clean Original Image -->
          <div class="canvas-panel">
            <div class="canvas-title">1. Pristine Original Image</div>
            <div class="canvas-wrapper">
              <canvas id="origCanvas" width="280" height="280"></canvas>
            </div>
          </div>

          <!-- Right: What the Model Actually Sees (Isolated Patches) -->
          <div class="canvas-panel">
            <div class="canvas-title active-title">2. What the Model Sees (K Patches)</div>
            <div class="canvas-wrapper">
              <canvas id="modelCanvas" width="280" height="280"></canvas>
            </div>
          </div>
        </div>

        <div class="controls-card">
          <div class="slider-row">
            <span>Visual Patch Budget (K):</span>
            <span><span class="budget-val" id="kDisplay">12</span> / 196 patches (<span id="pctDisplay">6.1%</span> of pixels)</span>
          </div>
          <input type="range" id="kSlider" min="1" max="196" value="12">
        </div>
      </div>

      <!-- Right Column: Live Model Predictions -->
      <div class="sidebar">
        <div class="card">
          <div class="section-title">
            <span>Model Predictions</span>
            <span style="font-size: 0.7rem; color: var(--text-muted);">DeiT-Small (ImageNet)</span>
          </div>
          <div id="predictionsList"></div>
        </div>

        <div class="card">
          <div class="section-title">
            <span>Counterfactual Causal Test</span>
            <button style="background:transparent; border:1px solid var(--border); color:var(--text-muted); font-size:0.7rem; padding:0.15rem 0.4rem; border-radius:4px; cursor:pointer;" onclick="resetAblations()">Reset</button>
          </div>
          <div class="causal-box" id="causalBox">
            <div class="causal-title">
              <span>●</span> Causal Perturbation
            </div>
            <div class="causal-stats" id="causalStats">
              Click any patch in the filmstrip below to ablate it and observe the causal logit penalty.
            </div>
          </div>
        </div>

        <div class="explanation-callout" id="sceneHint"></div>
      </div>
    </div>

    <!-- Bottom Row: Extracted Patches Filmstrip Gallery -->
    <div class="card filmstrip-card">
      <div class="filmstrip-header">
        <div>
          <div class="filmstrip-title">Attended Patches Filmstrip (Ranked by Voting Score)</div>
          <div class="filmstrip-subtitle">These exact square tiles are the ONLY visual input received by downstream layers. Click any tile to ablate it.</div>
        </div>
      </div>
      <div class="filmstrip-scroll" id="filmstripScroll"></div>
    </div>
  </div>

  <script>
    const ALL_SCENES = {scenes_json};
    let currentSceneId = 'panda';
    let currentK = 12;
    let ablatedPatches = new Set();
    let hoveredPatch = null;

    const origCanvas = document.getElementById('origCanvas');
    const origCtx = origCanvas.getContext('2d');
    const modelCanvas = document.getElementById('modelCanvas');
    const modelCtx = modelCanvas.getContext('2d');

    const kSlider = document.getElementById('kSlider');
    const kDisplay = document.getElementById('kDisplay');
    const pctDisplay = document.getElementById('pctDisplay');
    const sceneHint = document.getElementById('sceneHint');
    const predictionsList = document.getElementById('predictionsList');
    const causalStats = document.getElementById('causalStats');
    const causalBox = document.getElementById('causalBox');
    const filmstripScroll = document.getElementById('filmstripScroll');

    const baseImg = new Image();

    function getScene() {{
      return ALL_SCENES[currentSceneId];
    }}

    function switchScene(sceneId) {{
      currentSceneId = sceneId;
      ablatedPatches.clear();
      document.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.getAttribute('onclick').includes(sceneId));
      }});
      const scene = getScene();
      sceneHint.textContent = scene.meta.hint;
      baseImg.src = scene.image_b64;
    }}

    baseImg.onload = () => {{
      render();
      updateUI();
      buildFilmstrip();
    }};

    function getTrajectoryForK(k) {{
      const scene = getScene();
      const avail = scene.available_k;
      let closest = avail[0];
      let minDiff = Math.abs(k - closest);
      for (const cand of avail) {{
        const diff = Math.abs(k - cand);
        if (diff < minDiff) {{
          minDiff = diff;
          closest = cand;
        }}
      }}
      return scene.trajectory[closest];
    }}

    function getActivePatches() {{
      const scene = getScene();
      const list = scene.ranked_indices.slice(0, currentK);
      return list.filter(idx => !ablatedPatches.has(idx));
    }}

    function render() {{
      const w = 280;
      const h = 280;
      const gw = 14;
      const gh = 14;
      const patchW = w / gw;
      const patchH = h / gh;

      // 1. Render Left Canvas: Pristine Original (100% clean, no clutter)
      origCtx.clearRect(0, 0, w, h);
      origCtx.drawImage(baseImg, 0, 0, w, h);

      // Delicate subtle outline if a patch is hovered
      if (hoveredPatch !== null) {{
        const r = Math.floor(hoveredPatch / gw);
        const c = hoveredPatch % gw;
        origCtx.strokeStyle = '#00FFCC';
        origCtx.lineWidth = 2;
        origCtx.strokeRect(c * patchW, r * patchH, patchW, patchH);
      }}

      // 2. Render Right Canvas: What the Model Actually Sees
      // Background is clean pitch black
      modelCtx.fillStyle = '#06080D';
      modelCtx.fillRect(0, 0, w, h);

      const activeList = getActivePatches();

      for (let rank = 0; rank < activeList.length; rank++) {{
        const idx = activeList[rank];
        const r = Math.floor(idx / gw);
        const c = idx % gw;
        const x = c * patchW;
        const y = r * patchH;

        // Draw ONLY the selected patch crop
        modelCtx.drawImage(baseImg, x, y, patchW, patchH, x, y, patchW, patchH);

        // Clean, delicate neon border around each isolated patch
        modelCtx.strokeStyle = (hoveredPatch === idx) ? '#FFF' : 'rgba(0, 255, 204, 0.7)';
        modelCtx.lineWidth = (hoveredPatch === idx) ? 2 : 1;
        modelCtx.strokeRect(x, y, patchW, patchH);
      }}

      // If any active patches are ablated, show red cross-hatch
      ablatedPatches.forEach(idx => {{
        const r = Math.floor(idx / gw);
        const c = idx % gw;
        const x = c * patchW;
        const y = r * patchH;

        modelCtx.fillStyle = 'rgba(255, 51, 102, 0.35)';
        modelCtx.fillRect(x, y, patchW, patchH);
        modelCtx.strokeStyle = '#FF3366';
        modelCtx.lineWidth = 1;
        modelCtx.strokeRect(x, y, patchW, patchH);
        modelCtx.beginPath();
        modelCtx.moveTo(x, y); modelCtx.lineTo(x + patchW, y + patchH);
        modelCtx.moveTo(x + patchW, y); modelCtx.lineTo(x, y + patchH);
        modelCtx.stroke();
      }});
    }}

    function buildFilmstrip() {{
      const scene = getScene();
      const topKIndices = scene.ranked_indices.slice(0, currentK);
      const gw = 14;
      const patchW = 280 / gw;
      const patchH = 280 / gw;

      filmstripScroll.innerHTML = '';

      topKIndices.forEach((pIdx, rank) => {{
        const isAblated = ablatedPatches.has(pIdx);
        const card = document.createElement('div');
        card.className = 'patch-card' + (isAblated ? ' ablated' : '');

        // Create micro crop canvas
        const pCanvas = document.createElement('canvas');
        pCanvas.width = 54;
        pCanvas.height = 54;
        pCanvas.className = 'patch-crop-canvas';
        const pCtx = pCanvas.getContext('2d');

        const r = Math.floor(pIdx / gw);
        const c = pIdx % gw;
        // Draw zoomed crop
        pCtx.drawImage(baseImg, c * patchW, r * patchH, patchW, patchH, 0, 0, 54, 54);

        if (isAblated) {{
          pCtx.fillStyle = 'rgba(255, 51, 102, 0.4)';
          pCtx.fillRect(0, 0, 54, 54);
          pCtx.strokeStyle = '#FF3366';
          pCtx.lineWidth = 2;
          pCtx.beginPath();
          pCtx.moveTo(0,0); pCtx.lineTo(54,54);
          pCtx.moveTo(54,0); pCtx.lineTo(0,54);
          pCtx.stroke();
        }}

        const badge = document.createElement('div');
        badge.className = 'patch-badge';
        badge.textContent = '#' + (rank + 1);

        const action = document.createElement('div');
        action.className = 'patch-action';
        action.textContent = isAblated ? 'ABLATED' : 'CLICK TO CUT';

        card.appendChild(pCanvas);
        card.appendChild(badge);
        card.appendChild(action);

        card.addEventListener('mouseenter', () => {{
          hoveredPatch = pIdx;
          render();
        }});
        card.addEventListener('mouseleave', () => {{
          hoveredPatch = null;
          render();
        }});
        card.addEventListener('click', () => {{
          if (ablatedPatches.has(pIdx)) {{
            ablatedPatches.delete(pIdx);
          }} else {{
            ablatedPatches.add(pIdx);
          }}
          render();
          updateUI();
          buildFilmstrip();
        }});

        filmstripScroll.appendChild(card);
      }});
    }}

    function updateUI() {{
      kDisplay.textContent = currentK;
      pctDisplay.textContent = ((currentK / 196) * 100).toFixed(1) + '%';

      const traj = getTrajectoryForK(currentK);
      if (traj && traj.top_preds) {{
        predictionsList.innerHTML = traj.top_preds.map((pred) => {{
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

      const scene = getScene();
      if (ablatedPatches.size > 0) {{
        causalBox.classList.add('active');
        let totalDelta = 0;
        let details = [];
        ablatedPatches.forEach(idx => {{
          const ab = scene.ablations[idx];
          if (ab) {{
            totalDelta += ab.delta_logit;
            details.push(`Patch #${{scene.ranked_indices.indexOf(idx) + 1}} (Δ Logit: -${{ab.delta_logit.toFixed(2)}})`);
          }}
        }});
        causalStats.innerHTML = `
          <strong>${{ablatedPatches.size}} patch(es) causally knocked out:</strong><br>
          ${{details.join('<br>')}}<br><br>
          <span style="color: var(--danger);">Target Class Logit Penalty: -${{totalDelta.toFixed(2)}}</span>
        `;
      }} else {{
        causalBox.classList.remove('active');
        causalStats.textContent = 'Click any patch in the filmstrip below to ablate it and observe the causal logit penalty.';
      }}
    }}

    kSlider.addEventListener('input', (e) => {{
      currentK = parseInt(e.target.value, 10);
      render();
      updateUI();
      buildFilmstrip();
    }});

    function resetAblations() {{
      ablatedPatches.clear();
      render();
      updateUI();
      buildFilmstrip();
    }}

    // Model canvas click to ablate
    modelCanvas.addEventListener('click', (e) => {{
      const rect = modelCanvas.getBoundingClientRect();
      const scaleX = modelCanvas.width / rect.width;
      const scaleY = modelCanvas.height / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;
      const col = Math.floor(x / (modelCanvas.width / 14));
      const row = Math.floor(y / (modelCanvas.height / 14));
      const pIdx = row * 14 + col;
      const scene = getScene();
      const topPatches = new Set(scene.ranked_indices.slice(0, currentK));
      if (topPatches.has(pIdx)) {{
        if (ablatedPatches.has(pIdx)) {{
          ablatedPatches.delete(pIdx);
        }} else {{
          ablatedPatches.add(pIdx);
        }}
        render();
        updateUI();
        buildFilmstrip();
      }}
    }});

    // Initialize
    switchScene('panda');
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
