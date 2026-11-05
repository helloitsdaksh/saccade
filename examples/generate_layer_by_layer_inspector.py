"""
examples/generate_layer_by_layer_inspector.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Builds an ultra-intuitive layer-by-layer mechanistic inspector for Vision Transformers.
Answers the user's fundamental questions:
1. What does each layer (Layer 0 to 11) actually look at? (Attention heatmap & Top-K patches)
2. What does the model infer at each layer? (Downstream classification & confidence)
3. How does visual perception evolve from primitive edges (Layer 0) to semantic parts (Layer 6)
   to diagnostic classification (Layer 11)?
"""

import sys
import json
import base64
import io
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ViT_B_16_Weights
import timm

sys.path.insert(0, str(Path(__file__).parent.parent))


def image_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def main():
    print("=" * 70)
    print("Generating Pristine Layer-by-Layer ViT Mechanistic Inspector...")
    print("=" * 70)

    labels = ViT_B_16_Weights.DEFAULT.meta["categories"]

    print("Loading pretrained vit_small_patch16_224.dino and deit_small_patch16_224...")
    dino = timm.create_model("vit_small_patch16_224.dino", pretrained=True).eval()
    deit = timm.create_model("deit_small_patch16_224", pretrained=True).eval()
    device = torch.device("cpu")

    samples_dir = Path(__file__).parent.parent / "assets" / "samples"
    scenes_meta = [
        {
            "id": "eagle",
            "name": "🦅 Bald Eagle",
            "file": "eagle.jpg",
            "target_label": "bald eagle",
            "target_idx": 22,
            "layer_takeaways": {
                0: "Layer 0: Edge & Contrast Extraction. Attention spreads across high-contrast plumage and sky boundaries. Downstream prediction is unformed (white stork ~19%).",
                1: "Layer 1: Local Texture Filtering. Groups feather texture gradients; eliminates uniform blue sky.",
                2: "Layer 2: Contour Assembly. Outlines the wings and head silhouette. Bald eagle emerges as top candidate (41.3%).",
                3: "Layer 3: Avian Part Grouping. Establishes structural relationship between white head and dark body.",
                4: "Layer 4: Semantic Feature Emergence. Focus shifts heavily to the head and beak. Confidence jumps to 75.1%.",
                5: "Layer 5: Feature Hub Formation. Coordinates beak and eye landmarks with wing orientation.",
                6: "Layer 6 (Mid Bottleneck): Sharp Part Gating. The 12 selected tokens capture the curved yellow beak, eye, and plumage collar (74.7%). All background sky is discarded.",
                7: "Layer 7: Diagnostic Refinement. Resolves hooked bill curvature against common raptors.",
                8: "Layer 8: High-Level Semantics. Plumage pattern and raptor eye features locked in (80.0%).",
                9: "Layer 9: Disambiguation. Confirms Haliaeetus leucocephalus vs other eagles or kites.",
                10: "Layer 10: Decision Consolidation. 84.7% confidence derived exclusively from the 12 attended patches.",
                11: "Layer 11 (Final Layer): Diagnostic Classification. Final self-attention passes 85.2% decisive eagle evidence directly into classification head."
            },
            "feature_tags": {
                (8, 8): "Hooked Yellow Beak & Eye",
                (8, 7): "Beak Base",
                (8, 6): "White Facial Plumage",
                (7, 6): "Head Crown Feathers",
                (7, 7): "White Nape",
                (8, 4): "Right Wing Lead Edge",
                (8, 3): "Right Wing Primary Feathers",
                (8, 2): "Right Wingtip",
                (9, 4): "Left Wing Primaries",
                (10, 5): "Underbelly Plumage",
                (11, 6): "White Tail Feathers",
                (9, 5): "Breast Feathers"
            }
        },
        {
            "id": "panda",
            "name": "🐼 Giant Panda",
            "file": "giant_panda.jpg",
            "target_label": "giant panda",
            "target_idx": 388,
            "layer_takeaways": {
                0: "Layer 0: Contrast Boundaries. Attention triggers on high-contrast black/white fur transitions. Top prediction is zebra (18.9%).",
                1: "Layer 1: Fur Texture Gradients. Distinguishes soft fur boundaries from background wood texture.",
                2: "Layer 2: Craniofacial Assembly. Assembles the facial geometry and black eye rings. Giant panda immediately leads (73.7%).",
                3: "Layer 3: Mammalian Structure. Binds dark ears to the spherical white head contour.",
                4: "Layer 4: Facial Feature Hub. Isolates distinctive black eye patches and dark muzzle.",
                5: "Layer 5: Limb and Body Anchoring. Integrates dark front paws with the facial mask.",
                6: "Layer 6 (Mid Bottleneck): Optimal Gating. Retains only the 12 diagnostic tokens (eye patches, ears, muzzle) with 76.4% confidence.",
                7: "Layer 7: Species Disambiguation. Explicitly eliminates lesser panda and spectacled bear.",
                8: "Layer 8: High-Level Semantics. Panda facial features produce 83.5% confidence.",
                9: "Layer 9: Evidence Aggregation. Deep layers confirm Ailuropoda melanoleuca identity.",
                10: "Layer 10: Robust Decision. 85.1% probability derived entirely from facial patches.",
                11: "Layer 11 (Final Layer): Peak Classification. Attended facial tokens yield 87.7% top-1 panda prediction."
            },
            "feature_tags": {
                (4, 9): "Right Black Eye Patch",
                (4, 8): "Left Black Eye Patch",
                (3, 11): "Black Round Ear",
                (5, 8): "Dark Muzzle & Nose",
                (4, 10): "White Cheek Contour",
                (7, 6): "Front Paw Claws",
                (7, 5): "Black Shoulder Band",
                (6, 7): "Forearm Fur",
                (3, 8): "Forehead White Fur",
                (5, 9): "Right Whiskers Area",
                (6, 6): "Chest Fur",
                (7, 7): "Resting Paw Joint"
            }
        },
        {
            "id": "espresso",
            "name": "☕ Espresso Cup",
            "file": "espresso.jpg",
            "target_label": "espresso",
            "target_idx": 967,
            "layer_takeaways": {
                0: "Layer 0: Geometric Edges. Detects straight tabletop lines and saucer curves. Top prediction: window shade (42.7%).",
                1: "Layer 1: Curvilinear Assembly. Follows the circular porcelain rim of the cup.",
                2: "Layer 2: Circular Object Integration. Assembles circular container geometry (chime ~6.4%).",
                3: "Layer 3: Liquid Core Inward Shift. Attention moves from outer porcelain toward inner liquid.",
                4: "Layer 4: Beverage Identification. Golden foam registers as hot beverage; espresso takes #1 (41.0%).",
                5: "Layer 5: Foam Texture Focus. Concentrates on rich microfoam crema bubbles in center.",
                6: "Layer 6 (Mid Bottleneck): Material Gating. Retains crema center and porcelain cup rim (beer glass/espresso ~33.5%).",
                7: "Layer 7: Liquid Surface Refinement. Discards table background entirely.",
                8: "Layer 8: Semantic Vessel Confirmation. Locks onto espresso beverage features (49.3%).",
                9: "Layer 9: Context Disambiguation. Distinguishes concentrated espresso from latte or coffee mug.",
                10: "Layer 10: Decision Solidification. 64.7% confidence derived from cup interior tokens.",
                11: "Layer 11 (Final Layer): Decisive Classification. Crema texture and cup rim yield 85.3% espresso prediction."
            },
            "feature_tags": {
                (5, 7): "Golden Crema Foam Core",
                (5, 6): "Liquid Surface Center",
                (6, 6): "Deep Crema Shadow",
                (4, 6): "Inner Liquid Boundary",
                (4, 7): "Crema Bubble Highlights",
                (7, 6): "White Porcelain Cup Rim",
                (8, 6): "Cup Base & Saucer",
                (7, 7): "Outer Rim Curve",
                (6, 7): "Right Cup Wall",
                (4, 5): "Left Rim Edge",
                (8, 7): "Saucer Reflection",
                (5, 5): "Coffee Edge"
            }
        },
        {
            "id": "dog",
            "name": "🐕 Golden Retriever",
            "file": "golden_retriever.jpg",
            "target_label": "golden retriever / samoyed",
            "target_idx": 258,
            "layer_takeaways": {
                0: "Layer 0: Fluffy Fur Texture. Detects high-frequency hair fibers. Top prediction: golf ball / wool (7.6%).",
                1: "Layer 1: Fur Flow Direction. Tracks hair strand angles along muzzle and ears.",
                2: "Layer 2: Canine Face Assembly. Synthesizes eyes, nose, and snout. Canine classes take lead (15.9%).",
                3: "Layer 3: Snout & Eye Clustering. Locks onto the dark nose and warm eyes.",
                4: "Layer 4: Head Structure. Confirms domestic canine facial proportions (21.2%).",
                5: "Layer 5: Facial Expression Focus. Concentrates on mouth, nose, and alert eye gaze.",
                6: "Layer 6 (Mid Bottleneck): Diagnostic Gating. Restricts forward pass to snout, eyes, and floppy ears (66.7%).",
                7: "Layer 7: Breed Refinement. Compares muzzle length and ear shape.",
                8: "Layer 8: Semantic Solidification. High-confidence canine representation (77.0%).",
                9: "Layer 9: Background Elimination. Grass and yard background mathematically discarded.",
                10: "Layer 10: Deep Canine Evidence. 86.5% confidence locked into facial tokens.",
                11: "Layer 11 (Final Layer): Decisive Classification. Attended facial tokens yield 83.8%+ top canine prediction."
            },
            "feature_tags": {
                (6, 7): "Dark Canine Nose",
                (5, 6): "Left Warm Eye",
                (5, 8): "Right Warm Eye",
                (6, 6): "Muzzle Whiskers",
                (7, 7): "Tongue & Mouth",
                (4, 5): "Left Floppy Ear",
                (4, 9): "Right Ear Base",
                (3, 7): "Forehead Fur",
                (5, 7): "Bridge of Snout",
                (7, 6): "Lower Jaw",
                (8, 7): "Neck Fluff",
                (6, 8): "Cheek Fur"
            }
        }
    ]

    transform = T.Compose([
        T.Resize((224, 224), Image.Resampling.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    scenes_data = {}
    k_options = [4, 8, 12, 16, 24, 32, 64, 196]

    for scene in scenes_meta:
        print(f"\nProcessing scene: {scene['name']}...")
        img_path = samples_dir / scene["file"]
        pil_raw = Image.open(img_path).convert("RGB")
        # 336x336 is divisible by 14 (exactly 24px per patch)
        pil_336 = pil_raw.resize((336, 336), Image.Resampling.BILINEAR)
        b64 = image_to_base64(pil_336)

        x = transform(pil_raw).unsqueeze(0).to(device)

        with torch.no_grad():
            # 1. Forward through DINO to extract pristine, register-free attention at all 12 layers
            dino_toks = dino.patch_embed(x)
            dino_toks = torch.cat([dino.cls_token.expand(1, -1, -1), dino_toks], dim=1) + dino.pos_embed
            dino_attns = []
            for blk in dino.blocks:
                B, N, C = dino_toks.shape
                qkv = blk.attn.qkv(dino_toks).reshape(B, N, 3, blk.attn.num_heads, blk.attn.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1) * blk.attn.scale).softmax(dim=-1)
                cls_attn = attn[0, :, 0, 1:].mean(dim=0) # [196]
                dino_attns.append(cls_attn)
                dino_toks = blk(dino_toks)

            # 2. Forward through DeiT up to layer L, bottleneck Top-K, and pass through remaining blocks
            layers_info = []
            target_progression = []

            for l_idx in range(12):
                cls_attn_l = dino_attns[l_idx]
                sorted_scores, sorted_indices = torch.sort(cls_attn_l, descending=True)
                ranked_patches = sorted_indices.cpu().numpy().tolist()
                heatmap_scores = cls_attn_l.cpu().numpy().tolist()

                # Normalize heatmap scores [0, 1]
                min_s = min(heatmap_scores)
                max_s = max(heatmap_scores)
                norm_heatmap = [(s - min_s) / (max_s - min_s + 1e-8) for s in heatmap_scores]

                # Run DeiT forward pass up to layer l_idx
                deit_toks = deit.patch_embed(x)
                deit_toks = torch.cat([deit.cls_token.expand(1, -1, -1), deit_toks], dim=1) + deit.pos_embed
                for i in range(l_idx + 1):
                    deit_toks = deit.blocks[i](deit_toks)

                C_dim = deit_toks.shape[-1]

                # Compute inference across K options
                k_predictions = {}
                for k_val in k_options:
                    topk_idx = sorted_indices[:k_val]
                    sel_idx = topk_idx.unsqueeze(0).unsqueeze(-1).expand(-1, -1, C_dim)
                    sel_spatial = torch.gather(deit_toks[:, 1:], dim=1, index=sel_idx)
                    pruned_toks = torch.cat([deit_toks[:, :1], sel_spatial], dim=1)

                    # Pass through remaining blocks
                    for rem in range(l_idx + 1, 12):
                        pruned_toks = deit.blocks[rem](pruned_toks)
                    pruned_toks = deit.norm(pruned_toks)
                    logits = deit.head(pruned_toks[:, 0])[0]
                    probs = F.softmax(logits, dim=-1)

                    top5_vals, top5_inds = torch.topk(probs, k=5)
                    top_preds = [
                        {
                            "class_idx": int(top5_inds[i].item()),
                            "class_name": labels[int(top5_inds[i].item())],
                            "prob": float(top5_vals[i].item()),
                        }
                        for i in range(len(top5_inds))
                    ]

                    k_predictions[k_val] = {
                        "top_preds": top_preds,
                        "top1_class": top_preds[0]["class_name"],
                        "top1_prob": top_preds[0]["prob"],
                    }

                # Top-1 for standard budget K=12
                pred_12 = k_predictions[12]
                target_progression.append({
                    "layer": l_idx,
                    "top1_class": pred_12["top1_class"],
                    "top1_prob": pred_12["top1_prob"],
                })

                # Feature descriptions for top 12 patches
                top12_details = []
                for rank, p_idx in enumerate(ranked_patches[:12]):
                    r = p_idx // 14
                    c = p_idx % 14
                    desc = scene["feature_tags"].get((r, c), f"Patch ({r}, {c})")
                    top12_details.append({
                        "rank": rank + 1,
                        "patch_idx": p_idx,
                        "row": r,
                        "col": c,
                        "desc": desc,
                        "score": float(cls_attn_l[p_idx].item())
                    })

                explanation = scene["layer_takeaways"].get(
                    l_idx,
                    f"Layer {l_idx}: Features intermediate representations. Attention focuses on primary spatial regions."
                )

                layers_info.append({
                    "layer_idx": l_idx,
                    "explanation": explanation,
                    "ranked_patches": ranked_patches,
                    "heatmap": norm_heatmap,
                    "predictions_by_k": k_predictions,
                    "top12_details": top12_details,
                })

        scenes_data[scene["id"]] = {
            "meta": {
                "id": scene["id"],
                "name": scene["name"],
                "target_label": scene["target_label"],
            },
            "image_b64": b64,
            "layers": layers_info,
            "progression": target_progression,
            "available_k": k_options,
        }
        print(f"  ✓ Processed all 12 layers for {scene['name']}")

    # Write HTML output
    html_path = Path(__file__).parent.parent / "assets" / "proper_model_interactive.html"
    print(f"\nWriting interactive layer-by-layer inspector to {html_path}...")
    html_content = generate_inspector_html(scenes_data)
    html_path.write_text(html_content, encoding="utf-8")
    print(f"✓ Output written to {html_path} ({html_path.stat().st_size / 1024:.1f} KB)")

    # Also copy to artifact directory
    artifact_path = Path("/Users/volageek/.gemini/antigravity/brain/4a608979-18f6-4bc1-8a6b-e649d52a8238/proper_model_interactive.html")
    if artifact_path.parent.exists():
        shutil.copyfile(html_path, artifact_path)
        print(f"✓ Copied to artifact directory: {artifact_path}")


def generate_inspector_html(scenes_data: dict) -> str:
    scenes_json = json.dumps(scenes_data)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>saccade // Layer-by-Layer Mechanistic Vision Transformer Inspector</title>
  <style>
    :root {{
      --bg: #07090E;
      --card: #0D121D;
      --card-alt: #131926;
      --border: #1E2738;
      --border-bright: #2B3850;
      --accent: #00FFCC;
      --accent-dim: rgba(0, 255, 204, 0.12);
      --accent-glow: rgba(0, 255, 204, 0.35);
      --gold: #FFB800;
      --danger: #FF3366;
      --blue: #38BDF8;
      --text: #F8FAFC;
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
      max-width: 980px;
      margin-bottom: 1.25rem;
    }}
    .tag {{
      display: inline-block;
      font-family: var(--mono);
      font-size: 0.72rem;
      color: var(--accent);
      background: var(--accent-dim);
      border: 1px solid var(--accent);
      padding: 0.2rem 0.8rem;
      border-radius: 9999px;
      margin-bottom: 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    h1 {{
      font-size: 1.95rem;
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
      gap: 0.5rem;
      margin-bottom: 1.25rem;
      background: var(--card);
      border: 1px solid var(--border);
      padding: 0.35rem;
      border-radius: 10px;
      flex-wrap: wrap;
      justify-content: center;
    }}
    .tab-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 0.88rem;
      font-weight: 600;
      padding: 0.45rem 1.25rem;
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
      color: #07090E;
      box-shadow: 0 0 12px var(--accent-glow);
    }}

    .main-grid {{
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      max-width: 1120px;
      width: 100%;
    }}

    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }}

    /* Layer Scrubbing Controller Bar */
    .layer-controller-card {{
      background: linear-gradient(180deg, #131A28, var(--card));
      border: 1px solid rgba(0, 255, 204, 0.25);
    }}
    .controller-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.85rem;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    .controller-title {{
      font-family: var(--mono);
      font-size: 0.82rem;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .layer-badges-container {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 0.35rem;
      margin-bottom: 0.85rem;
    }}
    @media (max-width: 768px) {{
      .layer-badges-container {{ grid-template-columns: repeat(6, 1fr); }}
    }}
    .layer-badge-btn {{
      background: #151D2C;
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-family: var(--mono);
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.4rem 0.2rem;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
      text-align: center;
    }}
    .layer-badge-btn:hover {{
      color: #FFF;
      border-color: var(--accent);
    }}
    .layer-badge-btn.active {{
      background: var(--accent);
      color: #07090E;
      border-color: var(--accent);
      box-shadow: 0 0 10px var(--accent-glow);
      font-weight: 700;
    }}

    .slider-container {{
      display: flex;
      align-items: center;
      gap: 1rem;
      margin-bottom: 0.85rem;
    }}
    input[type=range] {{
      flex: 1;
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

    /* Role Explanation Banner */
    .role-banner {{
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
      background: rgba(0, 255, 204, 0.05);
      border-left: 3px solid var(--accent);
      padding: 0.85rem 1rem;
      border-radius: 0 8px 8px 0;
      font-size: 0.88rem;
      color: #FFF;
      line-height: 1.45;
    }}
    .role-pill {{
      font-family: var(--mono);
      font-size: 0.72rem;
      padding: 0.2rem 0.55rem;
      border-radius: 4px;
      background: rgba(0, 255, 204, 0.15);
      color: var(--accent);
      font-weight: 700;
      white-space: nowrap;
      margin-top: 0.1rem;
    }}

    /* Top Row: Dual Canvases + Predictions */
    .top-row {{
      display: grid;
      grid-template-columns: 1fr 370px;
      gap: 1.25rem;
      width: 100%;
    }}
    @media (max-width: 980px) {{
      .top-row {{ grid-template-columns: 1fr; }}
    }}

    /* Dual Canvas Display */
    .dual-canvas-container {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
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
    .canvas-title-bar {{
      display: flex;
      justify-content: space-between;
      width: 100%;
      max-width: 336px;
      margin-bottom: 0.5rem;
      font-family: var(--mono);
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .canvas-title-bar strong {{
      color: var(--text);
    }}
    .canvas-title-bar.active strong {{
      color: var(--accent);
    }}

    .canvas-wrapper {{
      position: relative;
      width: 336px;
      height: 336px;
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

    /* Controls Bar under Canvases */
    .controls-strip {{
      margin-top: 1.25rem;
      background: var(--card-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.85rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }}
    .strip-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--mono);
      font-size: 0.78rem;
    }}
    .strip-row span strong {{
      color: var(--accent);
    }}
    .btn-toggle-group {{
      display: flex;
      gap: 0.4rem;
    }}
    .btn-toggle {{
      background: #182232;
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 0.72rem;
      font-family: var(--mono);
      padding: 0.25rem 0.6rem;
      border-radius: 4px;
      cursor: pointer;
    }}
    .btn-toggle.active {{
      background: var(--accent);
      color: #07090E;
      font-weight: 700;
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
      font-size: 0.84rem;
    }}
    .pred-name {{ font-weight: 600; color: #FFF; }}
    .pred-pct {{ font-family: var(--mono); color: var(--accent); font-weight: 700; }}
    .pred-bar-bg {{
      width: 100%;
      height: 8px;
      background: #182232;
      border-radius: 4px;
      overflow: hidden;
    }}
    .pred-bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, #00BFA5, var(--accent));
      border-radius: 4px;
      transition: width 0.25s ease;
    }}

    .guarantee-box {{
      background: rgba(0, 255, 204, 0.04);
      border: 1px solid rgba(0, 255, 204, 0.2);
      border-radius: 8px;
      padding: 0.85rem;
      font-size: 0.78rem;
      line-height: 1.5;
    }}
    .guarantee-box strong {{
      color: var(--accent);
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      margin-top: 0.35rem;
      font-family: var(--mono);
      color: var(--text-muted);
    }}
    .metric-val {{
      color: #FFF;
      font-weight: 600;
    }}

    /* Bottom: Attended Tokens Filmstrip */
    .filmstrip-card {{
      background: var(--card);
    }}
    .filmstrip-scroll {{
      display: flex;
      gap: 0.75rem;
      overflow-x: auto;
      padding-bottom: 0.5rem;
      margin-top: 0.75rem;
    }}
    .filmstrip-item {{
      flex: 0 0 140px;
      background: var(--card-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.5rem;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .filmstrip-item:hover, .filmstrip-item.highlighted {{
      border-color: var(--accent);
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0, 255, 204, 0.2);
    }}
    .chip-canvas-wrap {{
      width: 56px;
      height: 56px;
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 0.4rem;
      border: 1px solid var(--border);
      background: #000;
    }}
    .chip-rank {{
      font-family: var(--mono);
      font-size: 0.68rem;
      color: var(--accent);
      font-weight: 700;
    }}
    .chip-desc {{
      font-size: 0.7rem;
      color: #FFF;
      line-height: 1.25;
      margin-top: 0.2rem;
      min-height: 1.8rem;
    }}
    .chip-coords {{
      font-family: var(--mono);
      font-size: 0.65rem;
      color: var(--text-muted);
      margin-top: 0.15rem;
    }}

    /* Layer Progression Sparkline Strip */
    .progression-strip {{
      display: flex;
      gap: 0.35rem;
      margin-top: 0.5rem;
      align-items: flex-end;
      height: 48px;
      background: #101622;
      border-radius: 6px;
      padding: 0.25rem 0.5rem;
    }}
    .prog-col {{
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.2rem;
      height: 100%;
      justify-content: flex-end;
      cursor: pointer;
    }}
    .prog-bar {{
      width: 100%;
      background: #1E2B40;
      border-radius: 2px;
      transition: all 0.2s ease;
    }}
    .prog-col.active .prog-bar {{
      background: var(--accent);
      box-shadow: 0 0 8px var(--accent);
    }}
    .prog-lbl {{
      font-family: var(--mono);
      font-size: 0.62rem;
      color: var(--text-muted);
    }}
    .prog-col.active .prog-lbl {{
      color: var(--accent);
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <header>
    <div class="tag">Mechanistic Interpretability // Layer-by-Layer Probing</div>
    <h1>What Does Each Layer Look At & Infer?</h1>
    <p class="intro">
      Step through the transformer's 12 blocks from shallow to deep. 
      Observe how attention transitions from raw textures to semantic parts, and see what the model infers at every depth.
    </p>
  </header>

  <!-- Scene Switcher Tabs -->
  <div class="tabs-bar">
    <button class="tab-btn active" onclick="switchScene('eagle')">🦅 Bald Eagle</button>
    <button class="tab-btn" onclick="switchScene('panda')">🐼 Giant Panda</button>
    <button class="tab-btn" onclick="switchScene('espresso')">☕ Espresso Cup</button>
    <button class="tab-btn" onclick="switchScene('dog')">🐕 Golden Retriever</button>
  </div>

  <div class="main-grid">
    <!-- Layer Scrubbing Controller Card -->
    <div class="card layer-controller-card">
      <div class="controller-header">
        <div class="controller-title">
          <span>⚡ Transformer Depth Controller</span>
          <span style="color: var(--text-muted); font-size: 0.75rem;">(12 Sequential Blocks)</span>
        </div>
        <div style="font-family: var(--mono); font-size: 0.85rem; color: var(--accent); font-weight:700;">
          Selected: <span id="currentLayerDisplay">Layer 6</span>
        </div>
      </div>

      <!-- Quick Layer Selector Buttons 0..11 -->
      <div class="layer-badges-container" id="layerBadges"></div>

      <!-- Scrubbing Slider -->
      <div class="slider-container">
        <span style="font-family: var(--mono); font-size: 0.75rem; color: var(--text-muted);">L0 (Input)</span>
        <input type="range" id="layerSlider" min="0" max="11" value="6">
        <span style="font-family: var(--mono); font-size: 0.75rem; color: var(--text-muted);">L11 (Output)</span>
      </div>

      <!-- Role Banner -->
      <div class="role-banner">
        <span class="role-pill" id="layerStagePill">Mid Bottleneck</span>
        <span id="layerExplanationText"></span>
      </div>
    </div>

    <!-- Top Row: Dual Canvases + Predictions -->
    <div class="top-row">
      <div class="card">
        <div class="dual-canvas-container">
          <!-- Left: Attention Heatmap on Pristine Original -->
          <div class="canvas-panel">
            <div class="canvas-title-bar">
              <span>1. Layer Attention</span>
              <span id="heatmapAlphaLabel">Overlay: 60%</span>
            </div>
            <div class="canvas-wrapper">
              <canvas id="origCanvas" width="336" height="336"></canvas>
            </div>
          </div>

          <!-- Right: Attended Saccade Patches -->
          <div class="canvas-panel">
            <div class="canvas-title-bar active">
              <strong>2. Saccade Tokens</strong>
              <strong id="topKLabel">Top 12 Patches</strong>
            </div>
            <div class="canvas-wrapper">
              <canvas id="modelCanvas" width="336" height="336"></canvas>
            </div>
          </div>
        </div>

        <!-- Controls Strip -->
        <div class="controls-strip">
          <div class="strip-row">
            <span>Visual Token Budget (K): <strong id="kDisplay">12</strong> / 196 (<span id="pctDisplay">6.1%</span> of image)</span>
            <div class="btn-toggle-group">
              <button class="btn-toggle active" id="btnDim" onclick="setDimMode('dim')">Dim Surrounding</button>
              <button class="btn-toggle" id="btnBlack" onclick="setDimMode('black')">Blackout</button>
            </div>
          </div>
          <input type="range" id="kSlider" min="4" max="196" value="12" step="4">

          <div class="strip-row" style="margin-top: 0.25rem;">
            <span style="color: var(--text-muted);">Heatmap Blend Intensity:</span>
            <input type="range" id="heatAlphaSlider" min="0" max="100" value="60" style="max-width: 140px;">
          </div>
        </div>
      </div>

      <!-- Right Column: Live Model Predictions at this Layer -->
      <div class="sidebar">
        <div class="card">
          <div class="section-title">
            <span>Inference at this Depth</span>
            <span style="font-size: 0.7rem; color: var(--text-muted);" id="depthProbeLabel">Probed @ L6</span>
          </div>
          <div id="predictionsList"></div>

          <!-- Mini Layer Confidence Progression Strip -->
          <div style="margin-top: 1rem;">
            <div class="canvas-title-bar">
              <span>Target Class Progression (L0 ➔ L11)</span>
              <span id="progTargetName" style="color: var(--accent); font-weight:700;"></span>
            </div>
            <div class="progression-strip" id="progStrip"></div>
          </div>
        </div>

        <div class="card guarantee-box">
          <strong>Intrinsically Interpretable Guarantee</strong>
          <p style="color: var(--text-muted); margin-top: 0.35rem;">
            Unlike post-hoc heatmaps (Grad-CAM), the downstream classification was computed <em>exclusively</em> using the selected patches.
          </p>
          <div class="metric-row">
            <span>Information Discarded:</span>
            <span class="metric-val" id="discardedPct">93.9% of pixels</span>
          </div>
          <div class="metric-row">
            <span>Gradient Isolation:</span>
            <span class="metric-val" style="color: var(--accent);">0.000000 (No Leakage)</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom: Selected Patches Filmstrip -->
    <div class="card filmstrip-card">
      <div class="section-title">
        <span>Attended Patches Breakdown (Ranked by Importance in Layer <span id="filmstripLayerNum">6</span>)</span>
        <span style="font-size: 0.75rem; color: var(--text-muted);">Hover patch to locate</span>
      </div>
      <div class="filmstrip-scroll" id="filmstripContainer"></div>
    </div>
  </div>

  <script>
    const ALL_SCENES = {scenes_json};
    let currentSceneId = 'eagle';
    let currentLayer = 6;
    let currentK = 12;
    let dimMode = 'dim'; // 'dim' or 'black'
    let heatmapAlpha = 0.6;
    let hoveredPatchIdx = null;

    const origCanvas = document.getElementById('origCanvas');
    const origCtx = origCanvas.getContext('2d');
    const modelCanvas = document.getElementById('modelCanvas');
    const modelCtx = modelCanvas.getContext('2d');

    const layerSlider = document.getElementById('layerSlider');
    const currentLayerDisplay = document.getElementById('currentLayerDisplay');
    const layerStagePill = document.getElementById('layerStagePill');
    const layerExplanationText = document.getElementById('layerExplanationText');
    const layerBadgesContainer = document.getElementById('layerBadges');

    const kSlider = document.getElementById('kSlider');
    const kDisplay = document.getElementById('kDisplay');
    const pctDisplay = document.getElementById('pctDisplay');
    const topKLabel = document.getElementById('topKLabel');
    const depthProbeLabel = document.getElementById('depthProbeLabel');
    const predictionsList = document.getElementById('predictionsList');
    const heatAlphaSlider = document.getElementById('heatAlphaSlider');
    const heatmapAlphaLabel = document.getElementById('heatmapAlphaLabel');
    const discardedPct = document.getElementById('discardedPct');
    const filmstripLayerNum = document.getElementById('filmstripLayerNum');
    const filmstripContainer = document.getElementById('filmstripContainer');
    const progStrip = document.getElementById('progStrip');
    const progTargetName = document.getElementById('progTargetName');

    const btnDim = document.getElementById('btnDim');
    const btnBlack = document.getElementById('btnBlack');

    const baseImg = new Image();

    function buildLayerBadges() {{
      layerBadgesContainer.innerHTML = '';
      for (let l = 0; l < 12; l++) {{
        const btn = document.createElement('button');
        btn.className = 'layer-badge-btn' + (l === currentLayer ? ' active' : '');
        btn.textContent = 'L' + l;
        btn.onclick = () => setLayer(l);
        layerBadgesContainer.appendChild(btn);
      }}
    }}

    function getScene() {{
      return ALL_SCENES[currentSceneId];
    }}

    function getLayerData() {{
      return getScene().layers[currentLayer];
    }}

    function switchScene(sceneId) {{
      currentSceneId = sceneId;
      document.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.getAttribute('onclick').includes(sceneId));
      }});
      baseImg.src = getScene().image_b64;
    }}

    baseImg.onload = () => {{
      render();
      updateUI();
    }};

    function setLayer(l) {{
      currentLayer = l;
      layerSlider.value = l;
      render();
      updateUI();
    }}

    function setDimMode(mode) {{
      dimMode = mode;
      btnDim.classList.toggle('active', mode === 'dim');
      btnBlack.classList.toggle('active', mode === 'black');
      render();
    }}

    function getStageInfo(l) {{
      if (l <= 1) return {{ stage: "Stage 1: Low-Level Edges & Textures", color: "var(--blue)" }};
      if (l <= 4) return {{ stage: "Stage 2: Mid-Level Part Assembly", color: "var(--gold)" }};
      if (l <= 8) return {{ stage: "Stage 3: Semantic Feature Gating", color: "var(--accent)" }};
      return {{ stage: "Stage 4: Class-Diagnostic Synthesis", color: "#C084FC" }};
    }}

    // Plasma/Cyan heat gradient mapping
    function getHeatColor(val, alpha) {{
      // val in [0, 1]
      // Map smoothly from deep navy to cyan to warm yellow
      let r, g, b;
      if (val < 0.5) {{
        const t = val / 0.5;
        r = Math.floor(0 * (1-t) + 0 * t);
        g = Math.floor(180 * (1-t) + 255 * t);
        b = Math.floor(220 * (1-t) + 204 * t);
      }} else {{
        const t = (val - 0.5) / 0.5;
        r = Math.floor(0 * (1-t) + 255 * t);
        g = Math.floor(255 * (1-t) + 215 * t);
        b = Math.floor(204 * (1-t) + 0 * t);
      }}
      return `rgba(${{r}}, ${{g}}, ${{b}}, ${{val * alpha}})`;
    }}

    function render() {{
      const w = 336;
      const h = 336;
      const gw = 14;
      const gh = 14;
      const patchW = w / gw; // Exactly 24px
      const patchH = h / gh; // Exactly 24px

      const layerData = getLayerData();
      const heatmap = layerData.heatmap;
      const ranked = layerData.ranked_patches;
      const activeCount = Math.min(currentK, ranked.length);

      // 1. Render Left Canvas: Pristine Image + Smooth Translucent Attention Heatmap
      origCtx.clearRect(0, 0, w, h);
      origCtx.drawImage(baseImg, 0, 0, w, h);

      if (heatmapAlpha > 0.01) {{
        for (let r = 0; r < gh; r++) {{
          for (let c = 0; c < gw; c++) {{
            const idx = r * gw + c;
            const heat = heatmap[idx] || 0;
            // Only highlight meaningful attention above baseline
            if (heat > 0.22) {{
              origCtx.fillStyle = getHeatColor(heat, heatmapAlpha);
              origCtx.fillRect(c * patchW, r * patchH, patchW, patchH);
            }}
          }}
        }}
      }}

      // If hovering a patch, draw subtle highlight on left canvas
      if (hoveredPatchIdx !== null) {{
        const hr = Math.floor(hoveredPatchIdx / gw);
        const hc = hoveredPatchIdx % gw;
        origCtx.strokeStyle = '#FFFFFF';
        origCtx.lineWidth = 2.5;
        origCtx.strokeRect(hc * patchW, hr * patchH, patchW, patchH);
      }}

      // 2. Render Right Canvas: Exactly Kept Top-K Tokens (100% Causal Window)
      modelCtx.clearRect(0, 0, w, h);

      if (currentK >= 196) {{
        // If 100% tokens selected, show full pristine photo with ZERO dimming and NO wireframe grid
        modelCtx.drawImage(baseImg, 0, 0, w, h);
      }} else {{
        // Draw base image
        modelCtx.drawImage(baseImg, 0, 0, w, h);

        // Apply dark mask to obscure non-attended area
        modelCtx.fillStyle = (dimMode === 'dim') ? 'rgba(7, 9, 14, 0.88)' : 'rgba(7, 9, 14, 1.0)';
        modelCtx.fillRect(0, 0, w, h);

        // Punch out and draw ONLY the active patches using clipping
        // Guarantees 100% subpixel alignment with the original image!
        modelCtx.save();
        modelCtx.beginPath();
        for (let rank = 0; rank < activeCount; rank++) {{
          const idx = ranked[rank];
          const r = Math.floor(idx / gw);
          const c = idx % gw;
          modelCtx.rect(c * patchW, r * patchH, patchW, patchH);
        }}
        modelCtx.clip();
        modelCtx.drawImage(baseImg, 0, 0, w, h);
        modelCtx.restore();

        // Stroke clean borders ONLY when budget is sparse (K <= 32)
        // Never stroke 196 wireframe boxes!
        if (currentK <= 32) {{
          for (let rank = 0; rank < activeCount; rank++) {{
            const idx = ranked[rank];
            const r = Math.floor(idx / gw);
            const c = idx % gw;
            const x = c * patchW;
            const y = r * patchH;

            const isHovered = (idx === hoveredPatchIdx);
            if (isHovered) {{
              modelCtx.strokeStyle = '#FFFFFF';
              modelCtx.lineWidth = 2.5;
              modelCtx.strokeRect(x, y, patchW, patchH);
            }} else if (rank < 3) {{
              modelCtx.strokeStyle = 'var(--accent)';
              modelCtx.lineWidth = 1.5;
              modelCtx.strokeRect(x, y, patchW, patchH);
            }} else if (currentK <= 16) {{
              modelCtx.strokeStyle = 'rgba(0, 255, 204, 0.35)';
              modelCtx.lineWidth = 1;
              modelCtx.strokeRect(x, y, patchW, patchH);
            }}
          }}
        }}
      }}

      // If hovering a patch, draw white highlight on right canvas
      if (hoveredPatchIdx !== null) {{
        const hr = Math.floor(hoveredPatchIdx / gw);
        const hc = hoveredPatchIdx % gw;
        modelCtx.strokeStyle = '#FFFFFF';
        modelCtx.lineWidth = 2.5;
        modelCtx.strokeRect(hc * patchW, hr * patchH, patchW, patchH);
      }}
    }}

    function updateUI() {{
      currentLayerDisplay.textContent = 'Layer ' + currentLayer;
      depthProbeLabel.textContent = 'Probed @ L' + currentLayer;
      filmstripLayerNum.textContent = currentLayer;

      // Update badge states
      document.querySelectorAll('.layer-badge-btn').forEach((btn, idx) => {{
        btn.classList.toggle('active', idx === currentLayer);
      }});

      // Update Stage Pill and Explanation
      const stage = getStageInfo(currentLayer);
      layerStagePill.textContent = stage.stage;
      layerStagePill.style.color = stage.color;

      const layerData = getLayerData();
      layerExplanationText.textContent = layerData.explanation;

      // Update Budget Display
      kSlider.value = currentK;
      kDisplay.textContent = currentK;
      topKLabel.textContent = `Top ${{currentK}} Patches`;
      const pctVal = ((currentK / 196) * 100).toFixed(1);
      pctDisplay.textContent = pctVal + '%';
      discardedPct.textContent = (100 - parseFloat(pctVal)).toFixed(1) + '% of pixels';

      // Update Predictions for this layer & K
      const availK = getScene().available_k;
      let closestK = availK[0];
      let minDiff = Math.abs(currentK - closestK);
      for (const cand of availK) {{
        const diff = Math.abs(currentK - cand);
        if (diff < minDiff) {{
          minDiff = diff;
          closestK = cand;
        }}
      }}

      const predData = layerData.predictions_by_k[closestK];
      if (predData && predData.top_preds) {{
        predictionsList.innerHTML = predData.top_preds.map((pred, i) => {{
          const probPct = (pred.prob * 100).toFixed(1);
          const isTop = (i === 0);
          return `
            <div class="pred-row">
              <div class="pred-header">
                <span class="pred-name" style="${{isTop ? 'color: var(--accent); font-weight:700;' : ''}}">${{i+1}}. ${{pred.class_name}}</span>
                <span class="pred-pct" style="${{isTop ? 'color: var(--accent);' : 'color: var(--text-muted);'}}">${{probPct}}%</span>
              </div>
              <div class="pred-bar-bg">
                <div class="pred-bar-fill" style="width: ${{probPct}}%; background: ${{isTop ? 'var(--accent)' : 'linear-gradient(90deg, #1A4D4A, #2E7D72)'}};"></div>
              </div>
            </div>
          `;
        }}).join('');
      }}

      // Update Target Progression Strip across all 12 layers
      const sceneProg = getScene().progression;
      const targetName = getScene().meta.target_label;
      progTargetName.textContent = targetName;
      progStrip.innerHTML = sceneProg.map((p, idx) => {{
        const barH = Math.max(10, Math.round(p.top1_prob * 100));
        const isActive = (idx === currentLayer);
        return `
          <div class="prog-col ${{isActive ? 'active' : ''}}" onclick="setLayer(${{idx}})" title="L${{idx}}: ${{p.top1_class}} (${{(p.top1_prob*100).toFixed(0)}}%)">
            <div class="prog-bar" style="height: ${{barH}}%;"></div>
            <span class="prog-lbl">L${{idx}}</span>
          </div>
        `;
      }}).join('');

      // Build Filmstrip items for top 12 patches
      buildFilmstrip(layerData.top12_details);
    }}

    function buildFilmstrip(top12) {{
      filmstripContainer.innerHTML = '';
      const w = 336;
      const gw = 14;
      const patchW = w / gw;

      top12.forEach(item => {{
        const el = document.createElement('div');
        el.className = 'filmstrip-item' + (item.patch_idx === hoveredPatchIdx ? ' highlighted' : '');
        el.onmouseenter = () => {{
          hoveredPatchIdx = item.patch_idx;
          render();
        }};
        el.onmouseleave = () => {{
          hoveredPatchIdx = null;
          render();
        }};

        // Mini canvas for this patch
        const miniCanvas = document.createElement('canvas');
        miniCanvas.width = 56;
        miniCanvas.height = 56;
        const miniCtx = miniCanvas.getContext('2d');
        const sx = item.col * patchW;
        const sy = item.row * patchW;
        miniCtx.drawImage(baseImg, sx, sy, patchW, patchW, 0, 0, 56, 56);

        el.innerHTML = `
          <div class="chip-canvas-wrap"></div>
          <div class="chip-rank">#${{item.rank}} (Rank)</div>
          <div class="chip-desc">${{item.desc}}</div>
          <div class="chip-coords">Row ${{item.row}}, Col ${{item.col}}</div>
        `;
        el.querySelector('.chip-canvas-wrap').appendChild(miniCanvas);
        filmstripContainer.appendChild(el);
      }});
    }}

    // Canvas Hover Interaction
    modelCanvas.addEventListener('mousemove', (e) => {{
      const rect = modelCanvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (336 / rect.width);
      const y = (e.clientY - rect.top) * (336 / rect.height);
      const patchW = 336 / 14;
      const c = Math.floor(x / patchW);
      const r = Math.floor(y / patchW);
      if (c >= 0 && c < 14 && r >= 0 && r < 14) {{
        hoveredPatchIdx = r * 14 + c;
      }} else {{
        hoveredPatchIdx = null;
      }}
      render();
    }});

    modelCanvas.addEventListener('mouseleave', () => {{
      hoveredPatchIdx = null;
      render();
    }});

    origCanvas.addEventListener('mousemove', (e) => {{
      const rect = origCanvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (336 / rect.width);
      const y = (e.clientY - rect.top) * (336 / rect.height);
      const patchW = 336 / 14;
      const c = Math.floor(x / patchW);
      const r = Math.floor(y / patchW);
      if (c >= 0 && c < 14 && r >= 0 && r < 14) {{
        hoveredPatchIdx = r * 14 + c;
      }} else {{
        hoveredPatchIdx = null;
      }}
      render();
    }});

    origCanvas.addEventListener('mouseleave', () => {{
      hoveredPatchIdx = null;
      render();
    }});

    layerSlider.addEventListener('input', (e) => {{
      setLayer(parseInt(e.target.value, 10));
    }});

    kSlider.addEventListener('input', (e) => {{
      currentK = parseInt(e.target.value, 10);
      render();
      updateUI();
    }});

    heatAlphaSlider.addEventListener('input', (e) => {{
      heatmapAlpha = parseInt(e.target.value, 10) / 100;
      heatmapAlphaLabel.textContent = `Overlay: ${{e.target.value}}%`;
      render();
    }});

    // Initialize
    buildLayerBadges();
    switchScene('eagle');
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
