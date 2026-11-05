"""
saccade - Quickstart Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrates basic usage of SaccadeViT with intrinsic Top-K attention bottleneck,
metadata extraction, static visualization, and interactive web artifact export.
"""

import sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from saccade import SaccadeViT, export_interactive_html, plot_saccade_breakdown


def main():
    print("=" * 60)
    print("saccade // Intrinsically Interpretable Vision Transformers")
    print("=" * 60)

    # 1. Instantiate SaccadeViT with an attention bottleneck at layer 6
    # Budget: K=12 patches out of 196 patches (only 6.1% visual budget!)
    print("\n[1] Instantiating SaccadeViT (vit_tiny_patch16_224)...")
    model = SaccadeViT.from_pretrained(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=1000,
        top_k_patches=12,
        bottleneck_layer=6,
    )
    model.eval()

    # 2. Create or load a sample image (geometric test scene)
    print("[2] Generating test image (224x224)...")
    img = Image.new("RGB", (224, 224), color=(20, 24, 34))
    draw = ImageDraw.Draw(img)
    # Target focal object (e.g. golden circle)
    draw.ellipse([70, 70, 154, 154], fill=(245, 158, 11), outline=(255, 255, 255), width=2)
    # Distractor object (blue rectangle)
    draw.rectangle([160, 20, 205, 65], fill=(59, 130, 246))

    # Convert to normalized tensor [1, 3, 224, 224]
    import numpy as np
    img_np = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_tensor = torch.from_numpy((img_np - mean) / std).permute(2, 0, 1).unsqueeze(0)

    # 3. Forward pass with intrinsic attention bottleneck
    print("[3] Running forward pass with Saccade attention bottleneck...")
    with torch.no_grad():
        logits, meta = model(img_tensor, return_saccades=True)

    print(f"\n--- Saccade Bottleneck Results ---")
    print(f"Top-K Patch Budget: {meta.top_k} / {meta.grid_size[0] * meta.grid_size[1]}")
    print(f"Selected Patch Indices: {meta.patch_indices[0].cpu().numpy().tolist()}")
    print(f"Bottleneck Layer Index: {meta.layer_idx}")
    print(f"Sample Bounding Box: {meta.bounding_boxes[0][0]}")

    # 4. Generate static visual breakdown figure
    assets_dir = Path(__file__).parent.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig_path = assets_dir / "quickstart_breakdown.png"
    print(f"\n[4] Saving static visual breakdown to: {fig_path}")
    plot_saccade_breakdown(img, meta, logits=logits, save_path=str(fig_path))

    # 5. Export standalone interactive web artifact
    html_path = assets_dir / "quickstart_interactive.html"
    print(f"[5] Exporting interactive HTML artifact to: {html_path}")
    export_interactive_html(
        model=model,
        image=img,
        output_path=html_path,
        default_top_k=12,
        title="saccade // Quickstart Interactive Demo",
        description="Touch, slide, and ablate the model's internal perception in real time. Zero backend required."
    )

    print("\n✓ Quickstart completed successfully!")
    print(f"Open '{html_path}' in any web browser to interact with the model live.")


if __name__ == "__main__":
    main()

