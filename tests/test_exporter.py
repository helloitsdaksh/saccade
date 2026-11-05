import os
import pytest
import torch
from PIL import Image
from saccade.models import SaccadeViT
from saccade.exporter import export_interactive_html, image_to_base64


def test_image_to_base64():
    img = Image.new("RGB", (32, 32), color="red")
    b64 = image_to_base64(img)
    assert b64.startswith("data:image/png;base64,")


def test_export_interactive_html(tmp_path):
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
    img = Image.new("RGB", (64, 64), color="blue")
    out_html = tmp_path / "test_saccade.html"

    result_path = export_interactive_html(
        model=model,
        image=img,
        output_path=str(out_html),
        class_names=["Class A", "Class B", "Class C"],
        default_top_k=4,
        title="Test Viewer",
    )

    assert os.path.exists(result_path)
    content = out_html.read_text(encoding="utf-8")
    assert "<canvas id=\"saccadeCanvas\"" in content
    assert "David Ha" in content
    assert "data:image/png;base64," in content
    assert "Test Viewer" in content

