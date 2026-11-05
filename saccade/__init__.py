"""
saccade
~~~~~~~

Intrinsic hard attention bottlenecks for Vision Transformers (ViT) and Large Language Models (LLM).
Physically prunes uninformative visual patches and prompt tokens via column-sum attention voting.
"""

from saccade.bottleneck import (
    SaccadeBottleneck,
    SaccadeMetadata,
    compute_patch_importance,
    patch_indices_to_bboxes,
)
from saccade.models import SaccadeViT, CleanViT
from saccade.visualization import (
    render_saccade_overlay,
    plot_saccade_breakdown,
)
from saccade.exporter import export_interactive_html
from saccade.text import PromptSaccade, SaccadeTextMetadata

__version__ = "0.1.0"
__author__ = "Daksh Patel"

__all__ = [
    "SaccadeViT",
    "CleanViT",
    "SaccadeBottleneck",
    "SaccadeMetadata",
    "compute_patch_importance",
    "patch_indices_to_bboxes",
    "render_saccade_overlay",
    "plot_saccade_breakdown",
    "export_interactive_html",
    "PromptSaccade",
    "SaccadeTextMetadata",
]

