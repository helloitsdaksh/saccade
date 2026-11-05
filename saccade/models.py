"""
saccade.models
~~~~~~~~~~~~~~

SaccadeViT: Intrinsically Interpretable Vision Transformers with hard Top-K
attention bottlenecks.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from saccade.bottleneck import SaccadeBottleneck, SaccadeMetadata, compute_patch_importance, patch_indices_to_bboxes

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class InterceptedAttention(nn.Module):
    """Wraps a timm Attention module to intercept and store attention affinity matrices."""

    def __init__(self, original_attn: nn.Module):
        super().__init__()
        self.original_attn = original_attn
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        orig = self.original_attn
        orig.fused_attn = False
        B, N, C = x.shape

        gate = orig.gate(x).sigmoid() if hasattr(orig, "gate") and orig.gate is not None else None
        qkv = orig.qkv(x).reshape(B, N, 3, orig.num_heads, orig.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        if hasattr(orig, "q_norm") and orig.q_norm is not None:
            q = orig.q_norm(q)
        if hasattr(orig, "k_norm") and orig.k_norm is not None:
            k = orig.k_norm(k)

        scale = getattr(orig, "scale", orig.head_dim ** -0.5)
        q = q * scale
        attn = q @ k.transpose(-2, -1)
        
        if attn_mask is not None:
            attn = attn + attn_mask
            
        attn = attn.softmax(dim=-1)
        self.last_attn_weights = attn.detach()

        attn_dropped = orig.attn_drop(attn) if hasattr(orig, "attn_drop") else attn
        x_out = attn_dropped @ v

        x_out = x_out.transpose(1, 2).reshape(B, N, getattr(orig, "attn_dim", C))
        if hasattr(orig, "norm") and orig.norm is not None:
            x_out = orig.norm(x_out)
        if gate is not None:
            x_out = x_out * gate
        x_out = orig.proj(x_out)
        if hasattr(orig, "proj_drop") and orig.proj_drop is not None:
            x_out = orig.proj_drop(x_out)

        return x_out


class CleanViT(nn.Module):
    """A self-contained native Vision Transformer with zero external weight dependencies.
    
    Ideal for rapid prototyping, unit testing, and training on custom medical or synthetic data.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            NativeBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class NativeBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NativeAttention(dim, num_heads=num_heads, drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class NativeAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.last_attn_weights = attn.detach()

        attn_drop = self.drop(attn)
        out = (attn_drop @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class SaccadeViT(nn.Module):
    """Vision Transformer equipped with David Ha's Top-K Intrinsic Attention Bottleneck.
    
    Tokens are routed through initial transformer blocks up to `bottleneck_layer`.
    At that point, spatial patch importance is computed by summing the column affinity
    votes in the probing self-attention matrix:
        p_j = \\sum_{i=1}^N A_{i, j}
    
    Only the top-K patches (plus CLS) are passed through subsequent layers, mathematically
    bounding downstream predictions exclusively to those visual coordinates.
    """

    def __init__(
        self,
        backbone: nn.Module,
        top_k_patches: int = 16,
        bottleneck_layer: int = 6,
        grid_size: Tuple[int, int] = (14, 14),
        has_cls_token: bool = True,
        cls_weight: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.top_k = top_k_patches
        self.bottleneck_layer = bottleneck_layer
        self.grid_size = grid_size
        self.has_cls_token = has_cls_token
        self.num_patches = grid_size[0] * grid_size[1]
        
        self.bottleneck = SaccadeBottleneck(
            top_k=top_k_patches,
            grid_size=grid_size,
            has_cls_token=has_cls_token,
            layer_idx=bottleneck_layer,
            cls_weight=cls_weight,
        )

        # Hook or wrap attention at bottleneck_layer
        self._intercept_bottleneck_attention()

    def _intercept_bottleneck_attention(self):
        """Intercepts attention weights at the designated bottleneck layer."""
        if hasattr(self.backbone, "blocks"):
            target_block = self.backbone.blocks[self.bottleneck_layer]
            if not isinstance(target_block.attn, (InterceptedAttention, NativeAttention)):
                target_block.attn = InterceptedAttention(target_block.attn)
            self._target_attn_module = target_block.attn
        else:
            raise AttributeError("Backbone does not have 'blocks' attribute.")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1000,
        top_k_patches: int = 16,
        bottleneck_layer: int = 6,
        cls_weight: float = 0.0,
        **kwargs,
    ) -> "SaccadeViT":
        """Instantiates a SaccadeViT from a timm model or native implementation."""
        if model_name.lower() == "native":
            backbone = CleanViT(num_classes=num_classes, **kwargs)
            grid_size = backbone.grid_size
        else:
            if not HAS_TIMM:
                raise ImportError("Package 'timm' is required for pretrained ViT models. Install via `pip install timm`.")
            backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                **kwargs,
            )
            # Infer grid_size
            if hasattr(backbone, "patch_embed") and hasattr(backbone.patch_embed, "grid_size"):
                grid_size = backbone.patch_embed.grid_size
            else:
                grid_size = (14, 14)

        return cls(
            backbone=backbone,
            top_k_patches=top_k_patches,
            bottleneck_layer=bottleneck_layer,
            grid_size=grid_size,
            has_cls_token=True,
            cls_weight=cls_weight,
        )

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input image tensor [B, C, H, W] to initial tokens [B, S, D]."""
        if hasattr(self.backbone, "patch_embed") and callable(self.backbone.patch_embed):
            # timm or CleanViT patch embed
            if isinstance(self.backbone.patch_embed, nn.Conv2d):
                # Native CleanViT style conv2d
                patches = self.backbone.patch_embed(x).flatten(2).transpose(1, 2)
            else:
                patches = self.backbone.patch_embed(x)
        else:
            raise AttributeError("Backbone missing patch_embed module.")

        B = x.shape[0]
        if self.has_cls_token:
            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_token, patches], dim=1)
        else:
            tokens = patches

        if hasattr(self.backbone, "pos_embed") and self.backbone.pos_embed is not None:
            tokens = tokens + self.backbone.pos_embed

        if hasattr(self.backbone, "pos_drop"):
            tokens = self.backbone.pos_drop(tokens)

        return tokens

    def forward_features_to_bottleneck(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs input through layers up to bottleneck_layer and extracts attention."""
        tokens = self._prepare_tokens(x)
        
        # Execute transformer blocks up to bottleneck_layer
        for i in range(self.bottleneck_layer + 1):
            tokens = self.backbone.blocks[i](tokens)

        # Retrieve attention weights intercepted at bottleneck_layer
        attn_weights = self._target_attn_module.last_attn_weights
        if attn_weights is None:
            raise RuntimeError(f"Failed to capture attention weights at layer {self.bottleneck_layer}.")

        return tokens, attn_weights

    def forward_from_bottleneck(self, pruned_tokens: torch.Tensor) -> torch.Tensor:
        """Executes remaining transformer blocks and classification head on pruned tokens."""
        x = pruned_tokens
        total_blocks = len(self.backbone.blocks)
        for i in range(self.bottleneck_layer + 1, total_blocks):
            x = self.backbone.blocks[i](x)

        if hasattr(self.backbone, "norm"):
            x = self.backbone.norm(x)

        # Forward head
        if hasattr(self.backbone, "forward_head"):
            logits = self.backbone.forward_head(x)
        elif hasattr(self.backbone, "head"):
            # If CLS token exists, head takes x[:, 0]
            cls_feat = x[:, 0] if self.has_cls_token else x.mean(dim=1)
            logits = self.backbone.head(cls_feat)
        else:
            logits = x

        return logits

    def forward(
        self,
        x: torch.Tensor,
        return_saccades: bool = False,
        override_top_k: Optional[int] = None,
        ablate_patch_indices: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, SaccadeMetadata]]:
        """Forward pass applying the causal Top-K attention bottleneck.
        
        Args:
            x: Input image tensor [B, C, H, W].
            return_saccades: If True, returns (logits, SaccadeMetadata).
            override_top_k: Dynamic patch budget K override.
            ablate_patch_indices: Optional list of patch indices to forcefully remove
                                  from the attended set (for counterfactual analysis).
                                  
        Returns:
            logits: Output logits [B, num_classes].
            metadata: SaccadeMetadata (if return_saccades is True).
        """
        # 1. Forward to bottleneck layer
        tokens, attn_weights = self.forward_features_to_bottleneck(x)

        # 2. If ablating patches, mask their importance scores before topk
        K = override_top_k if override_top_k is not None else self.top_k
        
        importance = compute_patch_importance(
            attention_matrix=attn_weights,
            has_cls_token=self.has_cls_token,
            cls_weight=self.bottleneck.cls_weight,
        )

        if ablate_patch_indices is not None and len(ablate_patch_indices) > 0:
            mask_val = -1e9
            for p_idx in ablate_patch_indices:
                importance[:, p_idx] = mask_val

        # Select top-K
        topk_scores, topk_indices = torch.topk(importance, k=K, dim=-1, largest=True, sorted=True)
        
        # Route tokens
        B, S, D = tokens.shape
        if self.has_cls_token:
            cls_token = tokens[:, :1, :]
            spatial_tokens = tokens[:, 1:, :]
            gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            selected_spatial_tokens = torch.gather(spatial_tokens, dim=1, index=gather_indices)
            pruned_tokens = torch.cat([cls_token, selected_spatial_tokens], dim=1)
        else:
            gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            pruned_tokens = torch.gather(tokens, dim=1, index=gather_indices)

        # 3. Downstream prediction
        logits = self.forward_from_bottleneck(pruned_tokens)

        if return_saccades:
            bboxes = patch_indices_to_bboxes(topk_indices, self.grid_size)
            meta = SaccadeMetadata(
                patch_indices=topk_indices,
                patch_importance=importance,
                top_k=K,
                grid_size=self.grid_size,
                bounding_boxes=bboxes,
                layer_idx=self.bottleneck_layer,
                attention_matrix=attn_weights if attn_weights.ndim == 3 else attn_weights.mean(dim=1),
            )
            return logits, meta

        return logits

    @torch.no_grad()
    def compute_sparsity_trajectory(
        self,
        x: torch.Tensor,
        k_values: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Efficiently sweeps K from 1 to N, reusing early layer activations.
        
        Used to power the interactive Otoro-style sparsity slider.
        """
        self.eval()
        tokens, attn_weights = self.forward_features_to_bottleneck(x)
        importance = compute_patch_importance(
            attention_matrix=attn_weights,
            has_cls_token=self.has_cls_token,
            cls_weight=self.bottleneck.cls_weight,
        )

        N = self.num_patches
        if k_values is None:
            # Default logarithmic/geometric + linear progression
            k_values = sorted(list(set([1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, N])))
            k_values = [k for k in k_values if k <= N]

        trajectory = []
        B, S, D = tokens.shape
        cls_token = tokens[:, :1, :] if self.has_cls_token else None
        spatial_tokens = tokens[:, 1:, :] if self.has_cls_token else tokens

        for k in k_values:
            topk_scores, topk_indices = torch.topk(importance, k=k, dim=-1, largest=True, sorted=True)
            gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            selected_tokens = torch.gather(spatial_tokens, dim=1, index=gather_indices)
            
            if self.has_cls_token:
                pruned_tokens = torch.cat([cls_token, selected_tokens], dim=1)
            else:
                pruned_tokens = selected_tokens

            logits = self.forward_from_bottleneck(pruned_tokens)
            probs = F.softmax(logits, dim=-1)
            top1_prob, top1_class = torch.topk(probs, k=1, dim=-1)

            trajectory.append({
                "k": k,
                "sparsity_ratio": float(k / N),
                "top1_class": int(top1_class[0, 0].item()),
                "top1_prob": float(top1_prob[0, 0].item()),
                "logits": logits[0].cpu().numpy().tolist(),
                "patch_indices": topk_indices[0].cpu().numpy().tolist(),
            })

        return trajectory

    @torch.no_grad()
    def compute_counterfactual_ablations(
        self,
        x: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Computes marginal contribution of each active patch by ablating it.
        
        Returns:
            Dict containing base prediction, active patch indices, and delta logits
            for each patch when removed: \\Delta y = y(K) - y(K \\ {j}).
        """
        self.eval()
        K = top_k if top_k is not None else self.top_k
        base_logits, meta = self.forward(x, return_saccades=True, override_top_k=K)
        base_probs = F.softmax(base_logits, dim=-1)
        base_class = int(torch.argmax(base_probs, dim=-1)[0].item())
        base_class_logit = float(base_logits[0, base_class].item())

        active_indices = meta.patch_indices[0].cpu().numpy().tolist()
        ablations = {}

        for p_idx in active_indices:
            # Ablate this patch
            ablated_logits = self.forward(
                x,
                return_saccades=False,
                override_top_k=K,
                ablate_patch_indices=[p_idx]
            )
            ablated_probs = F.softmax(ablated_logits, dim=-1)
            ablated_class_logit = float(ablated_logits[0, base_class].item())
            ablated_class_prob = float(ablated_probs[0, base_class].item())
            
            # Causal delta: drop in target class logit when patch j is removed
            delta_logit = base_class_logit - ablated_class_logit
            
            ablations[int(p_idx)] = {
                "delta_logit": delta_logit,
                "ablated_prob": ablated_class_prob,
                "base_prob": float(base_probs[0, base_class].item()),
            }

        return {
            "base_class": base_class,
            "base_prob": float(base_probs[0, base_class].item()),
            "active_patches": active_indices,
            "ablations": ablations,
        }

