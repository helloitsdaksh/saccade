"""
saccade/text/token_saccade.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hard attention bottlenecking for Text Prompts and LLMs.
Prunes redundant syntactic filler and low-attention tokens ("breaking grammar")
to slash LLM API token consumption, KV-cache memory, and latency while
preserving semantic comprehension and downstream answer accuracy.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class SaccadeTextMetadata:
    """Metadata returned by PromptSaccade compression."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    compression_ratio: float
    token_savings_pct: float
    selected_indices: List[int]
    token_importance_scores: List[float]
    token_strings: List[str]


class PromptSaccade:
    """
    Intrinsically compresses text prompts using self-attention voting.
    Discards low-information filler tokens to minimize downstream LLM cost.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: Optional[str] = None,
        probe_layer: int = -1,
    ):
        """
        Args:
            model_name: HuggingFace model used for attention extraction.
            device: 'cuda', 'mps', or 'cpu'. Defaults to autodetect.
            probe_layer: Which transformer layer to extract attention from (-1 for final).
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True
        ).to(self.device).eval()

        self.probe_layer = probe_layer

    @torch.no_grad()
    def compute_token_importance(
        self,
        text: str,
        query: Optional[str] = None
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Computes the attention & information-theoretic importance score for each token.
        Combines David Ha column voting with token surprise (-log P(t_i | t_<i))
        and neutralizes the autoregressive attention sink at position 0.
        """
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = encoded["input_ids"]
        seq_len = input_ids.shape[1]

        # Forward pass with attention extraction and logits
        outputs = self.model(**encoded)
        attentions = outputs.attentions
        logits = outputs.logits # [1, N, V]

        # 1. Compute Token Surprise (Self-Information) in bits: -log P(t_i | t_<i)
        # Frequent stop words have high probability -> low surprise
        # Content keywords have low probability -> high surprise
        log_probs = F.log_softmax(logits, dim=-1)
        shift_log_probs = log_probs[:, :-1, :]
        shift_ids = input_ids[:, 1:]
        token_surprises = -shift_log_probs.gather(2, shift_ids.unsqueeze(-1)).squeeze()
        if token_surprises.dim() == 0:
            token_surprises = token_surprises.unsqueeze(0)
        # Prefix first token with average surprise
        surprises = torch.cat([token_surprises[:1], token_surprises])

        # 2. Self-Attention Column Voting: p_j = sum_i A_{i, j}
        layer_attn = attentions[self.probe_layer] # [1, heads, N, N]
        mean_attn = layer_attn[0].mean(dim=0) # [N, N]
        column_votes = mean_attn.sum(dim=0) # [N]

        # 3. Neutralize Autoregressive Position-0 Attention Sink
        if seq_len > 1:
            column_votes[0] = column_votes[1:].median()

        # 4. Synthesize Saccade Importance Score: Attention Votes * Information Entropy
        norm_votes = column_votes / (column_votes.max() + 1e-8)
        norm_surprise = surprises / (surprises.max() + 1e-8)
        scores = 0.55 * norm_votes + 0.45 * norm_surprise

        token_strs = [self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        # If a specific query is provided, amplify tokens that the query attends to
        if query is not None:
            q_enc = self.tokenizer(f"{text} Query: {query}", return_tensors="pt").to(self.device)
            q_out = self.model(**q_enc)
            q_attn = q_out.attentions[self.probe_layer][0].mean(dim=0)
            q_len = q_enc["input_ids"].shape[1] - seq_len
            if q_len > 0:
                query_to_text = q_attn[-q_len:, :seq_len].sum(dim=0)
                if seq_len > 1:
                    query_to_text[0] = 0.0 # neutralize sink
                norm_q = query_to_text / (query_to_text.max() + 1e-8)
                scores = 0.4 * scores + 0.6 * norm_q

            # Entity match boost: queried entities receive high saliency
            query_words = set(w.lower().strip("?,.!") for w in query.split())
            for i, t_str in enumerate(token_strs):
                clean_w = t_str.lower().strip(" .,!?:;'\"")
                if clean_w and clean_w in query_words:
                    scores[i] += 1.5

        return scores, token_strs, input_ids[0]

    def compress(
        self,
        text: str,
        target_tokens: Optional[int] = None,
        compression_ratio: Optional[float] = None,
        query: Optional[str] = None,
        preserve_order: bool = True,
    ) -> SaccadeTextMetadata:
        """
        Compresses a prompt down to a hard visual/token budget.
        
        Args:
            text: Input prompt text to compress.
            target_tokens: Exact number of tokens K to keep.
            compression_ratio: Fraction of tokens to keep (e.g. 0.35 keeps 35%).
            query: Optional query to guide token importance.
            preserve_order: Whether to preserve original chronological token sequence.
            
        Returns:
            SaccadeTextMetadata with compressed string and token analytics.
        """
        scores, token_strs, input_ids = self.compute_token_importance(text, query=query)
        total_tokens = len(token_strs)

        if target_tokens is not None:
            k = max(1, min(target_tokens, total_tokens))
        elif compression_ratio is not None:
            k = max(1, min(int(total_tokens * compression_ratio), total_tokens))
        else:
            k = max(1, int(total_tokens * 0.40)) # Default 60% reduction

        # Top-K selection
        topk_scores, topk_indices = torch.topk(scores, k=k)
        selected_idx_list = topk_indices.cpu().numpy().tolist()

        if preserve_order:
            selected_idx_list = sorted(selected_idx_list)

        # Reconstruct text from selected tokens
        selected_tids = [input_ids[idx].item() for idx in selected_idx_list]
        compressed_text = self.tokenizer.decode(selected_tids).strip()

        tokens_saved = total_tokens - k
        savings_pct = (tokens_saved / total_tokens) * 100.0

        return SaccadeTextMetadata(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=total_tokens,
            compressed_tokens=k,
            tokens_saved=tokens_saved,
            compression_ratio=k / total_tokens,
            token_savings_pct=savings_pct,
            selected_indices=selected_idx_list,
            token_importance_scores=scores.cpu().numpy().tolist(),
            token_strings=token_strs,
        )
