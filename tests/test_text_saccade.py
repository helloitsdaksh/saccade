"""
tests/test_text_saccade.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for PromptSaccade text token pruning.
"""

import pytest
from saccade.text import PromptSaccade, SaccadeTextMetadata


@pytest.fixture(scope="module")
def compressor():
    return PromptSaccade(model_name="distilgpt2", device="cpu")


def test_prompt_saccade_compression_ratio(compressor):
    text = (
        "In accordance with standard legal due diligence, please review Section 14.2 "
        "of the Master Services Agreement to confirm whether there are any uncapped liability exceptions."
    )
    meta = compressor.compress(text, compression_ratio=0.40)

    assert isinstance(meta, SaccadeTextMetadata)
    assert meta.original_tokens > 0
    assert meta.compressed_tokens < meta.original_tokens
    assert meta.tokens_saved > 0
    assert 0.0 < meta.token_savings_pct < 100.0
    assert len(meta.compressed_text) > 0


def test_prompt_saccade_target_tokens(compressor):
    text = (
        "Please carefully analyze the surgical pathology report and inform the attending "
        "oncologist whether there is metastatic melanoma in the sentinel lymph node."
    )
    target = 10
    meta = compressor.compress(text, target_tokens=target)

    assert meta.compressed_tokens == target
    assert meta.original_tokens > target


def test_prompt_saccade_query_guidance(compressor):
    text = "The quick brown fox jumps over the lazy dog. The cat slept peacefully under the oak tree."
    meta_fox = compressor.compress(text, target_tokens=6, query="Where is the fox?")
    meta_cat = compressor.compress(text, target_tokens=6, query="Where is the cat?")

    assert "fox" in meta_fox.compressed_text.lower()
    assert "cat" in meta_cat.compressed_text.lower()

