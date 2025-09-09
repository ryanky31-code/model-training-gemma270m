import os
import pytest


def test_hf_token_and_model_import():
    # This test attempts to import transformers and instantiate the tokenizer/model
    # only when HF_TOKEN is present. It is safe in CI because it will be skipped when no token.
    token = os.environ.get('HF_TOKEN')
    if not token:
        pytest.skip('No HF_TOKEN set in environment; skipping HF access test')
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Use a small, permissive model id for quick validation (do not attempt to download large weights in CI)
        model_id = os.environ.get('TEST_HF_MODEL', 'google/gemma-3-270m')
        # Only load tokenizer to avoid heavy downloads in CI
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        assert tok is not None
    except Exception as e:
        pytest.fail(f'HF model/tokenizer import test failed: {e}')
