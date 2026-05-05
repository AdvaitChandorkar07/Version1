# models/llama_loader.py — load LLaMA locally via HuggingFace Transformers

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME


def load_model():
    """
    Load LLaMA model and tokenizer from a local path or HuggingFace hub.

    Returns
    -------
    model     : AutoModelForCausalLM  (fp16, on GPU if available)
    tokenizer : AutoTokenizer
    """
    print(f"[llama_loader] Loading tokenizer from: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    # Add a pad token if the model doesn't have one (LLaMA often lacks it)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[llama_loader] Loading model from: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",       # spreads across GPU(s); falls back to CPU
        low_cpu_mem_usage=True,
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"[llama_loader] Model loaded on: {device}")
    return model, tokenizer