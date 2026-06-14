# models/llama_loader.py — load LLaMA locally via HuggingFace Transformers

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from config import MODEL_NAME


def load_model():

    print(f"[llama_loader] Loading tokenizer from: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[llama_loader] Loading 4-bit quantized model from: {MODEL_NAME}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    device = next(model.parameters()).device

    print(f"[llama_loader] Model loaded on: {device}")

    return model, tokenizer