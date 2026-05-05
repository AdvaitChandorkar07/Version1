# config.py — central config for all paths and constants

import os

# ── Model ──────────────────────────────────────────────────────────────────────
# Point this at your local LLaMA weights folder (GGUF or HuggingFace format).
# HuggingFace example : "meta-llama/Llama-3.2-1B-Instruct"  (needs HF token)
# Local folder example: "/models/llama-3.2-1b-instruct"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Layer index where the steering vector is injected (mid-to-late layer).
# For a 16-layer 1B model, layer 12 is a good starting point.
STEERING_LAYER = 12

# How strongly the steering vector is applied (scale factor).
# 0 = no effect, 1 = full strength. Start at 0.5 and tune.
STEERING_ALPHA = 0.5

# Top-k preference contexts retrieved at inference time
TOP_K = 3

# Maximum new tokens the LLM generates per turn
MAX_NEW_TOKENS = 300

# ── Sentence Transformer ───────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"     # 384-dim, fast, good quality
EMBED_DIM   = 384

# ── Data paths ─────────────────────────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
DATA_DIR            = os.path.join(BASE_DIR, "data")

RAW_PREFS_PATH      = os.path.join(DATA_DIR, "raw_preferences.json")
SEMANTIC_VECS_PATH  = os.path.join(DATA_DIR, "semantic_vectors.npy")
STEERING_VECS_PATH  = os.path.join(DATA_DIR, "steering_vectors.npy")
FAISS_INDEX_PATH    = os.path.join(DATA_DIR, "faiss.index")

os.makedirs(DATA_DIR, exist_ok=True)