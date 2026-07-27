"""
ECHO — Semantic Embedding Generator
Uses BAAI/bge-large-en-v1.5 (768-dim, CPU-friendly, 335M params)
instead of Llama-3.1-8B (4096-dim, needs 16GB VRAM).

Novel Contribution #2: Lightweight real-time semantic encoder.

Generates:
  - semantic_long.pt  : dict{ "userX" -> tensor[768] }
  - semantic_short.pt : dict{ "userX" -> list[tensor[768]] }

These match the format expected by train_gru4rec.py.
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent.parent
DATA_DIR        = PROJECT_ROOT / "Beauty"
ITEM_LIST_PATH  = DATA_DIR / "gru_item_list.json"
DEMO_LIST_PATH  = DATA_DIR / "demo_gru_item_list.json"
TITLE_MAP_PATH  = DATA_DIR / "title_map.json"
ID_MAP_PATH     = DATA_DIR / "id_map.json"
OUTPUT_LONG     = PROJECT_ROOT / "ml/semantic_long.pt"
OUTPUT_SHORT    = PROJECT_ROOT / "ml/semantic_short.pt"
MODEL_NAME      = "BAAI/bge-large-en-v1.5"   # 768-dim, ~1.2GB download
BATCH_SIZE      = 256
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
FAST_DEMO_USERS = None # Set to None for full dataset (13+ hours on RTX 4050)
# ────────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading interaction sequences...")
    with open(ITEM_LIST_PATH, "r") as f:
        item_list = json.load(f)          # {"user0": [item_ids...], ...}

    print("Loading ID map...")
    with open(ID_MAP_PATH, "r") as f:
        id_map = json.load(f)             # {ASIN: item_id}
    # Reverse to item_id -> ASIN
    inv_id_map = {str(v): k for k, v in id_map.items()}

    print("Loading title map...")
    with open(TITLE_MAP_PATH, "r") as f:
        title_map = json.load(f)          # {ASIN: title_string}

    # Create direct map from item_id -> title
    item_id_to_title = {}
    for item_id, asin in inv_id_map.items():
        title = title_map.get(asin, "")
        if title:
            item_id_to_title[item_id] = title

    if FAST_DEMO_USERS:
        print(f"FAST DEMO MODE: Subsetting to first {FAST_DEMO_USERS} users...")
        item_list = dict(list(item_list.items())[:FAST_DEMO_USERS])
        with open(DEMO_LIST_PATH, "w") as f:
            json.put(item_list, f) if hasattr(json, 'put') else json.dump(item_list, f)
        print(f"Saved subset to {DEMO_LIST_PATH}")

    return item_list, item_id_to_title


def item_ids_to_titles(item_ids: list, title_map: dict) -> list:
    """Convert a list of integer item IDs to their title strings."""
    titles = []
    for iid in item_ids:
        t = title_map.get(str(iid), "")
        if t and str(t).strip().lower() not in ("nan", "none", ""):
            titles.append(str(t).strip())
    return titles


def build_user_texts(item_list: dict, title_map: dict):
    """
    For each user build:
      long_text  : titles of all items except the last (leave-one-out test item)
      short_texts: progressive prefix sequences (for short-term interest modelling)
                   short_texts[t] = titles[0..t]  for t in 0..n-3
                   (we need at least 2 items to have 1 short prefix)
    """
    user_long_texts  = {}
    user_short_texts = {}
    all_unique_texts = set()
    skipped = 0

    for user_key, item_ids in tqdm(item_list.items(), desc="Building user texts"):
        # item_ids[-1] is the test label — exclude it
        train_ids = item_ids[:-1]
        titles    = item_ids_to_titles(train_ids, title_map)

        # Need at least 2 items: 1 for a short-term prefix, 1 for long-term
        if len(titles) < 2:
            skipped += 1
            continue

        # Long-term: full training sequence as a comma-joined string
        long_text = ", ".join(titles)
        user_long_texts[user_key]  = long_text
        all_unique_texts.add(long_text)

        # Short-term: prefixes of length 1..n-2
        # (so last short prefix doesn't include the final training item)
        short_texts = []
        for end in range(1, len(titles)):          # end = 1, 2, ..., len-1
            prefix = ", ".join(titles[:end])
            short_texts.append(prefix)
            all_unique_texts.add(prefix)
        user_short_texts[user_key] = short_texts

    print(f"Users kept : {len(user_long_texts)} | Skipped : {skipped}")
    print(f"Unique texts to encode: {len(all_unique_texts)}")
    return user_long_texts, user_short_texts, list(all_unique_texts)


def encode_texts(model: SentenceTransformer, texts: list) -> dict:
    """Encode all unique texts in batches, return {text: tensor[768]}."""
    print(f"\nEncoding {len(texts)} unique texts on {DEVICE} ...")
    all_embeddings = model.encode(
        texts,
        batch_size        = BATCH_SIZE,
        show_progress_bar = True,
        convert_to_tensor = True,
        device            = DEVICE,
        normalize_embeddings = True,   # L2-normalise (better for cosine sim)
    )                                  # shape: [N, 768]
    text_to_embed = {text: all_embeddings[i].cpu() for i, text in enumerate(texts)}
    return text_to_embed


def build_output_dicts(user_long_texts, user_short_texts, text_to_embed):
    """Assemble the final dicts in the format expected by the training script."""
    semantic_long  = {}
    semantic_short = {}

    for user_key in tqdm(user_long_texts, desc="Assembling user embeddings"):
        lt = user_long_texts[user_key]
        if lt not in text_to_embed:
            continue
        semantic_long[user_key] = text_to_embed[lt]           # tensor[768]

        sts = user_short_texts.get(user_key, [])
        short_embeds = [text_to_embed[t] for t in sts if t in text_to_embed]
        if not short_embeds:
            continue
        semantic_short[user_key] = short_embeds               # list[tensor[768]]

    return semantic_long, semantic_short


def main():
    print("=" * 60)
    print("  ECHO — Semantic Embedding Generator")
    print(f"  Encoder : {MODEL_NAME}")
    print(f"  Device  : {DEVICE}")
    print("=" * 60)

    item_list, title_map = load_data()

    print("\nLoading sentence-transformer model (auto-downloads if first run) ...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    user_long_texts, user_short_texts, all_texts = build_user_texts(item_list, title_map)
    text_to_embed = encode_texts(model, all_texts)

    semantic_long, semantic_short = build_output_dicts(
        user_long_texts, user_short_texts, text_to_embed
    )

    print(f"\nSaving {OUTPUT_LONG} ...")
    torch.save(semantic_long, OUTPUT_LONG)

    print(f"Saving {OUTPUT_SHORT} ...")
    torch.save(semantic_short, OUTPUT_SHORT)

    # ── Sanity check ──
    sample_user = list(semantic_long.keys())[0]
    print("\n── Sanity check ──────────────────────────────────")
    print(f"Sample user        : {sample_user}")
    print(f"Long embed shape   : {semantic_long[sample_user].shape}")
    print(f"Short embeds count : {len(semantic_short[sample_user])}")
    print(f"Short embed[0] dim : {semantic_short[sample_user][0].shape}")
    print("──────────────────────────────────────────────────")
    print("\n✅ Done! semantic_long.pt and semantic_short.pt saved.")


if __name__ == "__main__":
    main()
