"""
ECHO — Inference Engine
Loaded once at API startup, handles all model predictions.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# All paths relative to project root (IADSR-main/)
PROJECT_ROOT   = Path(__file__).parent.parent
ITEM_LIST_PATH = PROJECT_ROOT / "Beauty/gru_item_list.json"
TITLE_MAP_PATH = PROJECT_ROOT / "Beauty/title_map.json"
MODEL_PATH     = PROJECT_ROOT / "ml/gru4rec_best.pth"
SEM_LONG_PATH  = PROJECT_ROOT / "ml/semantic_long.pt"
SEM_SHORT_PATH = PROJECT_ROOT / "ml/semantic_short.pt"
RESULTS_PATH   = PROJECT_ROOT / "ml/evaluation_results.json"

EMBEDDING_DIM  = 64
HIDDEN_DIM     = 128
NUM_LAYERS     = 2
SEM_DIM        = 1024
NUM_ITEMS      = 12102
SEQ_LEN        = 31
TAU            = 1.0
THETA          = -0.9


class GRU4RecECHO(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, num_layers=2,
                 num_items=12102, sem_dim=1024):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim)
        self.gru       = nn.GRU(embedding_dim, hidden_dim, num_layers,
                                 batch_first=True)
        self.act       = nn.Tanh()
        self.fc        = nn.Linear(hidden_dim, num_items)
        self.l1        = nn.Linear(sem_dim, hidden_dim, bias=False)
        self.l2        = nn.Linear(sem_dim, hidden_dim, bias=False)
        self.attn      = nn.Linear(3, 3, bias=False)
        self.decoder   = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        out, _           = self.gru(x)
        output_tensor, _ = torch.max(out, dim=1)
        logit            = self.act(self.fc(out[:, -1, :]))
        return logit, out, output_tensor

    def attention_score(self, c1, c2, c3):
        raw     = torch.stack([c1, c2, c3], dim=-1)
        weights = torch.softmax(self.attn(raw), dim=-1)
        return (weights * raw).sum(dim=-1)


def gumbel_sigmoid(logits, tau=1.0, hard=True):
    eps = 1e-10
    U   = torch.rand_like(logits)
    g   = -torch.log(-torch.log(U + eps) + eps)
    y   = ((logits + g) / tau).sigmoid()
    if hard:
        y_hard = (y > 0.5).float()
        y      = (y_hard - y).detach() + y
    return y


def pad_sequence(seq, length=31):
    if len(seq) < length:
        return [0] * (length - len(seq)) + seq
    return seq[-length:]


class ECHOInferenceEngine:
    """
    Singleton class loaded once at FastAPI startup.
    All inference methods are synchronous (called from async endpoints via run_in_executor).
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return
        print("Initialising ECHO inference engine ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {self.device}")

        # ── Load item data ────────────────────────────────────────────────
        with open(ITEM_LIST_PATH) as f:
            self.item_list = json.load(f)   # {"user0": [item_ids...]}
        with open(TITLE_MAP_PATH) as f:
            raw_title_map = json.load(f)
        self.title_map = {str(k): str(v) for k, v in raw_title_map.items()
                          if str(v).strip().lower() not in ("nan", "none", "")}

        # ── Load semantic embeddings ──────────────────────────────────────
        self.semantic_long  = torch.load(SEM_LONG_PATH,  map_location="cpu",
                                          weights_only=False)
        self.semantic_short = torch.load(SEM_SHORT_PATH, map_location="cpu",
                                          weights_only=False)

        # ── Load trained model ────────────────────────────────────────────
        self.model = GRU4RecECHO(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS,
                                   NUM_ITEMS, SEM_DIM).to(self.device)
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # ── Load evaluation results ───────────────────────────────────────
        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                self.eval_results = json.load(f)
        else:
            self.eval_results = {}

        self._initialized = True
        print("  ✅ ECHO engine ready.")

    # ─────────────────────────────────────────────────────────────────────
    def get_user_ids(self):
        return list(self.item_list.keys())

    def get_item_title(self, item_id: int) -> str:
        return self.title_map.get(str(item_id), f"Item #{item_id}")

    def get_user_history(self, user_id: str):
        """Return full interaction list (item ids) for a user."""
        return self.item_list.get(user_id, [])

    # ─────────────────────────────────────────────────────────────────────
    def _run_denoising(self, user_id: str):
        """
        Run one forward pass + denoising mask for a single user.
        Returns:
          logit       : [num_items] score tensor
          out         : [T, H] GRU hidden states
          h_long      : [H] long-term collab interest
          long_int    : [H] long-term semantic interest (projected)
          short_int   : [T', H] short-term semantic interests
          mask_probs  : [T'] binary keep/remove mask
          gru_slices  : [T', H] GRU short-term per timestep
          c_scores    : dict with c1, c2, c3 per timestep
          qualified   : bool — did user pass the θ threshold?
          ic          : actual sequence length (non-padded)
        """
        item_ids    = self.item_list[user_id][:-1]   # exclude test label
        train_ids   = item_ids[:-1]                  # exclude train label too
        ic          = len(train_ids)
        padded      = pad_sequence(train_ids, SEQ_LEN)

        x           = torch.tensor([padded], dtype=torch.long, device=self.device)
        emb         = self.model.embedding(x)                  # [1, T, 64]

        with torch.no_grad():
            logit, out, h_long = self.model(emb)
            logit    = logit[0]                                 # [num_items]
            out      = out[0]                                   # [T, H]
            h_long   = h_long[0]                               # [H]

            long_sem  = self.semantic_long.get(user_id)
            short_sem = self.semantic_short.get(user_id, [])

            if long_sem is None or not short_sem:
                return None

            long_int  = self.model.l2(long_sem.unsqueeze(0).to(self.device))[0]
            short_int = self.model.l1(
                torch.stack([s.to(self.device) for s in short_sem])
            )                                                   # [M, H]

            cos_sim   = F.cosine_similarity(h_long.unsqueeze(0),
                                             long_int.unsqueeze(0)).item()
            qualified = cos_sim >= THETA

            # GRU slice for this user's actual sequence length
            gru_slice = out[-ic:, :]                           # [ic, H]
            min_len   = min(gru_slice.size(0), short_int.size(0), SEQ_LEN)
            gru_slice = gru_slice[-min_len:]
            sem_slice = short_int[-min_len:]
            T         = gru_slice.size(0)

            c1s = torch.zeros(T, device=self.device)
            c2s = torch.zeros(T, device=self.device)
            c3s = torch.zeros(T, device=self.device)
            for j in range(T):
                c1s[j] = F.cosine_similarity(long_int.unsqueeze(0),
                                              gru_slice[j].unsqueeze(0))
                c2s[j] = F.cosine_similarity(h_long.unsqueeze(0),
                                              sem_slice[j].unsqueeze(0))
                c3s[j] = F.cosine_similarity(gru_slice[j].unsqueeze(0),
                                              sem_slice[j].unsqueeze(0))

            scores     = self.model.attention_score(c1s, c2s, c3s)
            mask_probs = gumbel_sigmoid(scores, tau=TAU, hard=True)

        return {
            "logit"       : logit,
            "h_long"      : h_long,
            "long_int"    : long_int,
            "gru_slice"   : gru_slice,
            "mask_probs"  : mask_probs,
            "c1s"         : c1s.cpu().tolist(),
            "c2s"         : c2s.cpu().tolist(),
            "c3s"         : c3s.cpu().tolist(),
            "scores"      : scores.cpu().tolist(),
            "qualified"   : qualified,
            "cos_sim"     : cos_sim,
            "T"           : T,
            "ic"          : ic,
        }

    # ─────────────────────────────────────────────────────────────────────
    def get_user_profile(self, user_id: str):
        """
        Full user profile with noise mask on interaction history.
        Returns list of items with keep/noise label and scores.
        """
        if user_id not in self.item_list:
            return None

        result = self._run_denoising(user_id)
        if result is None:
            return None

        item_ids    = self.item_list[user_id][:-1]   # training items
        train_ids   = item_ids[:-1]
        mask_probs  = result["mask_probs"].cpu().tolist()
        T           = result["T"]
        c1s         = result["c1s"]
        c2s         = result["c2s"]
        c3s         = result["c3s"]
        scores      = result["scores"]

        # Map mask back to original sequence positions
        # The mask covers the last T items of train_ids
        offset      = len(train_ids) - T
        history     = []
        for idx, iid in enumerate(train_ids):
            mask_idx = idx - offset
            if mask_idx >= 0 and mask_idx < T:
                kept   = bool(mask_probs[mask_idx] > 0.5)
                c1, c2, c3, sc = (c1s[mask_idx], c2s[mask_idx],
                                   c3s[mask_idx], scores[mask_idx])
            else:
                kept   = True   # padded / untouched positions
                c1 = c2 = c3 = sc = None

            history.append({
                "position" : idx + 1,
                "item_id"  : iid,
                "title"    : self.get_item_title(iid),
                "kept"     : kept,
                "c1"       : round(c1, 4)  if c1 is not None else None,
                "c2"       : round(c2, 4)  if c2 is not None else None,
                "c3"       : round(c3, 4)  if c3 is not None else None,
                "score"    : round(sc, 4)  if sc is not None else None,
                "explanation": self._explain(c1, c2, c3, kept)
                               if c1 is not None else "Not analysed",
            })

        test_item_id = item_ids[-1] if item_ids else None
        return {
            "user_id"          : user_id,
            "history"          : history,
            "noise_count"      : sum(1 for h in history if not h["kept"]),
            "keep_count"       : sum(1 for h in history if h["kept"]),
            "qualified"        : result["qualified"],
            "cross_modal_sim"  : round(result["cos_sim"], 4),
            "test_item_id"     : test_item_id,
            "test_item_title"  : self.get_item_title(test_item_id) if test_item_id else None,
        }

    # ─────────────────────────────────────────────────────────────────────
    def get_recommendations(self, user_id: str, top_k: int = 10):
        """Return top-K recommended items with and without denoising."""
        if user_id not in self.item_list:
            return None

        result = self._run_denoising(user_id)
        if result is None:
            return None

        logit     = result["logit"]
        item_ids  = self.item_list[user_id][:-1]
        seen      = set(item_ids)

        scores_np = logit.cpu().numpy()
        top_all   = np.argsort(scores_np)[::-1]
        recs      = []
        for iid in top_all:
            if int(iid) in seen:
                continue
            recs.append({
                "item_id": int(iid),
                "title"  : self.get_item_title(int(iid)),
                "score"  : round(float(scores_np[iid]), 4),
            })
            if len(recs) >= top_k:
                break
        return recs

    # ─────────────────────────────────────────────────────────────────────
    def get_metrics(self):
        return self.eval_results

    # ─────────────────────────────────────────────────────────────────────
    def _explain(self, c1, c2, c3, kept: bool) -> str:
        if kept:
            if c3 is not None and c3 > 0.5:
                return "Strong cross-modal consistency — aligns with your authentic interests."
            return "Consistent with your long-term interest profile."
        else:
            if c1 is not None and c1 < 0:
                return "Flagged: conflicts with your semantic long-term interests."
            if c2 is not None and c2 < 0:
                return "Flagged: behavioral pattern diverges from semantic profile."
            if c3 is not None and c3 < 0:
                return "Flagged: semantic and behavioral signals strongly disagree."
            return "Flagged: low cross-modal consistency — likely a noisy interaction."


# Singleton instance
engine = ECHOInferenceEngine()
