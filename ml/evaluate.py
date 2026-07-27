"""
ECHO — Evaluation Script
Computes HR@K and NDCG@K for:
  - Baseline GRU4Rec (no IADSR)
  - IADSR+ (our trained model)

Outputs results to evaluation_results.json for the API dashboard.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent.parent
ITEM_LIST_PATH  = PROJECT_ROOT / "Beauty/gru_item_list.json" # Use full dataset
MODEL_PATH      = PROJECT_ROOT / "ml/gru4rec_best.pth"
RESULTS_PATH    = PROJECT_ROOT / "ml/evaluation_results.json"

EMBEDDING_DIM   = 64
HIDDEN_DIM      = 128
NUM_LAYERS      = 2
SEM_DIM         = 1024
NUM_ITEMS       = 12102
SEQ_LEN         = 31
BATCH_SIZE      = 256
TOP_K_LIST      = [5, 10, 20]
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────────────────────


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
        out, _         = self.gru(x)
        output_tensor, _ = torch.max(out, dim=1)
        logit          = self.act(self.fc(out[:, -1, :]))
        return logit, out, output_tensor


def pad_to_length(seq, length=31):
    if len(seq) < length:
        return [0] * (length - len(seq)) + seq
    return seq[-length:]


def prepare_test_data(file_path):
    """
    For evaluation, use the FULL train sequence as input and
    predict the hold-out test item (last item in each user's list).
    """
    with open(file_path) as f:
        source_data = json.load(f)
    users, inputs, inputs_cnt, labels = [], [], [], []
    for key, items in source_data.items():
        # items[-1] is the test label (never seen during training)
        e_inputs = items[:-1]
        e_label  = items[-1]
        users.append(key)
        inputs.append(pad_to_length(e_inputs, SEQ_LEN))
        inputs_cnt.append(len(e_inputs))
        labels.append(e_label)
    return users, inputs, inputs_cnt, labels


def evaluate_topk(model, dataloader, top_k_list):
    model.eval()
    results = {k: {"hr": [], "ndcg": []} for k in top_k_list}
    max_k   = max(top_k_list)

    with torch.no_grad():
        for u, x, icnt, y in tqdm(dataloader, desc="Evaluating"):
            x      = x.to(DEVICE)
            y      = y.to(DEVICE)
            emb    = model.embedding(x)
            logit, _, _ = model(emb)                      # [B, num_items]
            _, topk_idx = torch.topk(logit, max_k, dim=-1)  # [B, max_k]

            for i in range(y.size(0)):
                true_item = y[i].item()
                pred_list = topk_idx[i].cpu().tolist()

                for k in top_k_list:
                    top_k_preds = pred_list[:k]
                    if true_item in top_k_preds:
                        rank = top_k_preds.index(true_item)
                        results[k]["hr"].append(1)
                        results[k]["ndcg"].append(1.0 / np.log2(rank + 2))
                    else:
                        results[k]["hr"].append(0)
                        results[k]["ndcg"].append(0.0)

    return {k: {"HR": float(np.mean(v["hr"])),
                "NDCG": float(np.mean(v["ndcg"]))}
            for k, v in results.items()}


def main():
    print(f"Device: {DEVICE}")
    print("Loading test data ...")
    users, inputs, inputs_cnt, labels = prepare_test_data(ITEM_LIST_PATH)
    users_t    = torch.tensor([int(u[4:]) for u in users])
    inputs_t   = torch.tensor(inputs, dtype=torch.long)
    inputs_cnt = torch.tensor(inputs_cnt, dtype=torch.long)
    labels_t   = torch.tensor(labels, dtype=torch.long)

    dataset    = TensorDataset(users_t, inputs_t, inputs_cnt, labels_t)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = GRU4RecECHO(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS,
                         NUM_ITEMS, SEM_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,
                                      weights_only=True))

    print("\n── Evaluating IADSR+ (ECHO) ──────────────────────────")
    iadsr_results = evaluate_topk(model, dataloader, TOP_K_LIST)

    # ── Print table ──────────────────────────────────────────────────────
    print(f"\n{'Metric':<15} {'IADSR+ (ECHO)':>15}")
    print("-" * 32)
    for k in TOP_K_LIST:
        print(f"HR@{k:<12} {iadsr_results[k]['HR']:>15.4f}")
    print("-" * 32)
    for k in TOP_K_LIST:
        print(f"NDCG@{k:<11} {iadsr_results[k]['NDCG']:>15.4f}")

    # ── Baseline numbers from paper (Table 2, GRU4Rec row) ───────────────
    # These are the published numbers for comparison in the paper/dashboard
    paper_baselines = {
        "GRU4Rec_base": {
            5: {"HR": 0.0153, "NDCG": 0.0087},
            10: {"HR": 0.0246, "NDCG": 0.0117},
            20: {"HR": 0.0390, "NDCG": 0.0143},
        },
        "IADSR_original_paper": {
            5: {"HR": 0.0300, "NDCG": 0.0196},
            10: {"HR": 0.0396, "NDCG": 0.0259},
            20: {"HR": 0.0486, "NDCG": 0.0321},
        },
    }

    output = {
        "IADSR_plus_ECHO": iadsr_results,
        "paper_baselines" : paper_baselines,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
