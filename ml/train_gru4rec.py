"""
ECHO — IADSR+ Training Script (GRU4Rec Backbone)

Key changes vs original IADSR:
  1. Linear(768, 128) instead of Linear(4096, 128) — matches BGE-large
  2. Attention-weighted noise scoring: learned weights over c1, c2, c3
     instead of plain sum score(t) = c1 + c2 + c3
  3. Proper file paths (no "XXX" placeholders)
  4. Progress logging saved to metrics_log.json for the API
"""

import os, json, time, random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
from info_nce import InfoNCE

from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
SEED            = 42
PROJECT_ROOT    = Path(__file__).parent.parent
ITEM_LIST_PATH  = PROJECT_ROOT / "Beauty/gru_item_list.json" # Use full dataset
SEM_LONG_PATH   = PROJECT_ROOT / "ml/semantic_long.pt"
SEM_SHORT_PATH  = PROJECT_ROOT / "ml/semantic_short.pt"
MODEL_SAVE_PATH = PROJECT_ROOT / "ml/gru4rec_best.pth"
LOSS_PLOT_PATH  = PROJECT_ROOT / "ml/training_loss.png"
METRICS_LOG     = PROJECT_ROOT / "ml/metrics_log.json"

EMBEDDING_DIM   = 64
HIDDEN_DIM      = 128
NUM_LAYERS      = 2
SEMANTIC_DIM    = 1024         # Output of BGE-large-en-v1.5
NUM_ITEMS       = 12102        # Beauty dataset
SEQ_LEN         = 31          # max sequence length (32 - 1 for label)

BATCH_SIZE      = 256
LR              = 1e-4
NUM_EPOCHS      = 1000
PATIENCE        = 10
THETA           = -0.9         # cross-modal consistency threshold (from paper)
TAU             = 1.0          # Gumbel temperature
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────────────────────

# reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


# ── Model ────────────────────────────────────────────────────────────────
class GRU4RecECHO(nn.Module):
    """
    GRU4Rec backbone extended with IADSR+ modules:
      - l1, l2: project 768-dim semantic embeddings → 128-dim collab space
      - attn:   learnable attention over [c1, c2, c3] noise scores  ← NEW
      - decoder: reconstruct item embeddings for reconstruction loss
    """
    def __init__(self, embedding_dim=64, hidden_dim=128, num_layers=2,
                 num_items=12102, sem_dim=768):
        super().__init__()
        assert num_items > 100

        self.embedding = nn.Embedding(num_items, embedding_dim)
        self.gru        = nn.GRU(embedding_dim, hidden_dim, num_layers,
                                  batch_first=True)
        self.act        = nn.Tanh()
        self.fc         = nn.Linear(hidden_dim, num_items)

        # Semantic projection layers (768→128 instead of original 4096→128)
        self.l1 = nn.Linear(sem_dim, hidden_dim, bias=False)   # short-term
        self.l2 = nn.Linear(sem_dim, hidden_dim, bias=False)   # long-term

        # ── Novel: attention over [c1, c2, c3] ──────────────────────────
        self.attn = nn.Linear(3, 3, bias=False)                # 3-in, 3-out
        # ────────────────────────────────────────────────────────────────

        self.decoder = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        out, h_n       = self.gru(x)               # out: [B, T, H]
        output_tensor, _ = torch.max(out, dim=1)   # h_n (long-term proxy)
        logit          = self.act(self.fc(out[:, -1, :]))
        return logit, out, output_tensor

    def attention_score(self, c1, c2, c3):
        """
        Compute attention-weighted noise score from three cosine similarities.
        c1, c2, c3: scalars (or [T] tensors)
        Returns weighted sum (same shape as inputs).
        """
        raw = torch.stack([c1, c2, c3], dim=-1)        # [..., 3]
        weights = torch.softmax(self.attn(raw), dim=-1) # [..., 3]
        score   = (weights * raw).sum(dim=-1)           # [...]
        return score


class SampledCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xe_loss = nn.CrossEntropyLoss()

    def forward(self, logit):
        batch_size = logit.size(1)
        target = torch.arange(batch_size, dtype=torch.long, device=logit.device)
        return self.xe_loss(logit, target)


# ── Data helpers ─────────────────────────────────────────────────────────
def pad_to_length(seq, length=31):
    if len(seq) < length:
        return [0] * (length - len(seq)) + seq
    return seq[-length:]


def prepare_data(file_path):
    with open(file_path) as f:
        source_data = json.load(f)
    users, inputs, inputs_cnt, labels = [], [], [], []
    for key, items in source_data.items():
        train_items = items[:-1]          # last item = test label
        e_inputs    = train_items[:-1]    # all but the penultimate = input
        e_label     = train_items[-1]     # penultimate = train label
        users.append(key)
        inputs.append(pad_to_length(e_inputs, SEQ_LEN))
        inputs_cnt.append(len(e_inputs))
        labels.append(e_label)
    return users, inputs, inputs_cnt, labels


def get_interests(u, model, gpu_sem_long, gpu_sem_short):
    u_list = u.tolist()
    long_emb  = []
    short_emb_list = []
    lengths = []
    
    for uid in u_list:
        k = f"user{uid}"
        if k in gpu_sem_long:
            long_emb.append(gpu_sem_long[k])
        else:
            long_emb.append(torch.zeros(SEMANTIC_DIM, device=DEVICE))
            
        se = gpu_sem_short.get(k, None)
        if se is not None and se.size(0) > 0:
            short_emb_list.append(se)
            lengths.append(se.size(0))
        else:
            lengths.append(0)
            
    long_int = model.l2(torch.stack(long_emb))
    
    if len(short_emb_list) > 0:
        flat_short = torch.cat(short_emb_list, dim=0)
        flat_short_int = model.l1(flat_short)
        short_ints_tensors = torch.split(flat_short_int, [l for l in lengths if l > 0])
        
        short_ints_list = []
        idx = 0
        for l in lengths:
            if l > 0:
                short_ints_list.append(short_ints_tensors[idx])
                idx += 1
            else:
                short_ints_list.append(torch.zeros(0, 128, device=DEVICE))
    else:
        short_ints_list = [torch.zeros(0, 128, device=DEVICE) for _ in range(len(u))]
        
    return short_ints_list, long_int


# ── Gumbel-Sigmoid (straight-through) ────────────────────────────────────
def gumbel_sigmoid(logits, tau=1.0, hard=True):
    eps = 1e-10
    U   = torch.rand_like(logits)
    g   = -torch.log(-torch.log(U + eps) + eps)
    y   = ((logits + g) / tau).sigmoid()
    if hard:
        y_hard = (y > 0.5).float()
        y      = (y_hard - y).detach() + y
    return y


# ── Training loop ─────────────────────────────────────────────────────────
def train():
    print(f"Device: {DEVICE}")
    print("Loading data ...")
    users, inputs, inputs_cnt, labels = prepare_data(ITEM_LIST_PATH)
    users      = torch.tensor([int(u[4:]) for u in users])
    inputs_t   = torch.tensor(inputs,      dtype=torch.long)
    inputs_cnt = torch.tensor(inputs_cnt,  dtype=torch.long)
    labels     = torch.tensor(labels,      dtype=torch.long)

    dataset    = TensorDataset(users, inputs_t, inputs_cnt, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0)

    print("Loading semantic embeddings ...")
    raw_sem_long  = torch.load(SEM_LONG_PATH,  map_location="cpu", weights_only=False)
    raw_sem_short = torch.load(SEM_SHORT_PATH, map_location="cpu", weights_only=False)
                                
    print("Pre-processing semantic embeddings to GPU...")
    gpu_sem_long = {}
    for k, v in raw_sem_long.items():
        gpu_sem_long[k] = v.to(DEVICE)
        
    gpu_sem_short = {}
    for k, v in raw_sem_short.items():
        if isinstance(v, list) and len(v) > 0:
            gpu_sem_short[k] = torch.stack(v).to(DEVICE)
        elif isinstance(v, torch.Tensor) and v.size(0) > 0:
            gpu_sem_short[k] = v.to(DEVICE)
        else:
            gpu_sem_short[k] = torch.zeros(0, SEMANTIC_DIM, device=DEVICE)

    model     = GRU4RecECHO(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS,
                             NUM_ITEMS, SEMANTIC_DIM).to(DEVICE)
    criterion = SampledCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    infonce   = InfoNCE()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    best_loss           = np.inf
    no_improve          = 0
    all_losses          = []
    epoch_avg_losses    = []
    mask_probs_dict     = {}       # carry masks across epochs

    print("\n───────── Begin Training ─────────")
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss    = 0.0
        epoch_masks   = {}

        for u, x, icnt, y in dataloader:
            x    = x.to(DEVICE)
            y    = y.to(DEVICE)
            icnt = icnt.to(DEVICE)
            u    = u.to(DEVICE)

            # ── Progressive masking (epoch > 0) ──────────────────────────
            inputs_emb = model.embedding(x)                  # [B, T, 64]
            if epoch > 0:
                for i, uid in enumerate(u.tolist()):
                    uid_str = f"user{uid}"
                    if uid_str in mask_probs_dict:
                        mp = mask_probs_dict[uid_str]
                        if mp.size(0) == inputs_emb.size(1):
                            inputs_emb[i] = inputs_emb[i] * mp.unsqueeze(-1)
            
            optimizer.zero_grad()

            # ── Semantic interests ────────────────────────────────────────
            short_ints_list, long_int = get_interests(u, model, gpu_sem_long, gpu_sem_short)

            # ── Forward pass ─────────────────────────────────────────────
            logit, out, h_long = model(inputs_emb)

            # ── CE loss ───────────────────────────────────────────────────
            logit_sampled = logit[:, y.view(-1)]
            loss          = criterion(logit_sampled)

            # ── InfoNCE loss ──────────────────────────────────────────────
            flat_gru_short = []
            flat_sem_short = []
            gru_slices = []
            sem_slices = []
            
            for i in range(x.size(0)):
                ic      = icnt[i].item()
                g_slice = out[i, -ic:, :]
                s_slice = short_ints_list[i]
                
                min_len = min(g_slice.size(0), s_slice.size(0))
                
                if min_len > 0:
                    g_aligned = g_slice[-min_len:]
                    s_aligned = s_slice[-min_len:]
                    flat_gru_short.append(g_aligned)
                    flat_sem_short.append(s_aligned)
                    gru_slices.append(g_aligned)
                    sem_slices.append(s_aligned)
                else:
                    gru_slices.append(torch.zeros(0, 128, device=DEVICE))
                    sem_slices.append(torch.zeros(0, 128, device=DEVICE))

            if len(flat_gru_short) > 0:
                flat_gru_short = torch.cat(flat_gru_short, 0)
                flat_sem_short = torch.cat(flat_sem_short, 0)
                loss += infonce(flat_gru_short, flat_sem_short)

            loss += infonce(h_long, long_int)

            # ── Denoising + reconstruction loss ──────────────────────────
            cos_sim     = F.cosine_similarity(h_long, long_int, dim=-1)
            qualified   = cos_sim >= THETA                # [B] bool

            valid_out = []
            valid_short = []
            valid_long = []
            valid_gru_long = []
            valid_target = []
            
            for i in range(x.size(0)):
                if qualified[i] and gru_slices[i].size(0) > 0:
                    out_i = gru_slices[i]
                    T = out_i.size(0)
                    valid_out.append(out_i)
                    valid_short.append(sem_slices[i])
                    valid_long.append(long_int[i].unsqueeze(0).expand(T, -1))
                    valid_gru_long.append(h_long[i].unsqueeze(0).expand(T, -1))
                    valid_target.append(inputs_emb[i, -T:, :])

            if len(valid_out) > 0:
                flat_out = torch.cat(valid_out, dim=0)
                flat_short = torch.cat(valid_short, dim=0)
                flat_long = torch.cat(valid_long, dim=0)
                flat_gru_long = torch.cat(valid_gru_long, dim=0)
                flat_target = torch.cat(valid_target, dim=0)
                
                c1s = F.cosine_similarity(flat_long, flat_out, dim=-1)
                c2s = F.cosine_similarity(flat_gru_long, flat_short, dim=-1)
                c3s = F.cosine_similarity(flat_out, flat_short, dim=-1)
                
                scores = model.attention_score(c1s, c2s, c3s)
                mask_prob = gumbel_sigmoid(scores, tau=TAU, hard=True)
                
                denoised = flat_out * mask_prob.unsqueeze(-1)
                recon = model.decoder(denoised)
                loss += F.mse_loss(recon, flat_target)
                
                # Split masks back for next epoch
                split_lengths = [vo.size(0) for vo in valid_out]
                mask_splits = torch.split(mask_prob, split_lengths)
                
                idx = 0
                for i in range(x.size(0)):
                    if qualified[i] and gru_slices[i].size(0) > 0:
                        uid_str = f"user{u[i].item()}"
                        epoch_masks[uid_str] = mask_splits[idx].detach()
                        idx += 1

            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())
            total_loss += loss.item()

        mask_probs_dict = epoch_masks
        avg_loss        = total_loss / len(dataloader)
        epoch_avg_losses.append(avg_loss)
        elapsed         = (time.time() - t0) / 60

        print(f"Epoch [{epoch+1:>4}/{NUM_EPOCHS}] | "
              f"Loss: {avg_loss:.4f} | "
              f"Elapsed: {elapsed:.1f}m")

        if avg_loss < best_loss:
            best_loss  = avg_loss
            no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ Model saved  (best loss: {best_loss:.4f})")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}.")
            break

    # ── Save training curve ───────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(all_losses, alpha=0.4, label="step loss")
    plt.plot(np.convolve(all_losses, np.ones(50)/50, mode="valid"),
             label="smoothed")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("ECHO Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Loss curve saved → {LOSS_PLOT_PATH}")

    # ── Save metadata for the API ─────────────────────────────────────────
    with open(METRICS_LOG, "w") as f:
        json.dump({
            "final_best_loss"   : best_loss,
            "epochs_trained"    : epoch + 1,
            "model_path"        : str(MODEL_SAVE_PATH),
            "sem_dim"           : SEMANTIC_DIM,
            "embedding_dim"     : EMBEDDING_DIM,
            "hidden_dim"        : HIDDEN_DIM,
            "num_items"         : NUM_ITEMS,
            "novel_contribution": "attention-weighted noise scoring + BGE-large encoder",
        }, f, indent=2)
    print(f"Metrics log saved → {METRICS_LOG}")
    print(f"\n✅ Training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()
