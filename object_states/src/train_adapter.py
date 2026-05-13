#!/usr/bin/env python3
"""
train_adapter.py
----------------
Trains and evaluates the MLP adapter on frozen SigLIP 2 So400m features.

Architecture (from CLAUDE.md):
  SigLIP 2 So400m frozen (1152-dim pooled output)
  → Linear(1152→512) → GELU → Dropout(0.1)
  → Linear(512→256)  → GELU → Dropout(0.1)
  → Linear(256→num_pairs)     one logit per state pair

Loss: BCEWithLogitsLoss with per-sample applicability mask.
      Only the head for the sample's own state pair receives gradient.

Training: AdamW lr=1e-3, weight_decay=1e-4, cosine schedule, 10 epochs, batch=32.

Evaluation:
  - Per-pair AP on 20% sim holdout
  - Comparison table: zero-shot AP vs linear probe AP vs adapter AP
"""

import os, sys, pickle
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
ROOT        = Path(__file__).parent.parent
DATA        = ROOT / "data"
RESULTS_CSV = ROOT / "results" / "csv"
sys.path.insert(0, str(ROOT / "utils"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, AutoModel
import pandas as pd

CROPS_PATH  = DATA / "sim_crops_all.pkl"
CACHE_PATH  = DATA / "embeddings_all_cache.npz"
MODEL_PATH  = DATA / "adapter.pt"

PAIR_NAMES = ["full_empty", "open_closed", "on_off", "cooked_raw", "dirty_clean", "broken_intact"]
NUM_PAIRS  = len(PAIR_NAMES)

BATCH_SIZE    = 32
EPOCHS_SWEEP  = 20           # epochs per candidate during LR selection
EPOCHS        = 30           # epochs for final training run
LR_CANDIDATES = [1e-4, 3e-4, 1e-3]
WEIGHT_DECAY  = 1e-3         # stronger regularisation for 723K params / ~500 samples
DROPOUT       = 0.2          # increased from 0.1 for same reason
EMBED_DIM     = 1152

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

# ── Load crops ───────────────────────────────────────────────────────────────
with open(CROPS_PATH, "rb") as f:
    samples = pickle.load(f)

images    = [s["image"]    for s in samples]
labels    = np.array([s["label"]    for s in samples])
pair_idxs = np.array([s["pair_idx"] for s in samples])

print(f"Loaded {len(samples)} crops across {NUM_PAIRS} state pairs")
for i, name in enumerate(PAIR_NAMES):
    mask = pair_idxs == i
    n0 = int((labels[mask] == 0).sum())
    n1 = int((labels[mask] == 1).sum())
    print(f"  [{i}] {name:<16} neg={n0:>3}  pos={n1:>3}")

# ── Extract or load embeddings ────────────────────────────────────────────────
def extract_embeddings(images):
    print("\nLoading SigLIP 2 So400m...")
    proc  = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-384")
    model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-384").to(device).eval()
    embs  = []
    for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Extracting embeddings"):
        batch  = images[i : i + BATCH_SIZE]
        inputs = proc(images=batch, return_tensors="pt", padding="max_length").to(device)
        with torch.no_grad():
            pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        embs.append(pooled.cpu().float().numpy())
    del model
    if device == "mps": torch.mps.empty_cache()
    return np.concatenate(embs)

if os.path.exists(CACHE_PATH):
    print(f"Loading cached embeddings → {CACHE_PATH}")
    embeddings = np.load(CACHE_PATH)["embeddings"]
else:
    embeddings = extract_embeddings(images)
    np.savez(CACHE_PATH, embeddings=embeddings)
    print(f"Saved → {CACHE_PATH}")

# ── Train / test split (stratified per pair) ──────────────────────────────────
# Stratify jointly on (pair_idx, label) to keep class balance per pair
strat_key = pair_idxs * 2 + labels
idx = np.arange(len(samples))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=strat_key)

X_train, y_train, p_train = embeddings[idx_train], labels[idx_train], pair_idxs[idx_train]
X_test,  y_test,  p_test  = embeddings[idx_test],  labels[idx_test],  pair_idxs[idx_test]
print(f"\nTrain: {len(idx_train)}  Test: {len(idx_test)}")

# Carve 10% of train for LR selection — stratified, never touches the test set
idx_sub, idx_val_sel = train_test_split(
    np.arange(len(idx_train)), test_size=0.1, random_state=42,
    stratify=p_train * 2 + y_train)
X_sub,     y_sub,     p_sub     = X_train[idx_sub],     y_train[idx_sub],     p_train[idx_sub]
X_val_sel, y_val_sel, p_val_sel = X_train[idx_val_sel], y_train[idx_val_sel], p_train[idx_val_sel]
print(f"LR selection split — sub-train: {len(idx_sub)}  val: {len(idx_val_sel)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class StateDataset(Dataset):
    def __init__(self, embeddings, labels, pair_idxs):
        self.X = torch.from_numpy(embeddings).float()
        self.y = torch.from_numpy(labels).float()
        self.p = torch.from_numpy(pair_idxs).long()

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.p[i]

test_loader = DataLoader(StateDataset(X_test, y_test, p_test),
                         batch_size=BATCH_SIZE, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
class StateAdapter(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, num_pairs=NUM_PAIRS, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),   nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_pairs),
        )

    def forward(self, x):
        return self.net(x)   # [B, num_pairs]

_tmp = StateAdapter()
n_params = sum(p.numel() for p in _tmp.parameters())
print(f"Adapter parameters: {n_params:,}  (~{n_params/1e6:.2f}M)")
del _tmp

# ── Loss ──────────────────────────────────────────────────────────────────────
def masked_bce_loss(logits, labels, pair_idxs):
    B = logits.shape[0]
    target = torch.full((B, NUM_PAIRS), -1.0, device=logits.device)
    for i in range(B):
        target[i, pair_idxs[i]] = labels[i]
    mask = (target >= 0).float()
    loss = F.binary_cross_entropy_with_logits(logits, target.clamp(0, 1), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)

# ── Helpers ───────────────────────────────────────────────────────────────────
def train_model(X_tr, y_tr, p_tr, epochs, lr, verbose=False):
    m   = StateAdapter().to(device)
    opt = AdamW(m.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=epochs)
    dl  = DataLoader(StateDataset(X_tr, y_tr, p_tr),
                     batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    for epoch in range(1, epochs + 1):
        m.train()
        total_loss = 0.0
        for x, yb, pb in dl:
            x, yb, pb = x.to(device), yb.to(device), pb.to(device)
            opt.zero_grad()
            loss = masked_bce_loss(m(x), yb, pb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(x)
        sch.step()
        if verbose:
            print(f"  Epoch {epoch:>2}/{epochs}  loss={total_loss/len(X_tr):.4f}")
    return m

def eval_mean_ap(m, X, y, p):
    m.eval()
    with torch.no_grad():
        logits = m(torch.from_numpy(X).float().to(device)).cpu().numpy()
    aps = []
    for i in range(NUM_PAIRS):
        mask = p == i
        if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
            continue
        scores = torch.sigmoid(torch.from_numpy(logits[mask, i])).numpy()
        aps.append(average_precision_score(y[mask], scores))
    return float(np.mean(aps)) if aps else 0.0

# ── LR selection ──────────────────────────────────────────────────────────────
print(f"\nLR sweep ({EPOCHS_SWEEP} epochs each, selecting on val mean-AP)...")
best_lr, best_val_ap = LR_CANDIDATES[0], -1.0
for lr in LR_CANDIDATES:
    m_cand = train_model(X_sub, y_sub, p_sub, EPOCHS_SWEEP, lr)
    val_ap = eval_mean_ap(m_cand, X_val_sel, y_val_sel, p_val_sel)
    marker = " ←" if val_ap > best_val_ap else ""
    print(f"  lr={lr:.0e}  val mean-AP={val_ap:.3f}{marker}")
    if val_ap > best_val_ap:
        best_val_ap, best_lr = val_ap, lr
print(f"Selected lr={best_lr:.0e}  (val mean-AP={best_val_ap:.3f})")

# ── Final training on full train set ──────────────────────────────────────────
print(f"\nFinal training: lr={best_lr:.0e}, {EPOCHS} epochs...")
model = train_model(X_train, y_train, p_train, EPOCHS, best_lr, verbose=True)

torch.save(model.state_dict(), MODEL_PATH)
print(f"Saved → {MODEL_PATH}")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
all_logits, all_labels, all_pairs = [], [], []
with torch.no_grad():
    for x, y, p in test_loader:
        logits = model(x.to(device)).cpu()
        all_logits.append(logits)
        all_labels.append(y)
        all_pairs.append(p)

all_logits = torch.cat(all_logits).numpy()    # [N_test, NUM_PAIRS]
all_labels = torch.cat(all_labels).numpy()
all_pairs  = torch.cat(all_pairs).numpy()

print(f"\n{'='*60}")
print(f"{'Pair':<18} {'n_test':>6} {'Adapter AP':>12} {'Adapter Acc':>12}")
print(f"{'-'*60}")

pair_aps, pair_accs = [], []
rows = []
for i, name in enumerate(PAIR_NAMES):
    mask = all_pairs == i
    if mask.sum() < 2:
        continue
    scores = torch.sigmoid(torch.from_numpy(all_logits[mask, i])).numpy()
    gt     = all_labels[mask]
    if len(np.unique(gt)) < 2:
        continue
    ap  = average_precision_score(gt, scores)
    acc = float((scores >= 0.5).astype(int) == gt.astype(int)).mean() if False else \
          float(((scores >= 0.5).astype(int) == gt.astype(int)).mean())
    pair_aps.append(ap); pair_accs.append(acc)
    print(f"{name:<18} {int(mask.sum()):>6} {ap:>12.3f} {acc:>12.3f}")
    rows.append({"pair": name, "n_test": int(mask.sum()), "adapter_ap": ap, "adapter_acc": acc})

mean_ap  = np.mean(pair_aps)
mean_acc = np.mean(pair_accs)
print(f"{'-'*60}")
print(f"{'Mean':<18} {'':>6} {mean_ap:>12.3f} {mean_acc:>12.3f}")
print(f"{'='*60}")

# ── Summary vs zero-shot and linear probe ─────────────────────────────────────
# Reference numbers from earlier experiments (full_empty only)
ZS_SIGLIP_LIQUID = 0.881
LP_SIGLIP_SIM    = 0.954

full_empty_ap = next((r["adapter_ap"] for r in rows if r["pair"] == "full_empty"), None)

print(f"\nfull/empty  —  zero-shot: {ZS_SIGLIP_LIQUID:.3f}  "
      f"linear probe: {LP_SIGLIP_SIM:.3f}  "
      f"adapter: {f'{full_empty_ap:.3f}' if full_empty_ap is not None else 'n/a'}")
print(f"Mean AP across {len(pair_aps)} pairs: {mean_ap:.3f}")

df = pd.DataFrame(rows)
df.to_csv(RESULTS_CSV / "results_adapter.csv", index=False)
print(f"\nSaved → {RESULTS_CSV / 'results_adapter.csv'}")
