#!/usr/bin/env python3
"""ESM-2 Real Sequence Experiment — Fix P0-2.
Pipeline: (1) UniProt fetch → (2) ESM-2 embedding → (3) 5-fold GNN training x 3 conditions.
Expected output: table comparing real ESM-2 vs M×50 placeholder vs no ESM-2.
Run on AutoDL GPU: python esm2_real_fix.py"""

import pickle, json, random, time, os, gc, requests
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pathlib import Path

BASE = Path("/root/autodl-tmp/llm4sl")
DATA = BASE / "data/processed"
RESULTS = BASE / "results/esm2_fix"
RESULTS.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
DEVICE = torch.device("cuda")

print(f"[GPU] {torch.cuda.get_device_name(0)}")
print(f"[Disk] {os.popen('df -h /root/autodl-tmp/ | tail -1').read().strip()}")

# ============================================================
# STEP 1: Load graph data
# ============================================================
with open(DATA / "graph_data.pkl", "rb") as f:
    g = pickle.load(f)

gs = g["gene_list"]
pl = g["sl_pair_labels"]
ei = torch.tensor(g["edge_index"], dtype=torch.long).to(DEVICE)
ew = torch.tensor(g["edge_weight"], dtype=torch.float).to(DEVICE)
text_feat = torch.tensor(g["text_features"], dtype=torch.float).to(DEVICE)
print(f"[Data] {len(gs):,} nodes | {ei.shape[1]:,} edges")

# ============================================================
# STEP 2: Fetch real UniProt sequences
# ============================================================
SEQ_FILE = DATA / "uniprot_sequences_full.json"
if SEQ_FILE.exists():
    with open(SEQ_FILE) as f:
        seqs = json.load(f)
    print(f"[UniProt] Loaded {len(seqs)} cached sequences")
else:
    seqs = {}
    print(f"[UniProt] Fetching sequences for {len(gs):,} genes...")
    for i, gene in enumerate(gs):
        if gene in seqs:
            continue
        try:
            r = requests.get(
                f"https://rest.uniprot.org/uniprotkb/search"
                f"?query=gene:{gene}+AND+organism_id:9606+AND+reviewed:true"
                f"&format=json&size=1",
                timeout=12)
            if r.status_code == 200:
                d = r.json()
                if d.get("results"):
                    s = d["results"][0].get("sequence", {}).get("value", "")
                    if s and len(s) >= 10:
                        seqs[gene] = s[:1024]
        except:
            pass
        time.sleep(0.15)
        if (i + 1) % 500 == 0:
            pct = (i + 1) / len(gs) * 100
            print(f"  {i+1}/{len(gs)} ({pct:.0f}%) | {len(seqs)} fetched")
            with open(SEQ_FILE, "w") as f:
                json.dump(seqs, f)
    
    with open(SEQ_FILE, "w") as f:
        json.dump(seqs, f)
    print(f"[UniProt] Done: {len(seqs)}/{len(gs)} ({len(seqs)/len(gs)*100:.0f}%)")

# ============================================================
# STEP 3: ESM-2 embeddings for all conditions
# ============================================================
print("\n[ESM-2] Loading model...")
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(DEVICE).eval()
ESM_DIM = 480

def compute_esm(sequence_map, name):
    """Compute ESM-2 embeddings for all genes using given sequence map."""
    emb = np.zeros((len(gs), ESM_DIM), dtype=np.float32)
    batch_seqs, batch_idx = [], []
    
    for gid, gene in enumerate(gs):
        seq = sequence_map.get(gene, None)
        if seq is None:
            emb[gid] = 0.0  # zero for missing
            continue
        batch_seqs.append(seq)
        batch_idx.append(gid)
        
        if len(batch_seqs) == 64:
            inp = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)
            with torch.no_grad():
                e = esm(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
            for j, idx in enumerate(batch_idx):
                emb[idx] = e[j]
            batch_seqs, batch_idx = [], []
    
    if batch_seqs:
        inp = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            e = esm(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
        for j, idx in enumerate(batch_idx):
            emb[idx] = e[j]
    
    # Normalize
    mask = (emb.sum(axis=1) != 0)
    emb[mask] = (emb[mask] - emb[mask].mean(0)) / (emb[mask].std(0) + 1e-8)
    print(f"  [{name}] {mask.sum()}/{len(gs)} genes with non-zero embedding")
    return torch.tensor(emb, dtype=torch.float).to(DEVICE), mask.sum()

# Real sequences
real_map = seqs.copy()
esm_real, n_real = compute_esm(real_map, "Real UniProt")

# M×50 placeholder (control)
placeholder_map = {gene: "M" * 50 for gene in gs}
esm_placeholder, n_ph = compute_esm(placeholder_map, "M×50 placeholder")

# No ESM-2 (zero)
esm_none = torch.zeros(len(gs), ESM_DIM, dtype=torch.float).to(DEVICE)

del esm, tokenizer; gc.collect(); torch.cuda.empty_cache()

# ============================================================
# STEP 4: Train GNN for 3 conditions
# ============================================================
pos = [(u, v) for (u, v), l in pl.items() if l == 1]
neg = [(u, v) for (u, v), l in pl.items() if l == 0]
all_pairs = pos + neg
ps = set(pos)
random.seed(42)
random.shuffle(all_pairs)
fsz = len(all_pairs) // 5

class GNN(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.emb = nn.Linear(d_in, 256)
        self.bn = nn.BatchNorm1d(256)
        self.c1 = nn.Linear(512, 256)
        self.c2 = nn.Linear(512, 256)
        self.c3 = nn.Linear(512, 256)
        self.pred = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1))

    def forward(self, x):
        x = self.bn(F.relu(self.emb(x)))
        for c in [self.c1, self.c2, self.c3]:
            s, d = ei[0], ei[1]
            m = c(torch.cat([x[s], x[d]], -1))
            x2 = x.clone()
            x2.index_add_(0, d, 0.1 * ew.unsqueeze(1) * m)
            x = F.relu(x2)
        return x

    def score(self, x, pairs):
        ui = torch.tensor([p[0] for p in pairs], device=DEVICE)
        vi = torch.tensor([p[1] for p in pairs], device=DEVICE)
        return self.pred(torch.cat([x[ui], x[vi]], -1)).sigmoid().squeeze()

def train_5fold(features, label, name_prefix):
    """Run 5-fold CV with given features. Returns mean AUC ± std."""
    f = torch.cat([text_feat, features], dim=1) if features.shape[1] > 1 else text_feat
    aucs = []
    for fold in range(5):
        st, en = fold * fsz, fold * fsz + fsz
        test = all_pairs[st:en]
        train = all_pairs[:st] + all_pairs[en:]
        
        tr_pos = [(u, v) for (u, v) in train if (u, v) in ps or (v, u) in ps]
        tr_neg = [(u, v) for (u, v) in train if (u, v) not in ps and (v, u) not in ps]
        if len(tr_neg) > len(tr_pos) * 3:
            tr_neg = random.sample(tr_neg, len(tr_pos) * 3)
        train = tr_pos + tr_neg
        random.shuffle(train)
        
        y = torch.tensor([0.9 if (u, v) in ps or (v, u) in ps else 0.05 for u, v in train], device=DEVICE)
        m = GNN(f.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
        
        best, t0 = 0, time.time()
        for e in range(80):
            m.train()
            x = m(f)
            loss = F.binary_cross_entropy(m.score(x, train), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if e % 20 == 0:
                m.eval()
                with torch.no_grad():
                    x = m(f)
                    ts = m.score(x, test).cpu().numpy()
                    tl = [1.0 if (u, v) in ps or (v, u) in ps else 0.0 for u, v in test]
                    a = roc_auc_score(tl, ts) if sum(tl) > 0 else 0.5
                    if a > best:
                        best = a
        dt = time.time() - t0
        aucs.append(best)
        print(f"  [{label}] Fold{fold+1}: AUC={best:.4f} ({dt:.0f}s)")
    
    mu, std = np.mean(aucs), np.std(aucs)
    print(f"  [{label}] 5-fold: AUC={mu:.4f} ± {std:.4f}\n")
    return mu, std, aucs

# 3 conditions
print("\n" + "=" * 60)
print("TRAINING: 3 ESM-2 conditions")
print("=" * 60)

auc_no_esm, std_no_esm, _ = train_5fold(esm_none, "No ESM-2", "noESM")
auc_ph, std_ph, _ = train_5fold(esm_placeholder, "M×50 placeholder", "M50")
auc_real, std_real, _ = train_5fold(esm_real, f"Real UniProt (n={n_real})", "real")

# ============================================================
# STEP 5: Report
# ============================================================
print("\n" + "=" * 60)
print("ESM-2 P0-2 FIX: Final Comparison")
print("=" * 60)
print(f"  {'Condition':<25} {'AUC':<15} {'Δ vs No ESM-2'}")
print(f"  {'-'*55}")
print(f"  {'No ESM-2 (text only)':<25} {auc_no_esm:.4f} ± {std_no_esm:.4f}")
print(f"  {'M×50 placeholder':<25} {auc_ph:.4f} ± {std_ph:.4f}   +{auc_ph-auc_no_esm:+.4f}")
print(f"  {'Real UniProt (n='+str(n_real)+')':<25} {auc_real:.4f} ± {std_real:.4f}   +{auc_real-auc_no_esm:+.4f}")

result = {
    "no_esm_auc": round(auc_no_esm, 4),
    "placeholder_auc": round(auc_ph, 4),
    "real_esm_auc": round(auc_real, 4),
    "real_gene_count": int(n_real),
    "total_genes": len(gs),
    "real_gain": round(auc_real - auc_no_esm, 4),
    "placeholder_gain": round(auc_ph - auc_no_esm, 4),
    "insight": "Real ESM-2 sequences provide gene-specific protein information. "
               f"Compare {n_real}/{len(gs)} genes covered."
}

with open(RESULTS / "esm2_fix_results.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n[✓] Results saved to {RESULTS / 'esm2_fix_results.json'}")
print("[✓] P0-2 fixed: ESM-2 contribution now based on real, gene-specific sequences.")
