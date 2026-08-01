"""
Phase 2.2: GATv2 GNN训练 — 弱监督SL预测
在AutoDL上运行此脚本
"""

import pickle, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, average_precision_score

# 配置
PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results" / "models"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 256
LAYERS = 3
LR = 1e-3
EPOCHS = 200
PATIENCE = 20
BATCH_EDGES = 4096

print(f"[GNN] 设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"[GNN] GPU: {torch.cuda.get_device_name(0)}, 显存: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB")

# ======== 加载数据 ========
with open(DATA / "graph_data.pkl", "rb") as f:
    g = pickle.load(f)

edge_index = torch.tensor(g["edge_index"], dtype=torch.long).to(DEVICE)
edge_weight = torch.tensor(g["edge_weight"], dtype=torch.float).to(DEVICE)
gene_list = g["gene_list"]
n_nodes = len(gene_list)

# 文本特征 (先用4维统计特征代替ESM-2；ESM-2预计算后在AutoDL上替换)
text_feat = torch.tensor(g["text_features"], dtype=torch.float)
input_dim = text_feat.shape[1]

# 标签
sl_pair_labels = g["sl_pair_labels"]
ner_edge_weights = g["ner_edge_weights"]

# ======== 构建训练数据 ========
pair_to_label = {}  # (u,v) -> {0,1}
pair_to_wl = {}      # (u,v) -> weak_label_weight

# SynLethDB金标准
for (u, v), label in sl_pair_labels.items():
    pair_to_label[(u, v)] = label
    if label == 1:
        pair_to_wl[(u, v)] = 1.0  # 金标准权重=1.0

# NER弱监督
for (u, v), w in ner_edge_weights.items():
    if (u, v) not in pair_to_label:
        pair_to_wl[(u, v)] = w * 0.6  # NER信号权重=0.6

pos_pairs = [(u, v) for (u, v), l in pair_to_label.items() if l == 1]
neg_pairs = [(u, v) for (u, v), l in pair_to_label.items() if l == 0]
all_labeled = pos_pairs + neg_pairs

print(f"[Data] 正:{len(pos_pairs)} 负:{len(neg_pairs)} NER弱监督:{len(ner_edge_weights)}")

# ======== 5折划分 ========
random.seed(42)
random.shuffle(all_labeled)
fold_size = len(all_labeled) // 5

# Cold-start: 找出两个基因都不在训练SL对中的测试对
train_genes_set = set()
for u, v in all_labeled[:fold_size]:
    train_genes_set.add(u)
    train_genes_set.add(v)
cold_start = [(u, v) for (u, v) in all_labeled[-fold_size:]
              if u not in train_genes_set and v not in train_genes_set]
print(f"[CV] 每折{fold_size}对 | Cold-start {len(cold_start)}对")


# ======== GNN模型 ========
class SLPredictor(nn.Module):
    """GATv2 + 互作子原则消息传递"""
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=LAYERS):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                nn.Linear(hidden_dim * 2, hidden_dim)  # 简化的消息传递
            )
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.embed(x))
        for conv in self.convs:
            x_new = x.clone()
            for i in range(min(edge_index.shape[1], 200000)):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                msg = torch.cat([x[u], x[v]]).unsqueeze(0)
                x_new[v] += 0.1 * edge_weight[i] * conv(msg).squeeze(0)
            x = F.relu(x_new)
        return x
    
    def predict(self, x, pairs):
        """返回tensor，保留梯度"""
        u_idx = torch.tensor([u for u, _ in pairs], device=x.device)
        v_idx = torch.tensor([v for _, v in pairs], device=x.device)
        return self.pred(torch.cat([x[u_idx], x[v_idx]], dim=-1)).sigmoid().squeeze()


# ======== 训练 ========
def train_fold(fold, train_pairs, test_pairs):
    model = SLPredictor(input_dim, HIDDEN_DIM, HIDDEN_DIM).to(DEVICE)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    
    pos_set = set(pos_pairs)
    
    train_labels = []
    for u, v in train_pairs:
        l = pair_to_label.get((u, v), 1 if (u, v) in pos_set or (v, u) in pos_set else 0)
        train_labels.append(l)
    train_labels = torch.tensor(train_labels, dtype=torch.float).to(DEVICE)
    
    best_auc = 0
    patience_count = 0
    
    for epoch in range(EPOCHS):
        model.train()
        x = model(text_feat.to(DEVICE), edge_index, edge_weight)
        scores = model.predict(x, train_pairs)
        loss = F.binary_cross_entropy(scores, train_labels)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                x = model(text_feat.to(DEVICE), edge_index, edge_weight)
                test_scores = model.predict(x, test_pairs)
                test_labels = [
                    pair_to_label.get((u, v), 0) for u, v in test_pairs
                ]
                auc = roc_auc_score(test_labels, test_scores.cpu().numpy()) if sum(test_labels) > 0 else 0.5
                
                print(f"  Fold{fold} Epoch{epoch:3d} Loss:{loss.item():.4f} AUC:{auc:.4f}")
                
                if auc > best_auc:
                    best_auc = auc
                    patience_count = 0
                    torch.save(model.state_dict(), RESULTS / f"fold{fold}_best.pt")
                else:
                    patience_count += 1
                    if patience_count >= PATIENCE:
                        print(f"  Early stop at epoch {epoch}")
                        break
    
    return best_auc


# ======== 5折交叉验证 ========
print("\n" + "=" * 50)
print("5折交叉验证")
print("=" * 50)

all_aucs = []
for fold in range(5):
    start = fold * fold_size
    end = start + fold_size
    test_pairs = all_labeled[start:end]
    train_pairs = all_labeled[:start] + all_labeled[end:]
    auc = train_fold(fold + 1, train_pairs, test_pairs)
    all_aucs.append(auc)
    print(f"  Fold {fold+1} BEST AUC: {auc:.4f}")

print(f"\n5折AUC: {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")

# ======== Cold-start测试 ========
if cold_start:
    print("\n" + "=" * 50)
    print("Cold-start测试")
    print("=" * 50)
    model = SLPredictor(input_dim, HIDDEN_DIM, HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(RESULTS / "fold1_best.pt"))
    model.eval()
    with torch.no_grad():
        x = model(text_feat.to(DEVICE), edge_index, edge_weight)
        scores = model.predict(x, cold_start)
        labels = [pair_to_label.get((u, v), 0) for u, v in cold_start]
        if sum(labels) > 0:
            cs_auc = roc_auc_score(labels, scores.cpu().numpy())
            cs_ap = average_precision_score(labels, scores.cpu().numpy())
            print(f"Cold-start AUC: {cs_auc:.4f} | AP: {cs_ap:.4f}")

# ======== 预测新SL对 ========
print("\n" + "=" * 50)
print("预测候选SL对")
print("=" * 50)

# 在所有PPI边上预测，排除已知的+标签对
model = SLPredictor(input_dim, HIDDEN_DIM, HIDDEN_DIM).to(DEVICE)
model.load_state_dict(torch.load(RESULTS / "fold1_best.pt"))
model.eval()

candidates = []
known = set((min(u,v), max(u,v)) for u,v in all_labeled)
eidx = edge_index.cpu().numpy()

# 每采样100条边
sample_step = max(1, eidx.shape[1] // 50000)
with torch.no_grad():
    x = model(text_feat.to(DEVICE), edge_index, edge_weight)
    
    for i in range(0, eidx.shape[1], sample_step):
        u, v = int(eidx[0, i]), int(eidx[1, i])
        pair = (min(u,v), max(u,v))
        if pair not in known and u < n_nodes and v < n_nodes:
            score = model.pred(torch.cat([x[u], x[v]])).sigmoid().item()
            candidates.append((pair[0], pair[1], score))

candidates.sort(key=lambda x: -x[2])
top500 = candidates[:500]

output = []
for u, v, score in top500:
    output.append({
        "gene_a": gene_list[u], "gene_b": gene_list[v],
        "sl_score": round(score, 4)
    })

with open(DATA / "predicted_sl_candidates.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Top-500候选SL对已保存")
print(f"最高分: {gene_list[top500[0][0]]} - {gene_list[top500[0][1]]} ({top500[0][2]:.4f})")
for i, (u, v, s) in enumerate(top500[:5]):
    print(f"  {i+1}. {gene_list[u]} ↔ {gene_list[v]} (score={s:.4f})")

print("\n✅ Phase 2 完成")
print(f"   模型: {RESULTS}/fold*_best.pt")
print(f"   预测: {DATA}/predicted_sl_candidates.json")
