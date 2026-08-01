"""消融实验 - 快速版 (1折50epoch, CPU)"""
import pickle, random, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

BASE = "/Users/taozhu/my researches/LLM_SL/sl_llm_gnn"
EPOCHS = 50

with open(f"{BASE}/data/processed/graph_data.pkl","rb") as f: g = pickle.load(f)
ei=torch.tensor(g["edge_index"],dtype=torch.long)
ew=torch.tensor(g["edge_weight"],dtype=torch.float)
tf_=torch.tensor(g["text_features"],dtype=torch.float)
zs=torch.zeros_like(tf_)
pl=g["sl_pair_labels"]
pos=[(u,v) for(u,v),l in pl.items() if l==1]
neg=[(u,v) for(u,v),l in pl.items() if l==0]
al=pos+neg; ps=set(pos); random.seed(42); random.shuffle(al)
fsz=len(al)//5; tp=al[:fsz]; tr=al[fsz:]

class SL(nn.Module):
    def __init__(self, in_dim, use_gnn=True): 
        super().__init__()
        self.use_gnn = use_gnn
        self.emb = nn.Linear(in_dim, 128)
        self.conv1 = nn.Linear(256, 128)
        self.conv2 = nn.Linear(256, 128)
        self.pred = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,1))
    def forward(self, x):
        x = F.relu(self.emb(x))
        if self.use_gnn:
            for c in [self.conv1, self.conv2]:
                s, d = ei[0], ei[1]
                m = c(torch.cat([x[s], x[d]], -1))
                x2 = x.clone()
                x2.index_add_(0, d, 0.1 * ew.unsqueeze(1) * m)
                x = F.relu(x2)
        return x
    def score(self, x, pairs):
        ui = torch.tensor([p[0] for p in pairs])
        vi = torch.tensor([p[1] for p in pairs])
        return self.pred(torch.cat([x[ui], x[vi]], -1)).sigmoid().squeeze()

def train_one(name, feat, use_gnn=True):
    y = torch.tensor([1.0 if(u,v) in ps or(v,u) in ps else 0.0 for u,v in tr])
    m = SL(feat.shape[1], use_gnn=use_gnn)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    best, t0 = 0, time.time()
    for e in range(EPOCHS):
        m.train(); x=m(feat); loss=F.binary_cross_entropy(m.score(x,tr),y)
        opt.zero_grad(); loss.backward(); opt.step()
        if e%10==0:
            m.eval()
            with torch.no_grad():
                x=m(feat); ts=m.score(x,tp).cpu().numpy()
                tl=[1.0 if(u,v) in ps or(v,u) in ps else 0.0 for u,v in tp]
                a=roc_auc_score(tl,ts) if sum(tl)>0 else 0.5
                if a>best: best=a
    dt=time.time()-t0
    print(f"  [{name}] AUC={best:.4f} ({dt:.0f}s)")
    return best

print(f"正:{len(pos)} 负:{len(neg)} | 训练:{len(tr)} 测试:{len(tp)} | {EPOCHS}epochs\n")

full = train_one("完整模型(文本+PPI)", tf_, use_gnn=True)
no_text = train_one("Abl-1: 去掉文本(仅PPI)", zs, use_gnn=True)
no_gnn = train_one("Abl-2: 去掉PPI(仅文本)", tf_, use_gnn=False)

print(f"\n{'='*50}")
print(f"消融结果 (1-fold/{EPOCHS}epochs/CPU)")
print(f"{'='*50}")
print(f"{'完整模型':<20} AUC={full:.4f}")
print(f"{'去文本信号':<20} AUC={no_text:.4f} (Δ={no_text-full:+.4f})")
print(f"{'去PPI拓扑':<20} AUC={no_gnn:.4f} (Δ={no_gnn-full:+.4f})")
print(f"\n▶ 文本信号贡献: +{full-no_text:.4f} AUC")
print(f"▶ PPI拓扑贡献: +{full-no_gnn:.4f} AUC")
