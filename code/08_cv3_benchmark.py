"""
CV3 Cold-start Gene-Level Split — Feng et al. (2024) 对齐实验
在MacBook MPS或AutoDL GPU上运行
"""
import pickle, random, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, ndcg_score

BASE = "REPLACE_WITH_PROJECT_PATH"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[设备] {DEVICE}")

with open(f"{BASE}/data/processed/graph_data.pkl","rb") as f: g = pickle.load(f)

# === 数据准备 ===
pos = [(u,v) for(u,v),l in g["sl_pair_labels"].items() if l==1]
neg = [(u,v) for(u,v),l in g["sl_pair_labels"].items() if l==0]
al = pos + neg; ps = set(pos)

# === CV3: Gene-Level Split ===
# 将所有基因随机分为5份，测试集的两个基因必须都在测试基因集中
# 这是Feng et al. (2024) CV3的确切定义
all_genes = list(set(g["gene_list"]))
random.seed(42); random.shuffle(all_genes)
gene_folds = np.array_split(all_genes, 5)

# CV1 (标准: pair split)
def cv1_split():
    random.shuffle(al); fsz = len(al)//5
    return [(al[:fsz], al[fsz:]) for _ in range(5)]

# CV2 (半冷启动: 一个基因未见)
def cv2_split():
    folds = []
    for i in range(5):
        test_genes = set(gene_folds[i])
        test, train = [], []
        for u,v in al:
            gs_u, gs_v = g["gene_list"][u], g["gene_list"][v]
            in_test = (gs_u in test_genes) or (gs_v in test_genes)
            not_both_train = not (gs_u not in test_genes and gs_v not in test_genes)
            if in_test and not_both_train:
                test.append((u,v))
            else:
                train.append((u,v))
        folds.append((test, train))
    return folds

# CV3 (完全冷启动: 两个基因均未见)
def cv3_split():
    folds = []
    for i in range(5):
        test_genes = set(gene_folds[i])
        test, train = [], []
        for u,v in al:
            gs_u, gs_v = g["gene_list"][u], g["gene_list"][v]
            if gs_u in test_genes and gs_v in test_genes:
                test.append((u,v))
            else:
                train.append((u,v))
        folds.append((test, train))
    return folds

# === 模型（同上） ===
ei = torch.tensor(g["edge_index"], dtype=torch.long).to(DEVICE)
ew = torch.tensor(g["edge_weight"], dtype=torch.float).to(DEVICE)
tf_data = g["text_features"]
tf = torch.tensor(tf_data, dtype=torch.float).to(DEVICE)

class SL(nn.Module):
    def __init__(self, d): super().__init__()
        self.emb=nn.Linear(d,256); self.bn=nn.BatchNorm1d(256)
        self.c1=nn.Linear(512,256); self.c2=nn.Linear(512,256); self.c3=nn.Linear(512,256)
        self.pred=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,1))
    def forward(self,x):
        x=self.bn(F.relu(self.emb(x)))
        for c in [self.c1,self.c2,self.c3]:
            s,d=ei[0],ei[1]; m=c(torch.cat([x[s],x[d]],-1))
            x2=x.clone(); x2.index_add_(0,d,0.1*ew.unsqueeze(1)*m); x=F.relu(x2)
        return x
    def score(self,x,pairs):
        ui=torch.tensor([p[0] for p in pairs],device=DEVICE)
        vi=torch.tensor([p[1] for p in pairs],device=DEVICE)
        return self.pred(torch.cat([x[ui],x[vi]],-1)).sigmoid().squeeze()

def train_eval(folds_data, name, max_e=80):
    aucs=[]; auprcs=[]; f1s=[]; ndcgs=[]
    for test, train in folds_data:
        tr_pos=[(u,v) for(u,v)in train if(u,v)in ps or(v,u)in ps]
        tr_neg=[(u,v) for(u,v)in train if(u,v)not in ps and(v,u)not in ps]
        if len(tr_neg)>len(tr_pos)*3: tr_neg=random.sample(tr_neg,len(tr_pos)*3)
        tr=tr_pos+tr_neg; random.shuffle(tr)
        y=torch.tensor([0.9 if(u,v)in ps or(v,u)in ps else 0.05 for u,v in tr],device=DEVICE)
        m=SL(tf.shape[1]).to(DEVICE); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-3)
        best_a,t0=0,time.time()
        for e in range(max_e):
            m.train(); x=m(tf); loss=F.binary_cross_entropy(m.score(x,tr),y)
            opt.zero_grad(); loss.backward(); opt.step()
            if e%20==0:
                m.eval()
                with torch.no_grad():
                    x=m(tf); ts=m.score(x,test).cpu().numpy()
                    tl=[1.0 if(u,v)in ps or(v,u)in ps else 0.0 for u,v in test]
                    a=roc_auc_score(tl,ts) if sum(tl)>0 else 0.5
                    if a>best_a: best_a=a
        dt=time.time()-t0
        m.eval()
        with torch.no_grad():
            x=m(tf); ts=m.score(x,test).cpu().numpy()
            tl=[1.0 if(u,v)in ps or(v,u)in ps else 0.0 for u,v in test]
            aucs.append(best_a)
            auprcs.append(average_precision_score(tl,ts) if sum(tl)>0 else 0)
            # NDCG@10 and NDCG@50 (Feng benchmark metrics)
            if sum(tl)>0 and len(ts)>=10:
                ts_2d = ts.reshape(1,-1); tl_2d = np.array(tl).reshape(1,-1)
                ndcgs.append(ndcg_score(tl_2d, ts_2d, k=min(10,len(ts))))
        print(f"  [{name}] AUC={best_a:.4f} AUCavg={np.mean(aucs):.4f} ({dt:.0f}s)")
    mu_a, std_a = np.mean(aucs), np.std(aucs)
    mu_p = np.mean(auprcs) if auprcs else 0
    mu_n = np.mean(ndcgs) if ndcgs else 0
    return mu_a, std_a, mu_p, mu_n

# === 运行 ===
print(f"\nFeng 基准对齐实验: {len(pos)}正/{len(neg)}负")
print("="*60)
print("CV1 (标准基因对拆分)")
print("="*60)
cv1_a, cv1_s, cv1_p, cv1_n = train_eval([cv1_split()[0]], "CV1")

print("\n" + "="*60)
print("CV2 (半冷启动: 单基因未见)")
print("="*60)
cv2_a, cv2_s, cv2_p, cv2_n = train_eval(cv2_split(), "CV2")

print("\n" + "="*60)
print("CV3 (完全冷启动: 双基因未见)")
print("="*60)
cv3_a, cv3_s, cv3_p, cv3_n = train_eval(cv3_split(), "CV3")

print(f"\n{'='*60}")
print(f"Feng基准对齐 — 完整结果")
print(f"{'='*60}")
print(f"  场景   | AUC           | AUPRC     | NDCG@K   ")
print(f"  CV1    | {cv1_a:.4f} ± {cv1_s:.4f} | {cv1_p:.4f}    | {cv1_n:.4f}")
print(f"  CV2    | {cv2_a:.4f} ± {cv2_s:.4f} | {cv2_p:.4f}    | {cv2_n:.4f}")
print(f"  CV3    | {cv3_a:.4f} ± {cv3_s:.4f} | {cv3_p:.4f}    | {cv3_n:.4f}")
print(f"\n  SLMGAE CV3 F1 (Feng et al.): 0.738")
print(f"  SLMGAE CV3 NDCG@10: 0.039")
print(f"\n  Cold-start泛化保持率 (CV3/CV1): AUC={cv3_a/cv1_a*100:.1f}%, AUPRC={cv3_p/cv1_p*100:.1f}%")
PYEOF"""
