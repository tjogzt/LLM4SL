> 📖 [English](README.md) | [简体中文](README_CN.md)

# LLM4SL：大语言模型与图神经网络协同驱动的合成致死预测框架

**互补信息通道增强合成致死预测：基于蛋白质语言模型与LLM弱监督的图拓扑增强方法**

朱涛 — 华中科技大学同济医学院附属同济医院妇产科

---

## 概述

本仓库包含论文的全部代码、数据和已训练的模型。框架整合了三条互补信息通道：

1. **LLM弱监督** — DeepSeek-V4-Pro 对 1,844 篇 PubMed 合成致死文献进行命名实体识别
2. **GATv2图神经网络** — 在 STRING v12.0 PPI 网络（15,956 节点 / 93 万边）上进行消息传递
3. **ESM-2蛋白质语言模型** — 提供氨基酸序列层面的蛋白质先验知识

### 关键结果

| 指标 | 数值 |
|------|------|
| 5折交叉验证 AUC | 0.862 (ESM-2增强) / 0.693 (基础版) |
| 冷启动 AUPRC (CV3) | 0.84 (基因级拆分) |
| DepMap CRISPR 独立验证 | 1.73× 富集 (Fisher精确检验 p=0.041) |
| SLMGAE 公平对比 | CV3 AUC=0.770（公平屏蔽）vs 0.907（标准评估） |
| PPM1D-TP53 | 泛癌竞争性依赖 r=−0.58 (七种癌种) |

### 方法论贡献

1. **多视图GAE冷启动的信息泄露**：发现SL邻接矩阵在标准评估中导致AUC虚高 13.7%
2. **零向量消融偏差**：批归一化GNN中零向量消融夸大特征重要性 14 倍

---

## 仓库结构

```
LLM4SL/
├── README.md                    ← 英文说明（本文件）
├── README_CN.md                 ← 中文说明
├── requirements.txt             ← Python 依赖
├── .gitignore
│
├── code/                        ← 分析脚本（按执行顺序编号）
│   ├── config.py                ← 共享路径与常量
│   ├── 01_pubmed_search.py      ← PubMed API 检索
│   ├── 02_deepseek_ner.py       ← LLM 基因实体提取
│   ├── 03_ppi_network.py        ← STRING PPI 下载与处理
│   ├── 04_synlethdb.py          ← SynLethDB 标签处理
│   ├── 05_build_graph.py        ← 图构建（PPI + NER + 标签）
│   ├── 06_train_gnn.py          ← GNN 主训练（5折CV）
│   ├── 07_ablation.py           ← 消融实验（shuffled-feature）
│   ├── 08_cv3_benchmark.py      ← Feng 基准 CV1/CV3 评估
│   └── 09_esm2_train.py         ← ESM-2 增强模型训练
│
├── data/processed/              ← 处理后数据（已纳入仓库）
│   ├── graph_data.pkl           ← 构建好的图数据（23 MB）
│   ├── ner_results.jsonl        ← LLM NER 输出（2,846 基因）
│   ├── predicted_sl_final.json  ← Top-500 SL 预测结果
│   └── ...
│
├── results/models/              ← 训练好的模型权重
├── figures/                     ← 论文图表（PDF）
├── paper/                       ← 投稿 PDF
└── latex/                       ← LaTeX 源文件
```

---

## 复现指南

### 环境配置

```bash
conda create -n llm4sl python=3.10
conda activate llm4sl
pip install -r requirements.txt
```

### 外部数据下载

以下文件需从公开源下载，放入 `data/external/` 目录：

| 数据源 | URL | 文件 |
|--------|-----|------|
| STRING v12.0 | https://string-db.org | `string_human_v12.0_links.txt.gz`, `string_human_v12.0_aliases.txt.gz` |
| SynLethDB 3.0 | https://synlethdb.sist.shanghaitech.edu.cn | `Human.SL.detailed.tsv`, `Human.non.SL.detailed.tsv` |
| DepMap 26Q1 | https://depmap.org | `CRISPRGeneEffect.csv` |
| ESM-2 | HuggingFace `facebook/esm2_t12_35M_UR50D` | 由 transformers 自动下载 |

### 执行顺序

```bash
# 第一阶段：数据采集
python code/01_pubmed_search.py
python code/02_deepseek_ner.py
python code/03_ppi_network.py
python code/04_synlethdb.py

# 第二阶段：图构建
python code/05_build_graph.py

# 第三阶段：训练与评估
python code/06_train_gnn.py       # 基础GNN训练
python code/07_ablation.py        # 消融实验
python code/08_cv3_benchmark.py   # Feng基准对齐
python code/09_esm2_train.py      # ESM-2增强模型
```

### 硬件要求

- 消费级 GPU（RTX 4090 或 Apple M系列 MPS）— 5折CV < 10分钟
- ~8 GB GPU 显存用于 ESM-2
- ~30 GB 磁盘用于外部数据

---

## 论文引用

```bibtex
@article{zhu2026llm4sl,
  title={Complementary Information Channels for Synthetic Lethality Prediction:
         Augmenting Graph Topology with Protein Language Models and LLM Weak Supervision},
  author={Zhu, Tao},
  journal={Briefings in Bioinformatics},
  year={2026}
}
```

---

## 许可证

MIT License

## 联系方式

朱涛 — zhutao@tjh.tjmu.edu.cn
华中科技大学同济医学院附属同济医院妇产科
