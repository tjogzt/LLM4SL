> 📖 [English](#llm-gnn-collaborative-framework-for-synthetic-lethality-prediction) | [简体中文](README_CN.md)

# LLM-GNN Collaborative Framework for Synthetic Lethality Prediction

**Complementary Information Channels for Synthetic Lethality Prediction: Augmenting Graph Topology with Protein Language Models and LLM Weak Supervision**

Tao Zhu — <sup>1</sup> Department of Obstetrics and Gynecology, National Clinical Research Center for Obstetrics and Gynecology, Tongji Hospital, Tongji Medical College, Huazhong University of Science and Technology, Wuhan, China; <sup>2</sup> Key Laboratory of Cancer Invasion and Metastasis (Ministry of Education), Hubei Key Laboratory of Tumor Invasion and Metastasis, Tongji Hospital, Tongji Medical College, Huazhong University of Science and Technology, Wuhan, China

---

## Overview

This repository contains the complete code, data, and trained models for our LLM-GNN collaborative framework for synthetic lethality (SL) prediction. The framework integrates three complementary information channels:

1. **LLM Weak Supervision** — DeepSeek-V4-Pro NER on 1,844 PubMed SL abstracts
2. **GATv2 Graph Neural Network** — Message passing on STRING v12.0 PPI network (15,956 nodes, 932K edges)
3. **ESM-2 Protein Language Model** — Sequence-level protein priors via `esm2_t12_35M`

**Key Results:**
- 5-fold CV AUC: 0.862 (ESM-2) / 0.693 (base)
- Cold-start AUPRC: 0.84 (CV3 gene-level split)
- DepMap CRISPR validation: 1.73× enrichment (Fisher's exact p=0.041)
- Two methodological contributions: SL adjacency leakage quantification (13.7%) + zero-vector ablation bias correction (14×)

---

## Repository Structure

```
SL_LLM_GNN/
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
├── .gitignore
│
├── code/                        ← All analysis scripts
│   ├── config.py                ← Shared paths and constants
│   ├── 01_pubmed_search.py      ← PubMed API search
│   ├── 02_deepseek_ner.py       ← LLM gene entity extraction
│   ├── 03_ppi_network.py        ← STRING PPI download & processing
│   ├── 04_synlethdb.py          ← SynLethDB label processing
│   ├── 05_build_graph.py        ← Graph construction (PPI + NER + labels)
│   ├── 06_train_gnn.py          ← Main GNN training (5-fold CV)
│   ├── 07_ablation.py           ← Ablation experiments (shuffled-feature)
│   ├── 08_cv3_benchmark.py      ← Feng-benchmark CV1/CV3 evaluation
│   └── 09_esm2_train.py         ← ESM-2 enhanced model training
│
├── data/processed/              ← Processed data (included in repo)
│   ├── graph_data.pkl           ← Constructed PyG graph (23 MB)
│   ├── ner_results.jsonl        ← LLM NER output (2,846 genes)
│   ├── predicted_sl_final.json  ← Top-500 SL predictions
│   ├── depmap_full_validation.json ← DepMap validation results
│   ├── cv3_results.json         ← CV1/CV3 benchmark results
│   ├── slmgae_head_to_head.json ← SLMGAE head-to-head comparison
│   ├── ablation_extended.json   ← Complete ablation data
│   └── ppm1d_tp53_lineage.json  ← PPM1D-TP53 lineage stratification
│
├── results/models/              ← Trained model weights
│   ├── fold1_best.pt ~ fold5_best.pt
│   └── esm_best.pt
│
├── figures/                     ← Publication-quality figures
│   ├── Fig1-5.pdf               ← Main text figures
│   └── FigS1-S3.pdf             ← Supplementary figures
│
├── paper/                       ← Compiled submission PDFs
│   ├── main.pdf
│   ├── supplementary.pdf
│   └── cover_letter.pdf
│
└── latex/                       ← LaTeX source files
    ├── main.tex
    ├── supplementary.tex
    └── cover_letter.tex
```

---

## Reproducibility

### Environment Setup

```bash
conda create -n sl_gnn python=3.10
conda activate sl_gnn
pip install -r requirements.txt
```

### Data Preparation

External data must be downloaded from public sources:

| Source | URL | File |
|--------|-----|------|
| STRING v12.0 | https://string-db.org | `string_human_v12.0_links.txt.gz`, `string_human_v12.0_aliases.txt.gz` |
| SynLethDB 3.0 | https://synlethdb.sist.shanghaitech.edu.cn | `Human.SL.detailed.tsv`, `Human.non.SL.detailed.tsv` |
| DepMap 26Q1 | https://depmap.org | `CRISPRGeneEffect.csv` |
| ESM-2 | HuggingFace `facebook/esm2_t12_35M_UR50D` | Auto-downloaded by transformers |

Place external files in `data/external/`. Then run scripts in numerical order:

```bash
# Phase 1: Data Collection
python code/01_pubmed_search.py        # → data/raw/
python code/02_deepseek_ner.py         # → data/processed/ner_results.jsonl
python code/03_ppi_network.py          # → data/processed/ppi_string_edges.tsv
python code/04_synlethdb.py            # → data/processed/sl_labels.json

# Phase 2: Graph Construction
python code/05_build_graph.py          # → data/processed/graph_data.pkl

# Phase 3: Training & Evaluation
python code/06_train_gnn.py            # → results/models/fold*_best.pt
python code/07_ablation.py             # → ablation results
python code/08_cv3_benchmark.py        # → CV1/CV3 results
python code/09_esm2_train.py           # → ESM-2 enhanced model
```

### Hardware Requirements

- Consumer GPU (RTX 4090 or Apple M-series MPS) — 5-fold CV completes in <10 minutes
- ~8 GB GPU memory for ESM-2 model
- ~30 GB disk for all external data

---

## Citation

If you use this code or data, please cite:

```
Zhu T. Complementary Information Channels for Synthetic Lethality Prediction:
Augmenting Graph Topology with Protein Language Models and LLM Weak Supervision.
Briefings in Bioinformatics, 2026.
```

---

## License

MIT License. See repository for details.

## Contact

Tao Zhu — zhutao@tjh.tjmu.edu.cn
Department of Obstetrics and Gynecology, Tongji Hospital, HUST, Wuhan, China
