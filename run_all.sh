#!/usr/bin/env bash
# End-to-end PPI-GNN pipeline. Run from the repo root.
set -e
cd "$(dirname "$0")/src"

python 01_download_data.py        # STRING v12.0 for H. sapiens
python 02_build_graph.py          # filter, LCC, structural + Laplacian-PE features
python 03_make_splits.py          # fixed train/val/test edge split (balanced)
python 04_baselines.py            # topological link-prediction baselines
python 05_train_gnn.py            # GraphSAGE encoder + MLP decoder
python 07_hybrid_model.py         # GNN + topology -> gradient boosting / logreg, ablation
python 06_compare_results.py      # merged comparison figure + table
python 08_predict_novel.py        # rank novel interactions with the hybrid model

echo
echo "All outputs written to results/"
