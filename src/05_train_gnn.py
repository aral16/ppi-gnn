"""Train the GraphSAGE link-prediction model and evaluate on the held-out test edges."""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from config import (
    GRAPH_FILE, SPLIT_FILE, RESULTS_DIR, SEED,
    HIDDEN_DIM, OUT_DIM, NUM_LAYERS, DROPOUT,
    LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE,
)
from model import GNNEncoder, LinkPredictor

CONV = "sage"
EVAL_EVERY = 5


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = get_device()
    print(f"device: {dev}")

    g = np.load(GRAPH_FILE)
    s = np.load(SPLIT_FILE)
    num_nodes = int(g["num_nodes"])
    struct_dim = int(g["struct_dim"])
    x = torch.tensor(g["features"], dtype=torch.float32, device=dev)
    mp_edge_index = torch.tensor(s["mp_edge_index"], dtype=torch.long, device=dev)

    def pairs(key):
        return torch.tensor(s[key], dtype=torch.long, device=dev)

    train_pos = pairs("train_pos")
    val_pos, val_neg = pairs("val_pos"), pairs("val_neg")
    test_pos, test_neg = pairs("test_pos"), pairs("test_neg")
    n_train_pos = train_pos.shape[1]

    encoder = GNNEncoder(x.shape[1], HIDDEN_DIM, OUT_DIM, NUM_LAYERS, DROPOUT, conv=CONV).to(dev)
    predictor = LinkPredictor(OUT_DIM, hidden=128, dropout=DROPOUT).to(dev)
    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"parameters: {sum(p.numel() for p in params):,}")

    def augment(feats):
        """Random sign flips on the Laplacian-PE columns (eigenvector sign is arbitrary)."""
        flip = (torch.randint(0, 2, (feats.shape[1] - struct_dim,), device=dev) * 2 - 1).float()
        out = feats.clone()
        out[:, struct_dim:] = out[:, struct_dim:] * flip
        return out

    @torch.no_grad()
    def evaluate(pos, neg):
        encoder.eval(); predictor.eval()
        z = encoder(x, mp_edge_index)
        scores = torch.cat([predictor(z, pos), predictor(z, neg)]).sigmoid().cpu().numpy()
        y = np.concatenate([np.ones(pos.shape[1]), np.zeros(neg.shape[1])])
        return roc_auc_score(y, scores), average_precision_score(y, scores), y, scores

    history = {"epoch": [], "loss": [], "val_auc": [], "val_ap": []}
    best_auc, best_state, bad = 0.0, None, 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        encoder.train(); predictor.train()
        opt.zero_grad()
        z = encoder(augment(x), mp_edge_index)

        rnd = torch.randint(0, num_nodes, (2, n_train_pos), device=dev)
        pos_logit = predictor(z, train_pos)
        neg_logit = predictor(z, rnd)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_logit, neg_logit]),
            torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)]),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if epoch == 1 or epoch % EVAL_EVERY == 0:
            val_auc, val_ap, *_ = evaluate(val_pos, val_neg)
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["val_auc"].append(val_auc)
            history["val_ap"].append(val_ap)
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  "
                  f"val ROC-AUC {val_auc:.4f}  val AP {val_ap:.4f}")
            if val_auc > best_auc:
                best_auc, bad = val_auc, 0
                best_state = (
                    {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                    {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()},
                )
            else:
                bad += 1
                if bad * EVAL_EVERY >= PATIENCE:
                    print(f"early stop at epoch {epoch}")
                    break

    encoder.load_state_dict(best_state[0])
    predictor.load_state_dict(best_state[1])
    print(f"trained in {time.time() - t0:.1f}s")

    test_auc, test_ap, y, scores = evaluate(test_pos, test_neg)
    val_auc, val_ap, *_ = evaluate(val_pos, val_neg)
    print(f"\nTEST  ROC-AUC {test_auc:.4f}   AP {test_ap:.4f}")

    metrics = {
        "model": f"GraphSAGE ({NUM_LAYERS} layers, out_dim={OUT_DIM}) + MLP decoder",
        "features": f"{struct_dim} structural + {x.shape[1] - struct_dim} Laplacian-PE",
        "num_nodes": num_nodes,
        "train_pos_edges": int(n_train_pos),
        "val": {"roc_auc": round(val_auc, 4), "ap": round(val_ap, 4)},
        "test": {"roc_auc": round(test_auc, 4), "ap": round(test_ap, 4)},
        "best_val_auc": round(best_auc, 4),
    }
    (RESULTS_DIR / "gnn_metrics.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS_DIR / "training_history.json").write_text(json.dumps(history, indent=2))

    with torch.no_grad():
        z = encoder(x, mp_edge_index).cpu().numpy()
    np.save(RESULTS_DIR / "embeddings.npy", z)

    _plot_training(history)
    _plot_curves(y, scores, test_auc, test_ap)
    print("Saved gnn_metrics.json, training_curve.png, roc_pr_curves.png, embeddings.npy")


def _plot_training(h):
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(h["epoch"], h["loss"], color="tab:red", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(h["epoch"], h["val_auc"], color="tab:blue", label="val ROC-AUC")
    ax2.plot(h["epoch"], h["val_ap"], color="tab:green", label="val AP")
    ax2.set_ylabel("validation metric")
    fig.legend(loc="lower right", bbox_to_anchor=(0.9, 0.15))
    ax1.set_title("Training dynamics")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curve.png", dpi=150)
    plt.close(fig)


def _plot_curves(y, scores, auc, ap):
    fpr, tpr, _ = roc_curve(y, scores)
    prec, rec, _ = precision_recall_curve(y, scores)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, lw=2, label=f"GraphSAGE (AUC = {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC — test edges"); axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec, lw=2, color="tab:green", label=f"GraphSAGE (AP = {ap:.3f})")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-Recall — test edges"); axes[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
