"""Create a fixed train / val / test edge split shared by every model.

Positive edges are split by fraction. For each positive edge we sample one
negative (non-edge) pair, so every split is balanced. The negatives are
disjoint from all positive edges and from each other. Message passing at
train time uses only the training positive edges.
"""
import numpy as np

from config import GRAPH_FILE, SPLIT_FILE, SEED, VAL_FRAC, TEST_FRAC


def undirected_edge_set(edge_index: np.ndarray) -> set:
    a = np.minimum(edge_index[0], edge_index[1])
    b = np.maximum(edge_index[0], edge_index[1])
    return set(zip(a.tolist(), b.tolist()))


def sample_negatives(n_nodes: int, forbidden: set, k: int, rng) -> np.ndarray:
    out = set()
    while len(out) < k:
        u = rng.integers(0, n_nodes, size=2 * k)
        v = rng.integers(0, n_nodes, size=2 * k)
        for a, b in zip(u.tolist(), v.tolist()):
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in forbidden or key in out:
                continue
            out.add(key)
            if len(out) == k:
                break
    return np.array(sorted(out), dtype=np.int64).T


def main() -> None:
    d = np.load(GRAPH_FILE)
    edge_index, n_nodes = d["edge_index"], int(d["num_nodes"])
    rng = np.random.default_rng(SEED)

    pos = np.array(sorted(undirected_edge_set(edge_index)), dtype=np.int64)
    rng.shuffle(pos)
    n_pos = len(pos)
    n_test = int(n_pos * TEST_FRAC)
    n_val = int(n_pos * VAL_FRAC)

    test_pos = pos[:n_test]
    val_pos = pos[n_test:n_test + n_val]
    train_pos = pos[n_test + n_val:]
    print(f"positives  train {len(train_pos):,}  val {len(val_pos):,}  test {len(test_pos):,}")

    forbidden = set(map(tuple, pos.tolist()))
    neg = sample_negatives(n_nodes, forbidden, n_pos, rng).T  # (n_pos, 2)
    test_neg = neg[:n_test]
    val_neg = neg[n_test:n_test + n_val]
    train_neg = neg[n_test + n_val:]

    # message-passing graph = training positives, both directions
    mp = np.concatenate([train_pos.T, train_pos.T[::-1]], axis=1)

    np.savez_compressed(
        SPLIT_FILE,
        mp_edge_index=mp,
        train_pos=train_pos.T, train_neg=train_neg.T,
        val_pos=val_pos.T, val_neg=val_neg.T,
        test_pos=test_pos.T, test_neg=test_neg.T,
    )
    print(f"Saved {SPLIT_FILE.name}")


if __name__ == "__main__":
    main()
