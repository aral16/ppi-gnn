"""Shared configuration for the PPI-GNN pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# --- STRING dataset -----------------------------------------------------------
SPECIES_ID = 9606          # Homo sapiens
STRING_VERSION = "12.0"
STRING_BASE = "https://stringdb-downloads.org/download"

LINKS_FILE = DATA_DIR / f"{SPECIES_ID}.protein.links.v{STRING_VERSION}.txt.gz"
INFO_FILE = DATA_DIR / f"{SPECIES_ID}.protein.info.v{STRING_VERSION}.txt.gz"

LINKS_URL = f"{STRING_BASE}/protein.links.v{STRING_VERSION}/{LINKS_FILE.name}"
INFO_URL = f"{STRING_BASE}/protein.info.v{STRING_VERSION}/{INFO_FILE.name}"

# Keep only high-confidence interactions (STRING combined score is 0-1000).
CONFIDENCE_THRESHOLD = 700

# --- Processed graph artifacts ----------------------------------------------
GRAPH_FILE = DATA_DIR / "graph.npz"          # edge_index + node features
SPLIT_FILE = DATA_DIR / "splits.npz"         # train/val/test pos & neg edges
NODE_MAP_FILE = DATA_DIR / "node_map.csv"    # idx -> STRING id -> gene symbol

# --- Split / training --------------------------------------------------------
SEED = 42
VAL_FRAC = 0.05
TEST_FRAC = 0.10

EMB_DIM = 128
HIDDEN_DIM = 256
OUT_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3
LR = 3e-3
WEIGHT_DECAY = 1e-6
MAX_EPOCHS = 400
PATIENCE = 40
