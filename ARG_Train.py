# ============================== CONFIG (edit only this section) ==============================
# Device / runtime
GPU_INDEX: int = 0              # which GPU to use (0-based). Set to 1 to use the 2nd GPU, etc.
USE_XLA: bool = False           # if True, keep XLA JIT; if False, disable to avoid device routing issues
SEED: int = 42                  # random seed for NumPy / TF

# Files & paths
EMBEDDING_TRAIN_FILENAME: str = "./Embeddings/Graph_Part_ARG_40_Train.h5"
SEQ_TRAIN_PATH: str            = "./Data/Graph_Part_ARG_40_Train.fasta"
MODEL_OUT_PATH: str            = "./Models/Graph_Part_ARG_40_Train.h5"

# Labels (FASTA labels -> index). Update here if your classes change.
LABEL_MAP = {
    "ARG": 0,
    "Non-ARG": 1,
}

# Model / training hyperparameters
LAYER_SIZES = [2500, 1500, 700, 300, 100]   # hidden units per layer
DROPOUT_AFTER_FIRST: bool = True            # apply dropout on layers after the first
DROPOUT_RATE: float = 0.20
OUTPUT_ACTIVATION: str = "sigmoid"          # keep 'sigmoid' for one-vs-all style
LOSS_FN: str = "binary_crossentropy"        # pairs with 'sigmoid' for 1-hot targets
LEARNING_RATE: float = 1e-4
EPOCHS: int = 20
BATCH_SIZE: int = 128
FIT_VERBOSE: int = 1

# Behavior
RAISE_ON_ID_MISMATCH: bool = False          # True -> raise if an H5 ID is missing in FASTA; False -> skip
# =============================================================================================

# --- Device pinning MUST happen before importing TensorFlow ---
import os
os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
if not USE_XLA:
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")  # optional: disable XLA JIT

import h5py
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from Bio import SeqIO

# ----------------------------- Reproducibility -----------------------------
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ----------------------------- Device setup -----------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')  # expose only the first visible GPU in this process
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass  # already initialized

# ----------------------------- Label maps -----------------------------
index_of_resistant_mechanisms = dict(LABEL_MAP)     # alias to keep original naming
NUM_CLASSES = len(index_of_resistant_mechanisms)

AMINO_ACID_VOCABULARY_with_index = {
    'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9,
    'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19
}

# ----------------------------- Helpers -----------------------------
def create_labels_of_protein_seq(args: str):
    """Build a one-hot/multi-hot vector from the last |field in FASTA header."""
    label = [0] * NUM_CLASSES
    for token in args.split(','):
        key = token.strip()
        if key in index_of_resistant_mechanisms:
            label[index_of_resistant_mechanisms[key]] = 1
        else:
            raise ValueError(f"Unknown label '{key}' in FASTA header.")
    return label

def read_fasta(file):
    """Return dict: {seq_id -> multi-hot label} where labels come from last |-separated field."""
    seq_dict = {}
    for sequence in SeqIO.parse(file, "fasta"):
        prot_description = str(sequence.description)
        this_prot_id = prot_description.split('|')[0]
        this_prot_label = prot_description.split('|')[-1]
        seq_dict[this_prot_id] = create_labels_of_protein_seq(this_prot_label)
    return seq_dict

def get_model(input_dim: int):
    """
    Build MLP. Use Keras-native activations to satisfy Keras 3 graph rules.
    `input_dim` is inferred from embeddings (4096 for ProtAlbert, 1024 for ProtBERT/ProtT5).
    """
    inp = Input(shape=(input_dim,), name="per_protein_embedding")
    x = inp
    for i, units in enumerate(LAYER_SIZES):
        x = Dense(units, activation='tanh', name=f"dense_{i+1}")(x)
        if DROPOUT_AFTER_FIRST and i > 0:
            x = Dropout(DROPOUT_RATE, name=f"dropout_{i+1}")(x)
    out = Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION, name="out")(x)
    return Model(inputs=[inp], outputs=out, name="ARG_Classifier")

def train_model(input1, y, model_path=MODEL_OUT_PATH):
    input1 = np.asarray(input1, dtype=np.float32)
    y      = np.asarray(y, dtype=np.float32)

    model = get_model(input_dim=input1.shape[1])
    model.compile(
        loss=LOSS_FN,
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy']
    )
    model.fit([input1], y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=FIT_VERBOSE)
    model.save(model_path)
    model.summary()

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    embedding_filename = EMBEDDING_TRAIN_FILENAME
    seq_path           = SEQ_TRAIN_PATH
    model_out_path     = MODEL_OUT_PATH

    # Read labels from FASTA
    gene_id_label_dict = read_fasta(seq_path)

    # Load embeddings (each HDF5 dataset is one protein vector)
    train_y = []
    train_x_embeddings_per_protein = []

    with h5py.File(embedding_filename, "r") as f:
        keys = list(f.keys())
        if not keys:
            raise RuntimeError(f"No datasets found in {embedding_filename}")
        for k in keys:
            if k not in gene_id_label_dict:
                if RAISE_ON_ID_MISMATCH:
                    raise KeyError(f"ID '{k}' present in embeddings but missing in FASTA labels.")
                else:
                    continue
            emb = f[k][:]
            train_x_embeddings_per_protein.append(emb)
            train_y.append(gene_id_label_dict[k])

    if not train_x_embeddings_per_protein:
        raise RuntimeError("No matching (embedding, label) pairs were found. "
                           "Check that FASTA IDs match HDF5 dataset names.")

    X = np.array(train_x_embeddings_per_protein, dtype=np.float32)
    Y = np.array(train_y, dtype=np.float32)

    # Shuffle (reproducible)
    X, Y = shuffle(X, Y, random_state=SEED)

    # Train on the selected device
    device_str = '/device:GPU:0' if gpus else '/CPU:0'
    try:
        with tf.device(device_str):
            train_model(X, Y, model_path=model_out_path)
    except RuntimeError as e:
        print("Falling back to default device due to:", e)
        train_model(X, Y, model_path=model_out_path)
