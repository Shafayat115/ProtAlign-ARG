# ============================== CONFIG (edit only this section) ==============================
GPU_INDEX: int = 0                         # which GPU to use (0-based). Set to 1 for the second GPU, etc.
USE_XLA: bool = False                      # if True, keep XLA JIT; if False, disable to avoid device routing issues
BATCH_SIZE: int = 64                       # predict() batch size
VERBOSE_PREDICT: int = 1                   # 0 = silent, 1 = progress bar

# Files & paths
EMBEDDING_TEST_FILENAME: str = "./Embeddings/Graph_Part_ARG_40_Test.h5"
SEQ_TEST_PATH: str          = "./Data/Graph_Part_ARG_40_Test.fasta"
MODEL_PATH: str             = "./Models/Graph_Part_ARG_40_Train.h5"   # adjust if your saved name differs

# Behavior
RAISE_ON_ID_MISMATCH: bool  = False        # True -> raise if an H5 ID is missing in FASTA; False -> skip that ID
# ============================================================================================


# --- Device pinning MUST happen before importing TensorFlow ---
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
if not USE_XLA:
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")  # optional: disable XLA JIT

import h5py
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from Bio import SeqIO
from sklearn.metrics import classification_report, confusion_matrix

# ===== Device setup (use the single visible GPU 0 in this process, if present) =====
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')  # expose only the first visible GPU in this process
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass  # already initialized

# ===== Label map =====
index_of_args = {
    'ARG': 0,
    'Non-ARG': 1
}

AMINO_ACID_VOCABULARY_with_index = {
    'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9,
    'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19
}

def create_labels_of_protein_seq(args):
    label = [0] * 2
    for token in args.split(','):
        key = token.strip()
        if key not in index_of_args:
            raise ValueError(f"Unknown label '{key}' in FASTA header")
        label[index_of_args[key]] = 1
    return label

def read_fasta(file):
    seq_dict = {}
    for sequence in SeqIO.parse(file, "fasta"):
        desc = str(sequence.description)
        prot_id = desc.split('|')[0]
        label_str = desc.split('|')[-1]
        seq_dict[prot_id] = create_labels_of_protein_seq(label_str)
    return seq_dict

# ===== Paths =====
embedding_test_filename = EMBEDDING_TEST_FILENAME
seq_test_path          = SEQ_TEST_PATH
model_path             = MODEL_PATH

# ===== Load labels =====
gene_id_label_dict_test = read_fasta(seq_test_path)

# ===== Load embeddings (and align with labels) =====
test_x_embeddings_per_protein, test_y, kept_ids = [], [], []
with h5py.File(embedding_test_filename, "r") as f:
    keys = list(f.keys())
    if not keys:
        raise RuntimeError(f"No datasets found in {embedding_test_filename}")
    for k in keys:
        if k not in gene_id_label_dict_test:
            if RAISE_ON_ID_MISMATCH:
                raise KeyError(f"Embedding id '{k}' missing in FASTA labels")
            else:
                continue
        test_x_embeddings_per_protein.append(f[k][:])
        test_y.append(gene_id_label_dict_test[k])
        kept_ids.append(k)

if not test_x_embeddings_per_protein:
    raise RuntimeError("No (embedding, label) pairs found. Check that H5 keys match FASTA IDs.")

X_test = np.asarray(test_x_embeddings_per_protein, dtype=np.float32)
Y_test = np.asarray(test_y, dtype=np.float32)
y_true = np.argmax(Y_test, axis=1)

# ===== Load model & predict on the SAME device =====
device_str = '/device:GPU:0' if gpus else '/CPU:0'
with tf.device(device_str):  # must match creation device
    model = load_model(model_path, compile=False)  # metrics not needed for predict
    y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=VERBOSE_PREDICT)

y_pred = np.argmax(y_pred_probs, axis=1)

# ===== Report =====
print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=4))
