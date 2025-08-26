# ============================== CONFIG (edit only this section) ==============================
# Device / runtime
GPU_INDEX: int = 0                     # which GPU to use (0-based). Set to 1 to use the 2nd GPU, etc.
USE_XLA: bool = False                  # if True, keeps XLA JIT on; if False, disables to avoid device issues

# Files & paths
EMBEDDING_TEST_FILENAME: str = "./Embeddings/ARG_Class_Beta_Test.h5"
SEQ_TEST_PATH: str          = "./Data/ARG_Class_Beta_Test.fasta"
MODEL_PATH: str             = "./Models/ARG_Class_Beta_Train.h5"
TRAIN_FASTA: str            = "Data/ARG_Class_Beta_Train.fasta"   # DIAMOND DB source FASTA
DIAMOND_DB_NAME: str        = "Train"                                 # name for .dmnd DB
DIAMOND_BIN: str            = "diamond"                               # full path if needed, e.g., "/usr/local/bin/diamond"

# Outputs (a single prefix controls all output filenames)
OUTPUT_PREFIX: str          = "low_confident_90_all"                  # e.g., "low_confident_90_all"
# Derived output names (don’t edit below unless you want different patterns)
OUT_ID_TXT: str             = f"{OUTPUT_PREFIX}.txt"                  # "low_confident_90_all.txt"
OUT_FASTA: str              = f"{OUTPUT_PREFIX}.fasta"                # "low_confident_90_all.fasta"
DIAMOND_OUT: str            = f"{OUTPUT_PREFIX}_diamond.txt"          # "low_confident_90_all_diamond.txt"
DIAMOND_OUT_FINAL: str      = f"{OUTPUT_PREFIX}_diamond_final.txt"    # "low_confident_90_all_diamond_final.txt"

# Inference / filtering
BATCH_SIZE: int             = 64
LOW_CONF_THRESH: float      = 0.90     # write IDs whose predicted probability < this
SKIP_DIAMOND_IF_EMPTY: bool = True     # if no low-confidence sequences, skip DIAMOND (recommended)

# Behavior
RAISE_ON_ID_MISMATCH: bool  = False    # True -> raise if an H5 ID is missing in FASTA; False -> skip it
# ============================================================================================

# --- Device pinning MUST happen before importing TensorFlow ---
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
if not USE_XLA:
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")  # optional: disable XLA JIT

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Masking
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten, concatenate
from tensorflow.keras.models import Sequential, load_model, save_model, Model
from tensorflow.keras.layers import BatchNormalization, Dropout
from Bio import SeqIO
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from tensorflow import keras

# ensure TF only uses the pinned GPU and grows memory as needed
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')   # expose only the first visible GPU in this proc
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass

# ------------------------ Label map ------------------------
index_of_args = {
    'fosmidomycin': 0,
    'isoniazid': 1,
    'fosfomycin': 2,
    'sulfonamide': 3
    #     'ARG': 0,
    # 'Non-ARG': 1
        # 'aminoglycoside' : 0,
        # 'bacitracin' : 1,
        # 'beta_lactam' : 2,
        # 'chloramphenicol' : 3,
        # 'fosfomycin' : 4,
        # 'glycopeptide' : 5,
        # 'multidrug' : 6,
        # 'polymyxin' : 7,
        # 'quinolone' : 8,
        # 'rifampin' : 9,
        # 'sulfonamide' : 10,
        # 'tetracycline' : 11,
        # 'macrolide-lincosamide-streptogramin' : 12,
        # 'trimethoprim' : 13
    #     'ARG' : 0,
    #    'Non-ARG':1
    #    'VF' : 0,
    #    'Non-VF':1
        # 'FOLATE-SYNTHESIS-INHABITOR' : 0,
        # 'TETRACYCLINE' : 1,
        # 'MULTIDRUG' : 2,
        # 'BETA-LACTAM' : 3,
        # 'STREPTOGRAMIN' : 4,
        # 'AMINOGLYCOSIDE' : 5,
        # 'FOSFOMYCIN' : 6,
        # 'TRIMETHOPRIM' : 7,
        # 'MACROLIDE' : 8,
        # 'MACROLIDE/LINCOSAMIDE/STREPTOGRAMIN' : 9,
        # 'RIFAMYCIN' : 10,
        # 'SULFONAMIDE' : 11,
        # 'GLYCOPEPTIDE' : 12,
        # 'QUINOLONE' : 13, 
        # 'BACITRACIN' : 14,
        # 'PHENICOL' : 15
    #  'mupirocin' : 0, 
    # 'elfamycin' : 1, 
    # 'tetracenomycin' : 2, 
    # 'tunicamycin' : 3, 
    # 'puromycin': 4, 
    # 'streptothricin': 5, 
    # 'pleuromutilin': 6, 
    # 'nitroimidazole': 7, 
    # 'qa_compound': 8, 
    # 'triclosan': 9, 
    # 'peptide': 10, 
    # 'fosmidomycin': 11, 
    # 'isoniazid': 12, 
    # 'kasugamycin': 13, 
    # 'aminocoumarin': 14, 
    # 'bleomycin': 15, 
    # 'thiostrepton': 16, 
    # 'ethambutol': 17, 
    # 'fusidic_acid': 18, 
    # 'chloramphenicol': 19, 
    # 'rifampin': 20, 
    # 'bacitracin': 21, 
    # 'polymyxin': 22, 
    # 'aminoglycoside': 23, 
    # 'trimethoprim': 24, 
    # 'sulfonamide': 25, 
    # 'quinolone': 26, 
    # 'fosfomycin': 27, 
    # 'tetracycline': 28, 
    # 'macrolide-lincosamide-streptogramin': 29,
    # 'glycopeptide' : 30,
    # 'multidrug': 31,
    # 'beta_lactam': 32
}
NUM_CLASSES = len(index_of_args)
reverse_of_args = {v: k for k, v in index_of_args.items()}

AMINO_ACID_VOCABULARY_with_index = {
    'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14,'S':15, 'T':16, 'V':17, 'W':18, 'Y':19
}

def create_lablels_of_protein_seq(args):
    label = [0] * NUM_CLASSES
    arg_list = args.split(',')
    for i in arg_list:
        key = i.strip()
        if key not in index_of_args:
            raise ValueError(f"Unknown label '{key}' in FASTA header")
        label[index_of_args[key]] = 1
    return label

def read_fasta(file):
    # dictionary contains seq_id as keys and labels as values
    seq_dict = {}
    for sequence in SeqIO.parse(file, "fasta"):
        prot_description = str(sequence.description)
        this_prot_id = prot_description.split('|')[0]
        this_prot_label = prot_description.split('|')[-1]
        seq_dict[this_prot_id] = create_lablels_of_protein_seq(this_prot_label)
    return seq_dict

# ------------------------ Paths ------------------------
embedding_test_filename = EMBEDDING_TEST_FILENAME
seq_test_path = SEQ_TEST_PATH
model_path = MODEL_PATH

# ------------------------ Load labels ------------------
gene_id_label_dict_test = read_fasta(seq_test_path)

# ------------------------ Load embeddings & align -------
test_x_embeddings_per_protein = []
test_y = []
test_id = []
with h5py.File(embedding_test_filename, "r") as f:
    L = list(f.keys())
    if not L:
        raise RuntimeError(f"No datasets found in {embedding_test_filename}")
    for i in L:
        if i not in gene_id_label_dict_test:
            if RAISE_ON_ID_MISMATCH:
                raise KeyError(f"Embedding id '{i}' missing in FASTA labels")
            else:
                continue
        test_x_embeddings_per_protein.append(f[i][:])
        test_y.append(gene_id_label_dict_test[i])
        test_id.append(i)

if not test_x_embeddings_per_protein:
    raise RuntimeError("No (embedding, label) pairs found. Check that H5 keys match FASTA IDs.")

X_test = np.array(test_x_embeddings_per_protein, dtype=np.float32)
Y_test = np.array(test_y, dtype=np.float32)
rounded_labels = np.argmax(Y_test, axis=1)

# ------------------------ Load model & predict on SAME GPU -------
with tf.device('/device:GPU:0'):
    model = load_model(model_path, compile=False)
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)

# Argmax model predictions
y_pred_model = np.argmax(y_pred, axis=1)

# ------------------------ Write low-confidence IDs ---------------
with open(OUT_ID_TXT, 'w') as file:
    file.write("id ground_truth predicted\n")
    wrote = 0
    for i, tid in enumerate(test_id):
        p_cls = y_pred_model[i]
        p_prob = float(y_pred[i][p_cls])
        if p_prob < LOW_CONF_THRESH:  # Confidence level check
            file.write(f"{tid.replace(' ', '_')} {rounded_labels[i]} {p_cls}\n")
            wrote += 1
print(f"[info] low-confidence count (<{LOW_CONF_THRESH:.2f}): {wrote}")

# We'll build this only if we actually run DIAMOND
knn_map = {}

# ------------------------ Build FASTA for low-confidence + DIAMOND ----------
if wrote == 0 and SKIP_DIAMOND_IF_EMPTY:
    print("[info] No low-confidence sequences. Skipping DIAMOND steps.")
else:
    mydata = pd.read_csv(OUT_ID_TXT, sep=r'\s+')  # delim_whitespace deprecated
    arg_list = list(mydata['id']) if not mydata.empty else []

    with open(OUT_FASTA, 'w') as f:
        for sequence in SeqIO.parse(seq_test_path, "fasta"):
            prot_description = str(sequence.description)
            this_prot_id = prot_description.split('|')[0]
            label = prot_description.split('|')[1]
            prot = str(sequence.seq).replace('U', 'X').replace('Z', 'X').replace('O', 'X')
            if this_prot_id in arg_list:
                f.write(">" + this_prot_id + "|" + label + "\n" + prot + "\n")

    if not os.path.exists(OUT_FASTA) or os.path.getsize(OUT_FASTA) == 0:
        print(f"[warn] {OUT_FASTA} is empty. Skipping DIAMOND.")
    else:
        # ------------------------ DIAMOND alignment ----------------------
        rc1 = os.system(f"{DIAMOND_BIN} makedb --in {TRAIN_FASTA} -d {DIAMOND_DB_NAME}")
        rc2 = os.system(f"{DIAMOND_BIN} blastp -d {DIAMOND_DB_NAME} -q {OUT_FASTA} -o {DIAMOND_OUT}")

        if rc2 != 0 or not os.path.exists(DIAMOND_OUT):
            raise RuntimeError("DIAMOND blastp failed or produced no output. "
                               "Check that 'diamond' is on PATH and the FASTA is valid.")

        # Add header
        with open(DIAMOND_OUT, 'r') as diamond_file:
            lines = diamond_file.readlines()
            with open(DIAMOND_OUT_FINAL, 'w') as dmd_file:
                dmd_file.write('qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore\n')
                for line in lines:
                    dmd_file.write(line)

        diamond_data = pd.read_csv(DIAMOND_OUT_FINAL, sep=r'\s+')

        # ------------------------ Aggregate hits per query ----------------
        if not diamond_data.empty:
            diamond_data = diamond_data.sort_values(by=['qseqid']).reset_index(drop=True)

        i = 0
        while i < diamond_data.shape[0]:
            query_id_full = diamond_data['qseqid'][i]
            query_id = query_id_full.split('|')[0]  # strip label from qseqid

            candidate_class = [0] * NUM_CLASSES
            candidate_class_count = [0] * NUM_CLASSES

            while i < diamond_data.shape[0] and diamond_data['qseqid'][i].split('|')[0] == query_id:
                hit_tokens = str(diamond_data['sseqid'][i]).split('|')
                if len(hit_tokens) >= 2:
                    hit_label = hit_tokens[1]
                    if hit_label in index_of_args:
                        cls = index_of_args[hit_label]
                        candidate_class[cls] += float(diamond_data['bitscore'][i])
                        candidate_class_count[cls] += 1
                i += 1

            # average by count where applicable
            for j in range(NUM_CLASSES):
                if candidate_class_count[j] != 0:
                    candidate_class[j] /= candidate_class_count[j]

            knn_id = int(np.argmax(candidate_class))
            knn_map[query_id] = knn_id

# ------------------------ Merge: use DIAMOND label if present, else keep model ----------
y_pred_final = np.array([knn_map.get(tid, y_cls) for tid, y_cls in zip(test_id, y_pred_model)])
num_overridden = int(np.sum([tid in knn_map for tid in test_id]))
print(f"[info] predictions overridden by DIAMOND: {num_overridden}")

# ------------------------ Report ---------------------------------
labels = list(range(NUM_CLASSES))
target_names = [reverse_of_args[i] for i in labels]
print("\nConfusion matrix:")
print(confusion_matrix(rounded_labels, y_pred_final, labels=labels))
print("\nClassification report:")
print(classification_report(rounded_labels, y_pred_final, labels=labels, target_names=target_names, digits=4, zero_division=0))
