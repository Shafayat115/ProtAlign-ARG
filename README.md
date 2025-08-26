
![2](https://github.com/Shafayat115/ProtAlign-ARG/assets/94142950/b7d433b8-ac87-4ef9-92fd-5f7afb8350f7)
# ProtAlign-ARG
Increasing antibiotic resistance poses a severe threat to human health. Detecting and categorizing antibiotic resistance genes (ARGs), genes conferring resistance to antibiotics in sequence data is vital for mitigating the spread of antibiotic resistance. Recently, large protein language models have been used to identify ARGs. Comparatively, these deep learning methods show superior performance in identifying distant related ARGs over traditional alignment-base methods, but poorer performance for ARG classes with limited training data. Here we introduce ProtAlign-ARG,
a novel self-supervised hybrid model that combines a pre-trained protein language model and an alignment scoring-based model to identify/classify ARGs. 
ProtAlign-ARG learns from vast unannotated protein sequences, utilizing raw protein language model embeddings to classify ARGs. In instances where the model lacks confidence, 
ProtAlign-ARG 
employs an alignment-based scoring method, incorporating bit scores and e-values to classify ARG drug classes. ProtAlign-ARG demonstrates remarkable accuracy in identifying and classifying ARGs, particularly excelling in recall compared to existing ARG identification and classification tools. We also extend ProtAlign-ARG
to predict the functionality and mobility of these genes, highlighting the model's robustness in various predictive tasks. 
A comprehensive comparison of  ProtAlign-ARG with both the alignment-based scoring model
and the pre-trained protein language model clearly shows the superior performance of ProtAlign-ARG.


Quick start
# Clone

git clone https://github.com/Shafayat115/ProtAlign-ARG.git
cd ProtAlign-ARG

# Environment (choose one)

Conda (recommended)
conda env create -f environment.yml
conda activate protalignARG

# Pip (virtualenv)
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Install DIAMOND (for the alignment fallback)
easiest via bioconda
conda install -y -c bioconda diamond
# or install a binary and ensure `diamond` is on your PATH

# Data

contains (your) train/test FASTA splits.

IDs must match between FASTA headers and HDF5 embedding keys.

Generate clustered splits via GraphPart/CD-HIT if you want non-redundant training.

>SEQ_ID|LABEL[,LABEL2,...]
MSEQUENCE...

For binary classification, you’ll see labels like ARG or Non-ARG. For multi-class experiments, you’ll see drug-class labels.


Workflow
A) Generate embeddings

Use the provided script (e.g., ProtAlbert via 🤗 Transformers). It writes per-protein embeddings to HDF5.

python Generate_Embedding.py


Outputs land in ./Embeddings/. Make sure the script’s input/output paths point to your FASTA and desired H5 file.

B) Train the classifier

We provide a configurable training script (edit only the CONFIG block at the top).

python ARG_Train.py

Uses embeddings (e.g., 4096-dim ProtAlbert) + MLP

Saves a model (e.g., ./Models/ARG_Class_Beta_Train.h5 or ./Models/Graph_Part_ARG_40_Train.h5 depending on your config)

C) Evaluate (model-only)

Simple evaluation without alignment fallback:


python ARG_Test.py

Artifacts:

low_confident_*.txt — list of IDs, their gt/pred classes

low_confident_*.fasta — low-confidence sequences

*_diamond.txt / *_diamond_final.txt — DIAMOND outputs

Final metrics printed to console


Scripts at a glance

Generate_Embedding.py — makes per-protein embeddings (HDF5)

ARG_Train.py — configurable trainer (edit CONFIG block only)

ARG_Test.py — evaluates model-only

Test.py — configurable hybrid evaluator (model + DIAMOND fallback)

Both ARG_Train.py and Test.py expose:

GPU selection, XLA toggle

Paths (embeddings, FASTA, model)

Label map (auto-sizes output layer)

MLP architecture & training hyper-params

Low-confidence threshold (hybrid)

Behavior on H5↔FASTA ID mismatches

Repro tips & troubleshooting

GPU selection
Set GPU_INDEX in scripts (or CUDA_VISIBLE_DEVICES) to pick a device. Scripts also set memory growth to avoid full pre-allocation.

Transformers / ProtAlbert weights
First run will download weights. If you see a warning about torch.load security, prefer safetensors or use a recent PyTorch.

NumPy / h5py ABI mismatch
If you ever hit ValueError: numpy.dtype size changed, use conda-forge pinning (e.g., numpy<2 with compatible h5py) or reinstall both from conda-forge.

TensorFlow device errors
Our Keras code uses Keras-native activations (no tf.nn.* on symbolic tensors) and pins to a single GPU to avoid “variable on a different device” errors.

DIAMOND not found
Ensure diamond is on PATH or edit DIAMOND_BIN in the script CONFIG to point to the binary.

Results

Trained models are saved under ./Models/

Embeddings in ./Embeddings/

Use these artifacts to reproduce figures or run external validation

Citation

If you use ProtAlign-ARG, please cite:

@article{ahmed2025protalign,
  title={ProtAlign-ARG: antibiotic resistance gene characterization integrating protein language models and alignment-based scoring},
  author={Ahmed, Shafayat and Emon, Muhit Islam and Moumi, Nazifa Ahmed and Huang, Lifu and Zhou, Dawei and Vikesland, Peter and Pruden, Amy and Zhang, Liqing},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={30174},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
