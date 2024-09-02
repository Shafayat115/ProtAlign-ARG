
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

# Data
The Data folder contains the dataset processed into train and test sets. To achieve the clustering results, you can use the graph part and CDHIT tool.

# Train
The ARG_Train.py and ARG_Test.py are the training and testing scripts. You can use Generate_Embedding.py to generate embedding for your train or test set. Use requirements.txt to have the necessary libraries installed. 

# Results
The Models and Embeddings folder has the generated models and embeddings for the experiments. You can use these models to further analyze or validate the results.

# Citation
If you would like to use ARG-BERT, please cite our paper
```sh
<div style="border: 1px solid black; padding: 10px; background-color: #f9f9f9;">
  <strong>Important Note:</strong>
  <p>This is a box created using HTML within a Markdown file. You can customize the style as needed.</p>
</div>
```sh
