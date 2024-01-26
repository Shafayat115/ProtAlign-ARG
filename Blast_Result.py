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
import os
from Bio import SeqIO
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report
from tensorflow import keras
index_of_args = {
    'aminoglycoside' : 0,
        'bacitracin' : 1,
        'beta_lactam' : 2,
        'chloramphenicol' : 3,
        'fosfomycin' : 4,
        'glycopeptide' : 5,
        'multidrug' : 6,
        'polymyxin' : 7,
        'quinolone' : 8,
        'rifampin' : 9,
        'sulfonamide' : 10,
        'tetracycline' : 11,
        'macrolide-lincosamide-streptogramin' : 12,
        'trimethoprim' : 13
}

# true labels
y_true = []

# predicted labels
y_pred = []

# generate the classification report



Train_data = "Data/Graph_Part_Train.fasta"
Test_data = "Data/Graph_Part_Test.fasta"
os.system("diamond makedb --in "+Train_data+" -d Train")
os.system("diamond blastp -d Train -q "+Test_data+" -e 1e-10 -o diamond_output.txt")

with open('diamond_output.txt','r') as diamond_file:
    lines = diamond_file.readlines()        
    with open('diamond_output_final.txt','w') as dmd_file:
        dmd_file.write('qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore\n')
        for line in lines:
            dmd_file.write(line)

diamond_data = pd.read_csv('diamond_output_final.txt', delim_whitespace=True)


diamond_data = pd.read_csv('diamond_output_final.txt', delim_whitespace=True)
diamond_hits = []
i=0
query_list = []
query_label = []
while i < diamond_data.shape[0]:
    diamond_hits.append(diamond_data['qseqid'][i])
    
    candidate_class = [0]*14
    candidate_class_count = [0]*14
    query_id = (diamond_data['qseqid'][i]).split('|')[0]
    while(((diamond_data['qseqid'][i]).split('|')[0])==query_id):
        hit_id = (diamond_data['sseqid'][i]).split('|')[1]
        candidate_class[index_of_args[hit_id]]+=diamond_data['bitscore'][i]
        candidate_class_count[index_of_args[hit_id]]+=1
        if(i+1 < diamond_data.shape[0]):
            i+=1
        else:
            i+=1
            break        
    for j in range(0,14,1):
        if(candidate_class_count[j]!=0):
            candidate_class[j]=candidate_class[j]/candidate_class_count[j]
    knn_id = candidate_class.index(max(candidate_class))
    query_list.append(query_id)
    query_label.append(knn_id)


for sequence in SeqIO.parse(Test_data, "fasta"):
    prot_description = str(sequence.description)
    this_prot_id = prot_description.split('|')[0]
    label = prot_description.split('|')[1]
    prot = str(sequence.seq)
    
    if(this_prot_id in query_list):
        y_pred.append(query_label[(query_list.index(this_prot_id))])
        y_true.append(index_of_args[label])
    else:
        if(index_of_args[label]!=9):
            y_pred.append(9)
        else:
            y_pred.append(1)
        y_true.append(index_of_args[label])






print(y_pred.count(-1))


#target_names = ['class 0', 'class 1']
print(classification_report(y_true, y_pred))