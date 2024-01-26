
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
        'FOLATE-SYNTHESIS-INHABITOR' : 0,
        'TETRACYCLINE' : 1,
        'MULTIDRUG' : 2,
        'BETA-LACTAM' : 3,
        'STREPTOGRAMIN' : 4,
        'AMINOGLYCOSIDE' : 5,
        'FOSFOMYCIN' : 6,
        'TRIMETHOPRIM' : 7,
        'MACROLIDE' : 8,
        'MACROLIDE/LINCOSAMIDE/STREPTOGRAMIN' : 9,
        'RIFAMYCIN' : 10,
        'SULFONAMIDE' : 11,
        'GLYCOPEPTIDE' : 12,
        'QUINOLONE' : 13, 
        'BACITRACIN' : 14,
        'PHENICOL' : 15
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

reverse_of_args = {v: k for k, v in index_of_args.items()}

AMINO_ACID_VOCABULARY_with_index = {
    'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14,'S':15, 'T':16, 'V':17, 'W':18, 'Y':19
}

def create_lablels_of_protein_seq(args):
    label = [0] * 16 
    arg_list = args.split(',')
    for i in arg_list:
        label[index_of_args[i.strip()]] = 1
    
    return label

def read_fasta(file):
    
    # dictionary contains seq_id as keys and protein seqs as values
    seq_dict = {}
    for sequence in SeqIO.parse(file, "fasta"):

        prot_description = str(sequence.description)
        this_prot_id = prot_description.split('|')[0]

        this_prot_label = prot_description.split('|')[-1]
        seq_dict[this_prot_id] = create_lablels_of_protein_seq(this_prot_label)
        
    return seq_dict

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


embedding_test_filename = "./Embeddings/Greater_50_test.h5"
seq_test_path = "./Data/Greater_50_test.fasta"
gene_id_label_dict_test = read_fasta(seq_test_path)

test_x_embeddings_per_protein = []
test_y = []
test_id = []
with h5py.File(embedding_test_filename, "r") as f:
    # List all groups
    L = list(f.keys())
    #print("Here is L:",L)
    
    for i in L:
        test_x_embeddings_per_protein.append(f[i][:])
        test_y.append(gene_id_label_dict_test[i])
        test_id.append(i)
    print("test_y:--------------------------------",test_y[0],test_id[0])

test_x_embeddings_per_protein = np.array(test_x_embeddings_per_protein)
test_y = np.array(test_y)

model = load_model('./Models/Coala_90_Overall_Train.h5')

rounded_labels=np.argmax(test_y, axis=1)
#print("printing test_y: ",test_y)
try:
# Specify an invalid GPU device
    with tf.device('/device:GPU:0'):
        y_pred = model.predict(test_x_embeddings_per_protein, batch_size=64, verbose=1)
except RuntimeError as e:
    print(e)

y_pred_bool = np.argmax(y_pred, axis=1)


with open("low_confident_id_90_all.txt",'w') as file:
    file.write("id ground_truth predicted"+"\n")
    for i in range(0,y_pred_bool.size,1):
        j = y_pred[i][y_pred_bool[i]]
        if(j<1):    #Confidence level check    
            file.write(test_id[i].replace(" ","_")+" "+str(rounded_labels[i])+" "+str(y_pred_bool[i])+"\n")    



mydata = pd.read_csv('low_confident_id_90_all.txt', delim_whitespace=True)


arg_list = []
for i in range(0,mydata.shape[0],1):
    arg_list.append(mydata['id'][i])

#Make the fasta file that has the args with low confidence

with open('low_confident_90_all.fasta','w') as f:
    for sequence in SeqIO.parse(seq_test_path, "fasta"):
            prot_description = str(sequence.description)
            this_prot_id = prot_description.split('|')[0]
            label = prot_description.split('|')[1]
            prot = str(sequence.seq)
            prot = prot.replace('U','X').replace('Z','X').replace('O','X')
            if this_prot_id in arg_list:
                 f.write(">"+this_prot_id+"|"+label+"\n"+prot+"\n")


#Now run diamond with the training set

Train_data = "Data/Coala_90_Overall_Train.fasta"
os.system("diamond makedb --in "+Train_data+" -d Train")
os.system("diamond blastp -d Train -q low_confident_90_all.fasta -o low_confident_90_diamond.txt")


# Just adding the title. Unnecessary? 
with open('low_confident_90_diamond.txt','r') as diamond_file:
    lines = diamond_file.readlines()        
    with open('low_confident_90_diamond_final.txt','w') as dmd_file:
        dmd_file.write('qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore\n')
        for line in lines:
            dmd_file.write(line)

diamond_data = pd.read_csv('low_confident_90_diamond_final.txt', delim_whitespace=True)
diamond_hits = []
i=0
query_list = []
query_label = []
while i < diamond_data.shape[0]:
    diamond_hits.append(diamond_data['qseqid'][i])
    
    candidate_class = [0]*16
    candidate_class_count = [0]*16
    query_id = (diamond_data['qseqid'][i]).split('|')[0]    # query id added to the query list
    while(((diamond_data['qseqid'][i]).split('|')[0])==query_id):
        hit_id = (diamond_data['sseqid'][i]).split('|')[1] # Taking the label name of the hit id
        candidate_class[index_of_args[hit_id]]+=diamond_data['bitscore'][i]
        candidate_class_count[index_of_args[hit_id]]+=1
        if(i+1 < diamond_data.shape[0]):
            i+=1
        else:
            i+=1
            break        
    for j in range(0,16,1):
        if(candidate_class_count[j]!=0):
            candidate_class[j]=candidate_class[j]/candidate_class_count[j]
    knn_id = candidate_class.index(max(candidate_class))
    query_list.append(query_id)
    query_label.append(knn_id)

for i in range(0,len(test_y),1):
    if(test_id[i] in query_list):
       # print("id",test_id[i]," label: ",y_pred_bool[i]," true label: ",rounded_labels[i]," new_label: ",query_label[(query_list.index(test_id[i]))])
        y_pred_bool[i] = query_label[(query_list.index(test_id[i]))]
    else:           #Include this part for just calculating alignment accuracy
        y_pred_bool[i] = -1


        

print(classification_report(rounded_labels, y_pred_bool))