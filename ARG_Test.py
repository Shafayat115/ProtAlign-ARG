
import h5py
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Masking
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten, concatenate
from tensorflow.keras.models import Sequential, load_model, save_model, Model
from tensorflow.keras.layers import BatchNormalization, Dropout
from Bio import SeqIO
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report
from tensorflow import keras
index_of_args = {
    'ARG'  : 0,
    'Non-ARG' : 1
}

AMINO_ACID_VOCABULARY_with_index = {
    'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14,'S':15, 'T':16, 'V':17, 'W':18, 'Y':19
}

def create_lablels_of_protein_seq(args):
    label = [0] * 2 # there are 23 types of metals in BacMet DB
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


embedding_test_filename = "./Embeddings/Tetracycline.h5"
seq_test_path = "./Data/Tetracycline.fasta"
gene_id_label_dict_test = read_fasta(seq_test_path)

test_x_embeddings_per_protein = []
test_y = []
with h5py.File(embedding_test_filename, "r") as f:
    # List all groups
    L = list(f.keys())
    #print(f[L[0]][:])
    
    for i in L:
        test_x_embeddings_per_protein.append(f[i][:])
        test_y.append(gene_id_label_dict_test[i])


test_x_embeddings_per_protein = np.array(test_x_embeddings_per_protein)
test_y = np.array(test_y)
#model = keras.models.load_model('/home/shafayatpiyal/protAlbert/Results/model_with_per_prot_embed_from_protAlbert_tanh_cd_hit40_from_all_mrg.h5')

model = load_model('./Models/ARG_Identification_Tetracycline_30.h5')
#for row in test_x_embeddings_per_protein:
    #print(row)
#y_pred1 = model.predict(test_x_embeddings_per_protein)
#y_pred = np.argmax(y_pred1, axis=1)
        


#Print f1, precision, and recall scores
#print(precision_score(test_y, y_pred , average="macro"))
#print(recall_score(test_y, y_pred , average="macro"))
#print(f1_score(test_y, y_pred , average="macro"))

rounded_labels=np.argmax(test_y, axis=1)
try:
# Specify an invalid GPU device
    with tf.device('/device:GPU:1'):
        y_pred = model.predict(test_x_embeddings_per_protein, batch_size=64, verbose=1)
except RuntimeError as e:
    print(e)

y_pred_bool = np.argmax(y_pred, axis=1)


count = 0
count2 = 0
count3 = 0
for i in range(0,y_pred_bool.size,1):
    j = y_pred[i][y_pred_bool[i]]
    #if(j<0.7):
    # if(y_pred_bool[i]!=rounded_labels[i] and (y_pred_bool[i]==1 or rounded_labels[i]==0)):
    #     count2+=1
    print("y_pred: ",j," label: ",y_pred_bool[i]," true label: ",rounded_labels[i])
    #    count+=1
    #if(y_pred_bool[i]!=14 and j<0.7):
    #     y_pred_bool[i]=15
#print(count)
print(count2)


print(classification_report(rounded_labels, y_pred_bool))