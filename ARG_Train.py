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


#seed = 42
#np.random.seed(seed)

index_of_resistant_mechanisms = {         
        'ARG'  : 0,
        'Non-ARG' : 1
}

AMINO_ACID_VOCABULARY_with_index = {
    'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14,'S':15, 'T':16, 'V':17, 'W':18, 'Y':19
}

def create_lablels_of_protein_seq(args):
    label = [0] * 2
    arg_list = args.split(',')
    for i in arg_list:
        label[index_of_resistant_mechanisms[i.strip()]] = 1
    
    return label


def get_model(number_of_neurons_in_each_hlayer = [2500, 1500, 700, 300, 100]):

    dimension_of_per_prot_embedding = 4096 # 4096 for prot-albert pretrained model and 1024 for prot-bert pre-trined model and prot-T5-XL

    # per protein embeddings generated from protAlbert or other prottrans pre-trained models
    input1 = Input(shape=(dimension_of_per_prot_embedding,))
    
    
    for i in range(len(number_of_neurons_in_each_hlayer)):
        if i == 0:
            x = Dense(number_of_neurons_in_each_hlayer[i])(input1)
            x = tf.nn.tanh(x)
        else:
            x = Dense(number_of_neurons_in_each_hlayer[i])(x)
            x = tf.nn.tanh(x)
            x = Dropout(0.2)(x)
    
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[input1], outputs=output)
    return model 

def train_model(input1, y):

    model = get_model()

    verbose, epochs, batch_size = 1, 20, 128
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
	# fit network

    model.fit([input1], y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    model.save('./Models/ARG_Identification_Tetracycline_30.h5')
    model.summary()

def read_fasta(file):
    
    # dictionary contains seq_id as keys and protein seqs as values
    seq_dict = {}
    for sequence in SeqIO.parse(file, "fasta"):

        prot_description = str(sequence.description)
        this_prot_id = prot_description.split('|')[0]

        this_prot_label = prot_description.split('|')[-1]
        seq_dict[this_prot_id] = create_lablels_of_protein_seq(this_prot_label)
        
    return seq_dict


if __name__ == "__main__":
    
    
    embedding_filename = "./Embeddings/ARG_Identification_Tetracycline_30.h5"
   
    seq_path = "./Data/ARG_Identification_Tetracycline_30.fasta"
    
    gene_id_label_dict = read_fasta(seq_path)
    
    train_y = []
    train_x_embeddings_per_protein = []
    
    with h5py.File(embedding_filename, "r") as f:
        # List all groups
        L = list(f.keys())
        #print(f[L[0]][:])
        
        for i in L:
            train_x_embeddings_per_protein.append(f[i][:])
            train_y.append(gene_id_label_dict[i])
        
    train_x_embeddings_per_protein = np.array(train_x_embeddings_per_protein)
    train_y = np.array(train_y)

    X, Y = shuffle(train_x_embeddings_per_protein, train_y)

    try:
    # Specify an invalid GPU device
        with tf.device('/device:GPU:1'):
            train_model(X, Y)
    except RuntimeError as e:
        print(e)
    
    

    #print(tf.__version__)
    #print(tf.config.list_physical_devices())