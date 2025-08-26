
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import torch
from transformers import AlbertModel, AlbertTokenizer
import h5py
import time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using {}".format(device))

# In the following you can define your desired output. Current options:
# per_residue embeddings
# per_protein embeddings
# secondary structure predictions

# Replace this file with your own (multi-)FASTA
# Headers are expected to start with ">";
seq_path = "./Data/ARG_Class_Beta_Test.fasta"

# whether to retrieve embeddings for each residue in a protein 
# --> Lx1024 matrix per protein with L being the protein's length
# as a rule of thumb: 1k proteins require around 1GB RAM/disk
per_residue = False 
per_residue_path = "./Embeddings/ARG_Class_Beta_Test.h5" # where to store the embeddings

# whether to retrieve per-protein embeddings 
# --> only one 1024-d or 4096-d vector per protein, irrespective of its length. CHECK the embedding dimension. Dimension depends on which pretrained model has been used.
per_protein = True
per_protein_path = "./Embeddings/ARG_Class_Beta_Test.h5" # where to store the embeddings

# whether to retrieve secondary structure predictions
# This can be replaced by your method after being trained on ProtT5 embeddings
sec_struct = False
sec_struct_path = "./Results/ss3_preds.fasta" # file for storing predictions

# make sure that either per-residue or per-protein embeddings are stor  ed
assert per_protein is True or per_residue is True or sec_struct is True, print(
    "Minimally, you need to active per_residue, per_protein or sec_struct. (or any combination)")


#@title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50) 
def get_Albert_model():
    model = AlbertModel.from_pretrained("Rostlab/prot_albert")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)

    return model, tokenizer

#@title Write embeddings to disk. { display-mode: "form" }
def save_embeddings(emb_dict,out_path):
    with h5py.File(str(out_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)
    return None


def read_fasta(file):

    # dictionary contains seq_id as keys and protein seqs as values
    seq_dict = {}
    for sequence in SeqIO.parse(file, "fasta"):

        prot_description = str(sequence.description)
        this_prot_id = prot_description.split('|')[0]

        prot = str(sequence.seq)
        prot = prot.replace('U','X').replace('Z','X').replace('O','X')

        seq_dict[this_prot_id] = prot
        
    return seq_dict


#@title Generate embeddings. { display-mode: "form" }
# Generate embeddings via batch-processing
# per_residue indicates that embeddings for each residue in a protein should be returned.
# per_protein indicates that embeddings for a whole protein should be returned (average-pooling)
# max_residues gives the upper limit of residues within one batch
# max_seq_len gives the upper sequences length for applying batch-processing
# max_batch gives the upper number of sequences per batch
def get_embeddings( model, tokenizer, seqs, per_residue, per_protein, sec_struct, 
                   max_residues=10000, max_seq_len=1600, max_batch=100 ):

    if sec_struct:
      sec_struct_model = load_sec_struct_model()

    results = {"residue_embs" : dict(), 
               "protein_embs" : dict(),
               "sec_structs" : dict() 
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
            
            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            if sec_struct: # in case you want to predict secondary structure from embeddings
              d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)


            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim  
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                if sec_struct: # get classification results
                    results["sec_structs"][identifier] = torch.max( d3_Yhat[batch_idx,:s_len], dim=1 )[1].detach().cpu().numpy().squeeze()
                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze() 
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()


    passed_time=time.time()-start
    #avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    #print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
     #   passed_time/60, avg_time ))
    print('\n############# END #############')
    return results


# Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
model, tokenizer = get_Albert_model()

# Load example fasta.
seqs = read_fasta( seq_path )

# Compute embeddings and/or secondary structure predictions
results = get_embeddings( model, tokenizer, seqs,
                         per_residue, per_protein, sec_struct)

# Store per-residue embeddings
if per_residue:
  save_embeddings(results["residue_embs"], per_residue_path)

if per_protein:
  save_embeddings(results["protein_embs"], per_protein_path)

if sec_struct:
  write_prediction_fasta(results["sec_structs"], sec_struct_path)