

import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from prettytable import PrettyTable
from IPython.display import Image
import torch
import h5py
import time
from sklearn.preprocessing import LabelEncoder

from keras.models import Model
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, GlobalMaxPooling1D
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using {}".format(device))

import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)


# from google.colab import drive
# drive.mount('/content/drive')

# data_path = 'drive/My Drive/Case_Study/pfam/random_split/'
# print('Available data', os.listdir(data_path))


# def read_data(partition):
#   data = []
#   for fn in os.listdir(os.path.join(data_path, partition)):
#     with open(os.path.join(data_path, partition, fn)) as f:
#       data.append(pd.read_csv(f, index_col=None))
#   return pd.concat(data)

# reading all data_partitions
df_train = pd.read_csv("/home/shafayatpiyal/protAlbert/ablation_study_train.csv", index_col=None)
df_val = pd.read_csv("/home/shafayatpiyal/protAlbert/ablation_study_validation.csv", index_col=None)
df_test = pd.read_csv("/home/shafayatpiyal/protAlbert/ablation_study_test.csv", index_col=None)


df_test.head()

df_train.head(1)['Seq'].values[0]


print('Train size: ', len(df_train))
print('Val size: ', len(df_val))
print('Test size: ', len(df_test))

# # Length of sequence in train data.
df_train['seq_char_count']= df_train['Seq'].apply(lambda x: len(x))
df_val['seq_char_count']= df_val['Seq'].apply(lambda x: len(x))
df_test['seq_char_count']= df_test['Seq'].apply(lambda x: len(x))

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def create_dict(codes):
  char_dict = {}
  for index, val in enumerate(codes):
    char_dict[val] = index+1

  return char_dict

char_dict = create_dict(codes)

print(char_dict)
print("Dict Length:", len(char_dict))


def integer_encoding(data):
  """
  - Encodes code sequence to integer values.
  - 20 common amino acids are taken into consideration
    and rest 4 are categorized as 0.
  """
  
  encode_list = []
  for row in data['Seq'].values:
    row_encode = []
    for code in row:
      row_encode.append(char_dict.get(code, 0))
    encode_list.append(np.array(row_encode))
  
  return encode_list


train_encode = integer_encoding(df_train) 
val_encode = integer_encoding(df_val) 
test_encode = integer_encoding(df_test) 

# padding sequences
max_length = 1600
train_pad = pad_sequences(train_encode, maxlen=max_length, padding='post', truncating='post')
val_pad = pad_sequences(val_encode, maxlen=max_length, padding='post', truncating='post')
test_pad = pad_sequences(test_encode, maxlen=max_length, padding='post', truncating='post')

print(train_pad.shape, val_pad.shape, test_pad.shape)


# One hot encoding of sequences
train_ohe = to_categorical(train_pad)
val_ohe = to_categorical(val_pad)
test_ohe = to_categorical(test_pad)

print(train_ohe.shape, test_ohe.shape, test_ohe.shape) 


le = LabelEncoder()

y_train_le = le.fit_transform(df_train['Label'])
y_val_le = le.transform(df_val['Label'])
y_test_le = le.transform(df_test['Label'])

y_train_le.shape, y_val_le.shape, y_test_le.shape

print('Total classes: ', len(le.classes_))
# le.classes_


# One hot encoding of outputs
y_train = to_categorical(y_train_le)
y_val = to_categorical(y_val_le)
y_test = to_categorical(y_test_le)

print(y_train.shape, y_val.shape, y_test.shape)

def plot_history(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  x = range(1, len(acc) + 1)

  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.plot(x, acc, 'b', label='Training acc')
  plt.plot(x, val_acc, 'r', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(x, loss, 'b', label='Training loss')
  plt.plot(x, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.show()

def display_model_score(model, train, val, test, batch_size):
  #model = model.to(device) # move model to GPU
  train_score = model.evaluate(train[0], train[1], batch_size=batch_size, verbose=1)
  print('Train loss: ', train_score[0])
  print('Train accuracy: ', train_score[1])
  print('-'*70)

  val_score = model.evaluate(val[0], val[1], batch_size=batch_size, verbose=1)
  print('Val loss: ', val_score[0])
  print('Val accuracy: ', val_score[1])
  print('-'*70)
  
  test_score = model.evaluate(test[0], test[1], batch_size=batch_size, verbose=1)
  print('Test loss: ', test_score[0])
  print('Test accuracy: ', test_score[1])




x_input = Input(shape=(1600,))
emb = Embedding(21, 128, input_length=max_length)(x_input)
bi_rnn = Bidirectional(CuDNNLSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))(emb)
x = Dropout(0.3)(bi_rnn)

# softmax classifier
x_output = Dense(15, activation='softmax')(x)

model1 = Model(inputs=x_input, outputs=x_output)
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model1.summary()

es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

history1 = model1.fit(
    train_pad, y_train,
    epochs=50, batch_size=256,
    validation_data=(val_pad, y_val),
    callbacks=[es]
    )
plot_history(history1)

model1.save_weights('/home/shafayatpiyal/protAlbert/Final_Models/ablation_lstm_model1.h5')
display_model_score(model1,
    [train_pad, y_train],
    [val_pad, y_val],
    [test_pad, y_test],
    256)

def residual_block(data, filters, d_rate):
  """
  _data: input
  _filters: convolution filters
  _d_rate: dilation rate
  """

  shortcut = data

  bn1 = BatchNormalization()(data)
  act1 = Activation('relu')(bn1)
  conv1 = Conv1D(filters, 1, dilation_rate=d_rate, padding='same', kernel_regularizer=l2(0.001))(act1)

  #bottleneck convolution
  bn2 = BatchNormalization()(conv1)
  act2 = Activation('relu')(bn2)
  conv2 = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(0.001))(act2)

  #skip connection
  x = Add()([conv2, shortcut])

  return x


x_input = Input(shape=(1600, 21))

#initial conv
conv = Conv1D(128, 1, padding='same')(x_input) 

# per-residue representation
res1 = residual_block(conv, 128, 2)
res2 = residual_block(res1, 128, 3)

x = MaxPooling1D(3)(res2)
x = Dropout(0.5)(x)

# softmax classifier
x = Flatten()(x)
x_output = Dense(15, activation='softmax', kernel_regularizer=l2(0.0001))(x)

model2 = Model(inputs=x_input, outputs=x_output)
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model2.summary()

es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

history2 = model2.fit(
    train_ohe, y_train,
    epochs=20, batch_size=256,
    validation_data=(val_ohe, y_val),
    callbacks=[es]
    )

model2.save_weights('/home/shafayatpiyal/protAlbert/Final_Models/ablation_resnet_model2.h5')
plot_history(history2)

display_model_score(
    model2,
    [train_ohe, y_train],
    [val_ohe, y_val],
    [test_ohe, y_test],
    256)



x = PrettyTable()
x.field_names = ['Sr.no', 'Model', 'Train Acc', 'Val Acc','Test Acc']

#x.add_row(['1.', 'Bidirectional LSTM', '0.964', '0.957', '0.958'])
#x.add_row(['2.', 'ProtCNN', '0.996', '0.988', '0.988'])

print(x)
# %%
