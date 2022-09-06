import sys
import numpy as np
from typing import Tuple

MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import os

directory = os.getcwd()

device = torch.device('cuda') if torch. cuda. is_available() else torch.device('cpu')
TrainUrl = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"
data = pd.read_csv(TrainUrl, sep='=', header=None, names = ["Input", "Output"])

# input sequence as list of strings format
data_list_input = data['Input'].values.tolist()
# output sequence as list of strings format
data_list_output = data['Output'].values.tolist()

# size of the dataset
size = data.shape[0]

# string to index mapping
stoi = {}
# index to string mapping
itos = {}

# Padding token
stoi['<PAD>'] = 0
itos[0] = '<PAD>'
# Out of vocabulary token
stoi['<OOV>'] = 1
itos[1] = '<OOV>'
# Start of sentence token
stoi['<SOS>'] = 2
itos[2] = '<SOS>'
# End of sentence token
stoi['<EOS>'] = 3
itos[3] = '<EOS>'

# traverse data and build vocabulary
for i in range(0, size):
  
  for j in range(0, len(data_list_input[i])):
    x = stoi.get(data_list_input[i][j], -1)
    index = len(stoi.keys())
    if (x == -1):
      stoi[data_list_input[i][j]] = index
      itos[index] = data_list_input[i][j]

  for j in range(0, len(data_list_output[i])):
    x = stoi.get(data_list_output[i][j], -1)
    index = len(stoi.keys())
    if (x == -1):
      stoi[data_list_output[i][j]] = index
      itos[index] = data_list_output[i][j]

# Rounding off max seq len to 32 after including <SOS>, <EOS> and <PAD> 
max_seq_len = 32
embedding_dimension = 256
b_size = 64

# Convert input data from string to integer format
# with the help of stoi vocabulary built previously

data_list_input_index = []
data_list_input_padding = []
data_list_output_index = []
data_list_output_padding = []

for i in range(0, len(data_list_input)):

  x = []
  x_pad = []
  
  for j in range(0, len(data_list_input[i])):
    x.append(stoi[data_list_input[i][j]])
    x_pad.append(False)

  # Append <EOS> token in the end
  x.append(stoi['<EOS>'])
  x_pad.append(False)

  while(len(x)<max_seq_len):
    x.append(stoi['<PAD>'])
    x_pad.append(True)

  data_list_input_index.append(x)
  data_list_input_padding.append(x_pad)

  y = []
  y_pad = []

  # Append <SOS> token in the end
  y.append(stoi['<SOS>'])
  y_pad.append(False)

  for j in range(0, len(data_list_output[i])):
    y.append(stoi[data_list_output[i][j]])
    y_pad.append(False)

  # Append <EOS> token in the end
  y.append(stoi['<EOS>'])
  y_pad.append(False)
  while(len(y)<max_seq_len):
    y.append(stoi['<PAD>'])
    y_pad.append(True)
  data_list_output_index.append(y)
  data_list_output_padding.append(y_pad)
  
# Define custom dataset from PyTorch Dataset class
class MyDataset (Dataset):

  def __init__(self, data_list_input_index, data_list_output_index, data_list_input_padding, data_list_output_padding):
    # Lists of lists
    self.X = data_list_input_index 
    self.Y = data_list_output_index
    self.X_pad_mask = data_list_input_padding
    self.Y_pad_mask = data_list_output_padding

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return torch.tensor(self.X[idx], dtype=torch.int), torch.tensor(self.Y[idx], dtype=torch.int) , torch.tensor(self.X_pad_mask[idx], dtype=torch.bool), torch.tensor(self.Y_pad_mask[idx], dtype=torch.bool)

# Train, Validation and Test data sets
# 60/20/20 split of the original data
train_data = MyDataset(data_list_input_index[0:(int)(0.6*size)], data_list_output_index[0:(int)(0.6*size)], data_list_input_padding[0:(int)(0.6*size)], data_list_output_padding[0:(int)(0.6*size)])
val_data = MyDataset(data_list_input_index[(int)(0.6*size):(int)(0.8*size)], data_list_output_index[(int)(0.6*size):(int)(0.8*size)], data_list_input_padding[(int)(0.6*size):(int)(0.8*size)], data_list_output_padding[(int)(0.6*size):(int)(0.8*size)])
test_data = MyDataset(data_list_input_index[(int)(0.8*size):], data_list_output_index[(int)(0.8*size):], data_list_input_padding[(int)(0.8*size):], data_list_output_padding[(int)(0.8*size):])

# Train, Validation and Test data loaders
train_loader = DataLoader(train_data, batch_size = b_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size = b_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size = b_size, shuffle=False)

# Positional Embeddings - in accordance with Attention Is All You Need paper
pos_embeddings = np.zeros((max_seq_len, embedding_dimension), dtype=float)

for i in range(0, max_seq_len):
  for j in range(0, embedding_dimension):
    omega = 1 / (10000**((j - (j%2))/embedding_dimension))
    if (j % 2 == 0):
      pos_embeddings[i][j] = math.sin(i*omega)
    else:
      pos_embeddings[i][j] = math.cos(i*omega)

pos_embeddings = ((torch.from_numpy(pos_embeddings)).float()).to(device)

# Embeddings layer - used for both transformer encode and decoder
class Embed_Model(nn.Module):

  def __init__(self, number_embeddings, embedding_dimension, padding_index, pos_embeddings):
        
    super(Embed_Model, self).__init__()

    self.number_embeddings = number_embeddings
    self.embedding_dimension = embedding_dimension
    self.padding_index = padding_index
    self.pos_embeddings = pos_embeddings

    self.embedding_layer = nn.Embedding(num_embeddings=self.number_embeddings, embedding_dim=self.embedding_dimension, padding_idx=self.padding_index)

  def forward(self, x):
    # Input dimension - (batch size, seq length)
    # Sequence length is set as max_seq_length. Padding tokens are added in the end
    output = self.embedding_layer(x) + self.pos_embeddings
    # Output dimension - (batch size, seq length, embedding dim)
    # Add the positional embeddings and return the output
    return output

# Complete Model
# Contains both embedding layer and transformer model (as implemented by PyTorch)

class Model(nn.Module):

  def __init__(self, number_embeddings, embedding_dimension, feedforward_dimension, padding_index, num_heads, encoder_number_layers, decoder_number_layers, pos_embeddings):
        
    super(Model, self).__init__()

    self.embeddings = Embed_Model(number_embeddings, embedding_dimension, padding_index, pos_embeddings)
    self.transformer = nn.Transformer(d_model=embedding_dimension, nhead=num_heads, num_encoder_layers=encoder_number_layers, 
                                      num_decoder_layers=decoder_number_layers, dim_feedforward=feedforward_dimension, batch_first=True)
    self.fc = nn.Linear(embedding_dimension, number_embeddings)

# src: (N, S, E) if batch_first=True.

# tgt: (N, T, E) if batch_first=True.

# tgt_mask: lower triangular matrix marked with False - (T, T)
# [False, True]
# [False, False]

# src_key_padding_mask: mark true for padded tokens - (N, S)

# tgt_key_padding_mask: mark true for padded tokens - (N, T)

# memory_key_padding_mask: kept same as source key padding mask - (N, S)

  def forward(self, src, tgt, tgt_mask, src_key_padding_mask, tgt_key_padding_mask):
    
    src_embedding = self.embeddings(src)
    tgt_embedding = self.embeddings(tgt)
    output = self.transformer(src_embedding, tgt_embedding, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
    output = self.fc(output)
    output = F.softmax(output, dim=2)
    return output

# All the hyperparameters (no of encoders, no of decoders, feed forward dimension, no attention heads) 
# were reduced by half to make the number of trainable parameters approximately 5 Million
number_embeddings = len(stoi.keys())
embedding_dimension = 256
padding_index = 0
num_heads = 4
encoder_number_layers = 3
decoder_number_layers = 3
feedforward_dimension = 1024

# To prevent decoder from cheating (so it doesn't attend to future positions)
# Lower Triangular Matrix marked as false
tgt_mask_tensor = torch.zeros((max_seq_len, max_seq_len), dtype=torch.bool)

for i in range(0, max_seq_len):
  for j in range(0, max_seq_len):

    if (j<=i):
      tgt_mask_tensor[i][j] = False

    else:
      tgt_mask_tensor[i][j] = True

model_path = directory + "/model_weight.pt"
pos_embeddings = pos_embeddings.to(device)
neural_net_loaded = Model(number_embeddings, embedding_dimension, feedforward_dimension, padding_index, num_heads, encoder_number_layers, decoder_number_layers, pos_embeddings)
neural_net_loaded = neural_net_loaded.to(device)
neural_net_loaded = torch.load(model_path, map_location=device)
neural_net_loaded.eval()

def predict(factors: str):

    factors_list = list(factors)
    factors_list.append('<EOS>') 

    while(len(factors_list) < max_seq_len):
      factors_list.append('<PAD>')

    # print(factors_list)

    factors_list_index = []
    factors_list_padding = []

    for i in range(0, len(factors_list)):
      factors_list_index.append(stoi.get(factors_list[i], stoi['<OOV>']))
      if (factors_list[i] == '<PAD>'):
        factors_list_padding.append(True)
      else:
        factors_list_padding.append(False)

    # Convert to input with batch size 1
    src = []
    src_mask_padding = []
    tgt_mask_padding = []
    src.append(factors_list_index)
    src_mask_padding.append(factors_list_padding)
    tgt_mask_padding.append([True for i in range(0, max_seq_len)])

    src = ((torch.from_numpy(np.asarray(src, np.int_))).long()).to(device)
    src_mask_padding = ((torch.from_numpy(np.asarray(src_mask_padding, np.bool_))).bool()).to(device)
    tgt_mask_padding = ((torch.from_numpy(np.asarray(tgt_mask_padding, np.bool_))).bool()).to(device)

    # start with a <SOS> token which corresponds with 2
    input_tgt = (torch.zeros((1, max_seq_len)).long()).to(device) 
    input_tgt[:, 0] = stoi['<SOS>']
    output_tgt = (torch.zeros((1, max_seq_len)).long()).to(device)

    tgt_mask_tensor_input = tgt_mask_tensor.to(device)
    
    for i in range(0, max_seq_len):
      tgt_mask_padding[0][i] = False
      output = neural_net_loaded(src, input_tgt, tgt_mask_tensor_input, src_mask_padding, tgt_mask_padding)
      output_tgt = torch.argmax(output, 2)
      if (i+1<max_seq_len):
        input_tgt[:, i+1] = output_tgt[:, i]

    expansions = ""

    for i in range(0, max_seq_len):

      if (output_tgt[0][i] == stoi['<EOS>']):
        break

      expansions = expansions + itos[(int)(output_tgt[0][i])]

    return expansions


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):
    factors, expansions = load_file(filepath)
    pred = [predict(f) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))


if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")