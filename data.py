import os
import torch
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class SpeechDataset(Dataset):
    def __init__(self, patterns_path, labels_path, sorting=True):
        """
          Parameters:
            patterns_path:  path to the patterns (x)
                            should be list of N arrays each of (L*, F)
                            - L* is the (variable) pattern length
                            - F is the number of features
            labels_path:    path to the labels (y)
          Variables:
            self.X:         list of numpy arrays, each of size (L x 40)
            self.Y:         list of numy arrays of words, each of size (L)
        """
        loader = lambda s: np.load(s, encoding='bytes')
        self.X = loader(patterns_path).T
        self.y = loader(labels_path)
        # sorting the data
        if sorting:
            zipped = zip(*sorted(zip(self.X, self.y), key=lambda x: x[0].shape[0]))
            self.X, self.Y = (list(l) for l in zipped)
        # turning the numpy arrays into tensors
        for i in range(len(self.X)):
            self.X[i] = torch.Tensor(self.X[i])
        for i in range(len(self.y)):
            self.y[i] = torch.Tensor(self.y[i])
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
    def __len__(self):
        return len(self.X)

def collate_padded(l):
    """
      Called by the data loader when collating lines. 
      Pad the sequences with zeros to turn them into a 3D tensor with 
      the second dimension representing the time (i.e. the length of
      the sequence). This is ensured to be divisible by eight.
    """
    x, y = zip(*l)
    x, y = list(x), list(y)
    # padding
    x = rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y = rnn.pad_sequence(y, batch_first=True, padding_value=0)
    # make sure the sequence lengths for the input are all 
    # multiples of 8
    if x.size(1) % 8 != 0:
        pad_len = (x.size(1) // 8 + 1) * 8 - x.size(1)
        x = F.pad(x, (0, 0, 0, pad_len))
    return x, y

def collate_unpadded(l):
    """
      Called by the data loader when collating lines. 
      Only return the lists, don't pad anything.
    """
    x, y = zip(*l)
    x, y = list(x), list(y)
    return x, y

def train_loader():
    """
      Loads the training data (letter wise). 
      Returns:
        DataLoader object that yields pairs of data and labels.
        Data:
            Tensor of size B x L x N
            B is the batch size, here B=64
            L is the length of the sequence, variable
            N is the number of features, here N=40
        Labels:
            Tensors of size B x T
            B is the batch size, here B=64
            T is the length of the label, variable
    """
    train_dataset = SpeechDataset('data/training_data_preprocessed.npy',
                                  'data/training_labels_proprocessed.npy',
                                  sorting=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_unpadded)
    return train_loader

def val_loader():
    """
      Loads the validation data (letter wise).
      Returns:
        DataLoader object that yields pairs of data and labels.
        Data:
            Tensor of size B x L x N
            B is the batch size, here B=64
            L is the length of the sequence, variable
            N is the number of features, here N=40
        Labels:
            Tensors of size B x T
            B is the batch size, here B=64
            T is the length of the label, variable
    """
    val_dataset = SpeechDataset('data/testing_data_preprocessed.npy',
                                'data/testing_labels_proprocessed.npy',
                                sorting=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_unpadded)
    return val_loader