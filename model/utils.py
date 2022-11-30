import math
import os
import pandas as pd
from multiprocessing import cpu_count
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

ID_COLS = ['serialnumber']
columnlabel = ['serialnumber','joint5x', 'joint5y', 'joint6x', 'joint6y', 'joint7x', 'joint7y','joint0x', 'joint0y', 'joint1x', 'joint1y', 'joint2x', 'joint2y', 'joint3x', 'joint3y', 'joint4x', 'joint4y']


def LengthFormatter(shomenlabel, targetlength):
  unitleap = (len(shomenlabel)-1)/(targetlength-1)
  indexlist = []
  for i in range(targetlength):
    indexlist.append(i*unitleap)

  newlist = []
  for j in indexlist:
    if math.ceil(j)>= len(shomenlabel):
      newlist.append(shomenlabel[math.floor(j)])
      break
    if j%1 == 0:      #when the index is integer
      newlist.append(shomenlabel[int(j)])  
    else:
      remain = j - math.floor(j)
      atemp = shomenlabel[math.floor(j)]+remain*(shomenlabel[math.ceil(j)] - shomenlabel[math.floor(j)])
      newlist.append(atemp)

  return newlist

def StandardTraningdatabuilder(dataframelist, columnlabel):
  stdflist = []
  for i in range(len(dataframelist)):
    
    stdf = pd.DataFrame(index=range(100),columns=range(17))
    stdf.columns = columnlabel
    
    for j in columnlabel:
      Name_list = dataframelist[i][j].tolist()
      templist = LengthFormatter(Name_list, 100)
      stdf[j] = templist

    stdflist.append(stdf)

  for i in range(len(stdflist)):
    stdflist[i]['serialnumber'] = stdflist[i]['serialnumber'].astype(int)
    if i == 0:
      X_traintotal= stdflist[0].diff(periods = -1)
      X_traintotal["serialnumber"]=  stdflist[0]['serialnumber'].values
      X_traintotal.drop(X_traintotal.tail(1).index,inplace=True) # drop last n rows
    else: 
      tempdiff = stdflist[i].diff(periods = -1)
      tempdiff["serialnumber"]=  stdflist[i]['serialnumber'].values
      tempdiff.drop(tempdiff.tail(1).index,inplace=True) # drop last n rows
      frames = [X_traintotal, tempdiff]#delete the first line and without serialnumber column
      X_traintotal = pd.concat(frames)
       
  return X_traintotal

def Traindataloader():
    your_path = '/content/20220302/data20220302'
    files = os.listdir(your_path)
    keyword = 'data'
    dataframelist = []

    for file in sorted(glob.glob('/content/data20220307kote/*.csv')):

        if keyword in file:
            print(file)
            X_train = pd.read_csv(file)
            dataframelist.append(X_train)

    #merge the datasets
    for i in range(len(dataframelist)):
        list1s = [i]*len(dataframelist[i])
        dataframelist[i].insert(0, '',list1s, True)
        
        dataframelist[i].columns =['serialnumber','joint5x', 'joint5y', 'joint6x', 'joint6y', 'joint7x', 'joint7y','joint0x', 'joint0y', 'joint1x', 'joint1y', 'joint2x', 'joint2y', 'joint3x', 'joint3y', 'joint4x', 'joint4y']
        print(dataframelist[i].head())

        if i == 0:
            X_traintotal= dataframelist[0]
        else: 
            frames = [X_traintotal, dataframelist[i]]
            X_traintotal = pd.concat(frames)
            
    columnlabel = ['serialnumber','joint5x', 'joint5y', 'joint6x', 'joint6y', 'joint7x', 'joint7y','joint0x', 'joint0y', 'joint1x', 'joint1y', 'joint2x', 'joint2y', 'joint3x', 'joint3y', 'joint4x', 'joint4y']

    X_train = StandardTraningdatabuilder(dataframelist, columnlabel)
    y_train = pd.read_csv('20220302/y_train.csv')
    return X_train, y_train

def create_datasets(X, y, test_size=0.1, dropcols=ID_COLS, time_dim_first=False):
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    X_grouped = create_grouped_array(X)
    if time_dim_first:
        X_grouped = X_grouped.transpose(0, 2, 1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_grouped, y_enc, test_size=0.1)
    X_train, X_valid = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_valid)]
    y_train, y_valid = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_valid)]
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    return train_ds, valid_ds, enc


def create_grouped_array(data, group_col='serialnumber', drop_cols=ID_COLS):
    X_grouped = np.row_stack([
        group.drop(columns=drop_cols).values[None]
        for _, group in data.groupby(group_col)])
    return X_grouped


def create_test_dataset(X, drop_cols=ID_COLS):
    X_grouped = np.row_stack([
        group.drop(columns=drop_cols).values[None]
        for _, group in X.groupby('serialnumber')])
    X_grouped = torch.tensor(X_grouped.transpose(0, 2, 1)).float()
    y_fake = torch.tensor([0] * len(X_grouped)).long()
    return TensorDataset(X_grouped, y_fake)


def create_loaders(train_ds, valid_ds, bs=512, jobs=0):
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl


def accuracy(output, target):
    return (output.argmax(dim=1) == target).float().mean().item()
    
class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]


def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + np.cos(np.pi*t/t_max))/2
    
    return scheduler

class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        #out = self.fc(out[:, -1, :])
        out = self.dropout(self.fc(out[:, -1, :]))
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]