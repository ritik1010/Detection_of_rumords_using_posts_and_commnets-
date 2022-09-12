# %%
import numpy as np

# from ast import arg
import numpy as np
import torch
import torch.nn as nn
# import config
# import utils
import torch.nn.functional as Fu
from numpy.linalg import norm
import pandas as pd
from torch.autograd import Variable
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
# from config import args
# from postcom2dr import PostCom2DR



# %%
from model import PostCom2DR
from args import args
from utils import *


# %%
model=PostCom2DR(args.m,args.n)
criterion=torch.nn.CrossEntropyLoss()
model.train()
params = [p for p in model.parameters() if p.requires_grad]
# print(len(params))
# params=params+[model.W0]

optimizer = torch.optim.Adam([
    {'params': params}
], lr= 0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

# %%
from tqdm import tqdm
import time
import random
import warnings
warnings.filterwarnings('ignore')
# total_rows=len(X_train)
start_time=time.time()
for epoch in (range(20)):
    loss_this_epoch=0
    num_batches=0
    for pkl in range(0,156):
        train_batch=pd.read_pickle("Train_pkl/df_train_embedded_"+str(pkl)+".pkl")
        X_train,y_train=pandas_to_dict(train_batch)
        # c = list(zip(X_train, y_train))
        # random.shuffle(c)

        # X_train, y_train = zip(*c)
        y_train=torch.Tensor(y_train).long()
        start = time.time()
        for row in range(0,len(X_train)-1,args.BatchSize):
            # print("row:",row)
            
            num_batches+=1
            X_train_batch=X_train[row:row+args.BatchSize]
            y_train_batch=y_train[row:row+args.BatchSize]
            # for i in range(len(X_train_batch)):
            #     print("X_Train_batch:",X_train_batch[i]['label'],"Y_Labele",y_train_batch[i])
            y_cap=model.forward(X_train_batch)
            # print(type(y_cap),type(y.shape),type(y_train))
            loss = criterion(y_cap, y_train_batch)
            # loss.requres_grad = True
            # loss.retain_grad()
            # Zero gradients, perform a backward pass,
            # and update the weights.
            

            loss.backward()
            # for param in model.parameters():
            #     print(param.grad.data.sum())

            # start debugger
            # import pdb; pdb.set_trace()
            optimizer.step()
            optimizer.zero_grad()
            loss_this_epoch+=loss.item()
            # break
        if pkl%8==0:
            print('epoch ',epoch,' pkl ', pkl ,' loss:', loss_this_epoch/num_batches,' TIme elapsed' ,(time.time()-start_time)/3600)
            torch.save(model.state_dict(),"model_"+str(epoch)+"_"+str(pkl)+".pth")
            # break
    scheduler.step()
    end = time.time()

    print('epoch ',epoch, 'loss,',loss_this_epoch/num_batches,' time taken this epoch ',(time.time()-start_time)/3600)



# %%


# %% [markdown]
# 


