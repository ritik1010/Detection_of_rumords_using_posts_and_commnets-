# %%
import os
import numpy as np
import os
import json
import gensim
from ast import arg
import numpy as np
import torch
import torch.nn as nn
# import config
# import utils
import math
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

class args:
    m=200 #Num of  words in tweets
    n=200#Max comments in posts and comments
    d=300#dimension of vector after word2vec
    hidden_lstm_dim=80
    l=80
    m_dash=80
    T=3 #num of comments being compared to calculate topic drift
    k=80 #number of filters
    padd=1
    BatchSize=8

# %%
def pandas_to_dict(input_dataframe):
    all_dict_list=[]
    y=[]
    for emb_data,adj,label in zip(input_dataframe['embedded_data'],input_dataframe['adj'],input_dataframe['label']):
        pres_dict={}
        pres_dict['embedded_data']=emb_data
        pres_dict['adj']=adj
        pres_dict['label']=label
        intlabel=0
        # if label=='true':
        #     intlabel=1
        # else:
        #     # print("int label becomes 0")
        #     intlabel=0
        y.append(label)
        all_dict_list.append(pres_dict)
    return all_dict_list,y


# %%

class PostCom2DR(nn.Module):
    def __init__(self,max_words_per_tweet=args.m,max_comments_per_post=args.n):
        super(PostCom2DR, self).__init__()
        # self.build()
        self.max_words_per_tweet=max_words_per_tweet
        self.max_comments_per_post=max_comments_per_post
        self.W0=nn.parameter.Parameter(torch.ones((args.hidden_lstm_dim,args.m_dash),dtype=torch.double),requires_grad=True)
        self.W1=nn.parameter.Parameter(torch.ones((args.m_dash,args.m_dash),dtype=torch.double),requires_grad=True)
        self.d=math.sqrt(300)
        self.Wcw=nn.parameter.Parameter(torch.ones((args.m_dash,args.hidden_lstm_dim),dtype=torch.double),requires_grad=True)
        #LSTM used in encoding
        self.post_LSTM = nn.LSTM(num_layers=1, input_size=args.d,
                                        hidden_size=int(args.hidden_lstm_dim))
        self.comment_LSTM=nn.LSTM(num_layers=1, input_size=args.d,
                                        hidden_size=int(args.hidden_lstm_dim),
                                        
                                        )
        self.tanh=nn.Tanh()
        self.Ww=nn.parameter.Parameter(torch.ones((args.m,args.l),dtype=torch.double),requires_grad=True)
        self.Wc=nn.parameter.Parameter(torch.ones((args.n,args.m_dash),dtype=torch.double),requires_grad=True)
        self.Whw=nn.parameter.Parameter(torch.ones((args.m,1),dtype=torch.double),requires_grad=True)
        self.Whc=nn.parameter.Parameter(torch.ones((args.n,1),dtype=torch.double),requires_grad=True)
        self.Wh=nn.parameter.Parameter(torch.ones((2,args.hidden_lstm_dim+args.m_dash+args.k),dtype=torch.double),requires_grad=True)
        self.bh=nn.parameter.Parameter(torch.ones((2,1),dtype=torch.double),requires_grad=True)
        self.final_softmax=nn.Softmax(dim=1)
        self.CNN_layer=nn.Conv1d(args.hidden_lstm_dim, args.k ,args.T , padding=args.padd)
        self.relu= torch.nn.ReLU()
        # self.max_pool=torch.nn.MaxPool1d()



        # self.soft_max=F.softmax(1)

    def get_lstm_encoded_data(self,data):
        encoded_posts_comments=[]# will store list of encoded post comments
        hidden_state_post_comments=[]#
        
        for pandc in data:

            post_comments=pandc['embedded_data']
            pres_encoded=[]
            pres_hidden=[]
            # print(len(post_comments))
            encoded_post,(hidden_out,_)=self.post_LSTM(torch.tensor(post_comments[0]))
            # print("enc")
            pres_encoded.append(encoded_post)
            pres_hidden.append(hidden_out)
            # print("enc:",encoded_post.shape)
            # print("hidden:",hidden_out.shape)
            # pres_encoded=torch.cat([pres_encoded,encoded_post]).
            # pres_hidden=torch.cat([pres_hidden,hidden_out])
            past_encoded_comment=hidden_out
            for comm in range(1,len(post_comments)):
                encoded_comment,(hidden_out,_)=self.comment_LSTM(torch.tensor(post_comments[comm]))
                # print("hidden_out:",hidden_out.shape)
                pres_encoded.append(encoded_comment)
                pres_hidden.append(hidden_out)
                # if(torch.equal(past_encoded_comment,hidden_out)):
                #     print("same")
                # else:
                #     print("not")
                past_encoded_comment=hidden_out
                # pres_comment=torch.cat([pres_encoded,encoded_comment])
                # pres_hidden=torch.cat([pres_hidden,hidden_out])
            encoded_posts_comments.append(torch.cat([x for x in pres_encoded]))
            hidden_state_post_comments.append(torch.cat([x for x in pres_hidden]))
        # if torch.equal(encoded_posts_comments[0],encoded_posts_comments[1]):
        #     print("same")
        # else:
        #     print("not same")
        
            # print(encoded_posts_comments[0].shape)
            
            # print(pres_hidden)
        return encoded_posts_comments,hidden_state_post_comments
    def forward(self,data):
        
        

        all_encoded_post_comments,all_hidden_states_post_comments=self.get_lstm_encoded_data(data)
        # print("len of enclen(all_encoded_post_comments[0][0][0]))
        # print((all_hidden_states_post_comments[0]))
        all_final_val=[]
        # final_val=torch.rand((1,1),dtype=torch.double)
        for data_row,encoded_post_comments,hidden_states_post_comments in zip(data,all_encoded_post_comments,all_hidden_states_post_comments):
            A=torch.from_numpy(data_row['adj'])
            # print(hidden_states_post_comments.shape)
            # print("encoded:",encoded_post_comments.shape)
            # print("A",A.shape)
            # print("X",X.shape)
            # print("W0",self.W0.shape)
            # torch.mm(A.double(),hidden_states_post_comments.double())
            # torch.mm(hidden_states_post_comments,self.W0)
            H1=self.tanh( torch.mm(A.double(),torch.mm(hidden_states_post_comments.double(),self.W0)))
            H2=self.tanh(torch.mm(A.double(),torch.mm(H1,self.W1)))   #N+1 * M_dash
            # print(H2.shape)
            Xc=H2[1:]
            # print(Xc.shape)
            Q=Xc
            K=Xc
            V=Xc
            Xc_attn=torch.t(torch.mm((Fu.softmax(torch.mm(Q,torch.t(K))/self.d)),V)).double()
            # print("Xc attn:",Xc_attn.shape)
            # print("Wcw :",self.Wcw.shape)
            
            W_dash=torch.t(encoded_post_comments[:args.m]).double()
            # print("W_dash:",W_dash.shape)
            # print("W_dash0:",W_dash[0].shape)
            # print(W_dash.shape)

            C=Xc_attn
            # print(torch.mm(self.Wcw,W_dash).shape)
           
            F=self.tanh(torch.mm(torch.t(C),torch.mm(self.Wcw,W_dash)))
            # print(torch.mm(C,F).shape)
            # print(torch.mm(self.Wc,torch.mm(C,F)).shape)
            Hc=self.tanh((torch.mm(self.Ww,W_dash)+torch.mm(self.Wc,torch.mm(C,F))))
            Hw=self.tanh((torch.mm(self.Wc,C)+torch.mm(self.Ww,torch.mm(W_dash,torch.t(F)))))
            
            # print("Hw",Hw.shape)
            # print("Hc",Hc.shape)
            ac=Fu.softmax(torch.mm(torch.t(self.Whw),Hw))
            aw=Fu.softmax(torch.mm(torch.t(self.Whc),Hc))
            w_bar=0
            # print("aw shape:",aw.shape)
            # print("ac shape:",ac.shape)
            W_dash_trans=torch.t(W_dash)
            # print("Wdash trans:",W_dash_trans.shape)
            for i in range(0,args.m):
                w_bar+=aw[0][i]*W_dash_trans[i]
            # print("C:",C.shape)
            # print("w_bar:",w_bar.shape)
            
            c_bar=0
            C_trans=torch.t(C)
            for i in range(0,C.shape[1]):
                c_bar=ac[0][i]*C_trans[i]
            # print("c_bar:",c_bar.shape)
            # print(c_bar)
            h_global= torch.cat([w_bar,c_bar],dim=0).reshape(1,args.l+args.m_dash)
            #CNN 
            comments_hidden_encoding=hidden_states_post_comments[1:]
            all_local= self.relu(self.CNN_layer(torch.t(comments_hidden_encoding)))
            temp_max_index=len(comments_hidden_encoding)+2*args.padd-args.T+1
            maxp=torch.nn.MaxPool1d(temp_max_index)
            h_local=maxp(all_local)
            # print("h_local:",h_local.shape)
            # print("h_global",h_global.shape)
            h_en= torch.cat([h_global,torch.t(h_local)],dim=1).reshape(1,args.l+args.m_dash+args.k)

            # if(torch.equal(final_val,h_en)):
            #     print("Equal")
            # else:
            #     print("Not equal")
            # final_val=h_en
            # print("h_global:",h_global.shape)
            y_pre=torch.t(torch.mm(self.Wh,torch.t(h_en))+self.bh) 
            # print("y_pres:",y_pre)
            # print("y_pre:",y_pre.shape)
            all_final_val.append(y_pre)

            # print(H1)
        output=torch.stack(all_final_val).squeeze()
        
        # output=self.final_softmax(all_final_val)
        return output

# %%
model=PostCom2DR(args.m,args.n)
criterion=torch.nn.CrossEntropyLoss()
model.load_state_dict(torch.load("model_13_48.pth"))
# params = [p for p in model.parameters() if p.requires_grad]
# print(len(params))
# params=params+[model.W0]
model.eval()
# optimizer = torch.optim.Adam([
#     {'params': params}
# ], lr= 0.005)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

# %%
from tqdm import tqdm
import time
import random
import warnings
warnings.filterwarnings('ignore')


# %%

for pkl in tqdm(range(0,40)):
        train_batch=pd.read_pickle("Test_pkl/df_test_embedded_"+str(pkl)+".pkl")
        # print(train_batch.shape)
        X_train,y_train=pandas_to_dict(train_batch)
        # print("pkl:",pkl)
        # c = list(zip(X_train, y_train))
        # random.shuffle(c)
        y_test_true=[]
        y_test_pred=[]
        # print(len(X_train))
        # X_train, y_train = zip(*c)
        y_train=torch.Tensor(y_train).long()
        start = time.time()
        for row in range(0,len(X_train)-1, int(args.BatchSize/2)):
            # print("row:",row)
            
            # num_batches+=1
            X_train_batch=X_train[row:row+int(args.BatchSize/2)]
            y_train_batch=y_train[row:row+int(args.BatchSize/2)]
            # for i in range(len(X_train_batch)):
            #     print("X_Train_batch:",X_train_batch[i]['label'],"Y_Labele",y_train_batch[i])
            y_cap=model.forward(X_train_batch)
            _, predictions = torch.max(y_cap, 1)
            predictions = predictions.numpy()
            y_test_pred=y_test_pred+list(predictions)
            # print(len(y_test_pred))
            y_test_true=y_test_true+list(np.array(y_train_batch))
        # print(len(y_test_pred))
        np.save("Array/y_test_pred"+str(pkl)+".npy",y_test_pred)
        np.save("Array/y_test_true"+str(pkl)+".npy",y_test_true)
            # print("r",row)
            # break
            # print(type(y_cap),type(y.shape),type(y_train))
            # loss = criterion(y_cap, y_train_batch)
    

# %%
# y_test_pred=torch.cat([x for x in y_test_pred])

# %%
# y_test_pred=torch.stack(y_test_pred)
len(y_test_pred)

# %%
len(y_test_true)

# %%


# %%
# _, predictions = torch.max(y_test_pred, 1)
# predictions = predictions.numpy()

# %%
# y_test_true

# %%
# y_test_pred[:20]

# %%
accuracy = accuracy_score(y_true=np.array(y_test_true), y_pred=np.array(y_test_pred))
print(accuracy)
# y_test=np.array(y_test_pred.detach().numpy())

# %%
report = classification_report(y_true=np.array(y_test_true), y_pred=y_test_pred, target_names=['Real', 'Fake'])
conf_matrix = confusion_matrix(y_true=np.array(y_test_true), y_pred=y_test_pred)

# %%
report


# %%
conf_matrix

# %%



