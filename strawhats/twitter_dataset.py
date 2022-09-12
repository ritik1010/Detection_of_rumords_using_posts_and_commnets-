# %%
import os
import numpy as np
# import os
import json
import gensim
# from ast import arg
import numpy as np
import torch
import torch.nn as nn
# import config
# import utils
import math
import torch.nn.functional as Fu
from numpy.linalg import norm
# from torch.autograd import Variable
# import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
# from config import args
# from postcom2dr import PostCom2DR


# %%
from args import args
from model import PostCom2DR

# %%
id_array=[]
id_to_label={}
tweet_id_to_tweet={}
label_path="twitter15/label.txt"
with open(label_path) as labelfile:
    #reading labels for all the files and mapping it to tweet id
    for line in labelfile:
        pres_arr=line.split(":")
        id_array.append(pres_arr[1].strip('\n'))
        id_to_label[pres_arr[1].strip('\n')]=pres_arr[0]



# %%
tweet_folder_path="userdata15_reply"
for tweet_id in id_array:
    #reading all the replies related to 
    tweet_path=os.path.join(tweet_folder_path,tweet_id+".json",)
    tweet_file=open(tweet_path)
    file_data=json.load(tweet_file)
    for key in file_data:
        if file_data[key]!="Failed":
            tweet_id_to_tweet[key]=file_data[key]
# print(len(tweet_id_to_tweet))


# %%
tweet_id_to_comment_dict={}
userdata_15_reply_path="userdata15_reply"
list_of_files=os.listdir(userdata_15_reply_path)
# print(list_of_files)
post_to_comments={}
num_comments_in_post={}
for file in list_of_files:
    with open(os.path.join(userdata_15_reply_path,file)) as pres_file:
        json_form=json.load(pres_file)
        # print(json_form)
        post=0
        pres_post=0
        num_comments_read=0
        for k in json_form.keys():
            tweet_id_to_tweet[k]=json_form[k]
            if num_comments_read>args.n or ( post==0 and json_form[k]=="Failed"):
                break
            if post==0 :
                num_comments_read=0
                pres_post=k
                post=1
                post_to_comments[pres_post]=[]
            elif json_form[k]!="Failed":
                num_comments_read+=1
                post_to_comments[pres_post].append(k)
                num_comments_in_post[k]=num_comments_read

        
            

        


# %%
#loading google word2vec model 
word_2_vec_model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz",limit=100000, binary=True)

        



# %%
vocab=word_2_vec_model.key_to_index.keys()

# %%
def perform_word_2_vec(tweet_id):
    #helper function to perform word2vec
    tweet=tweet_id_to_tweet[tweet_id]
    tweet=tweet.split(" ")
    if(len(tweet))>args.m:
        tweet=tweet[:args.m]
    else:
        pad=["o"]*(args.m-len(tweet))
        tweet=tweet+pad
        
    encoded_sentence=[]
    for word in tweet:
        if word in  vocab:
            encoded_sentence.append(word_2_vec_model[word])
        else:
            encoded_sentence.append([0]*300)

    return encoded_sentence


# %%

    
encoded_posts_and_comments={}

for k in list(post_to_comments.keys()):
    pres_adj=[]

    pres_dict={}
    pres_dict["post"]=perform_word_2_vec(k)
    pres_dict["comment"]=[]
    comments_list=[]
    for comment in post_to_comments[k]:
        pres_dict["comment"].append(perform_word_2_vec(comment))
    encoded_posts_and_comments[k]=pres_dict


# %%
def get_adj_matrix_for_post_comments(post_comment_dict):
    #returns adjacency matrix for comments
    num_of_comments=len(post_comment_dict["comment"])
    adj_mat=np.zeros((num_of_comments+1,num_of_comments+1))
    for row in range(0,num_of_comments+1):
        for col in range(0,num_of_comments+1):
            if row==col or row==0 or col==0:
                adj_mat[row][col]=1
    return adj_mat
            

# %%
all_encoded_data=[]# will contain list of dictionary for all the data
y=[]
for id in list(id_to_label.keys())[:]:
    pres_dic={}
    # print(str(id))
    if id in encoded_posts_and_comments.keys():
        embeded_data=[]
        embeded_data.append(encoded_posts_and_comments[id]['post'])
        for comments in encoded_posts_and_comments[id]['comment']:
            embeded_data.append(comments)
        if len(embeded_data)==1:
            continue
        pres_dic["embedded_data"]=embeded_data
        pres_dic["label"]=id_to_label[id]
        pres_dic["adj"]=get_adj_matrix_for_post_comments(encoded_posts_and_comments[id])
        all_encoded_data.append(pres_dic)
        if id_to_label[id]:
            y.append(1)
        else:
            y.append(0)
    # else:
    #     print("not there")


        

# %%
import pandas as pd
df=pd.DataFrame(all_encoded_data)#making dataframe from all the data

# %%
from sklearn.model_selection import train_test_split
y=df['label']
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=.2, random_state=0, shuffle=True,
    stratify=y
)

# %%
print("Number of rows for training:",len(X_train)," Number of rows for testing:", len(X_test))

# %%
def pandas_to_dict(input_dataframe):
    #helper function to convert pandas datafram into list of dictionary
    all_dict_list=[]
    y=[]
    for emb_data,adj,label in zip(input_dataframe['embedded_data'],input_dataframe['adj'],input_dataframe['label']):
        pres_dict={}
        pres_dict['embedded_data']=emb_data
        pres_dict['adj']=adj
        pres_dict['label']=label
        intlabel=0
        if label=='true':
            intlabel=1
        else:
            # print("int label becomes 0")
            intlabel=0
        y.append(intlabel)
        all_dict_list.append(pres_dict)
    return all_dict_list,y


# %%
X_train,y_train=pandas_to_dict(X_train)
X_test,y_test=pandas_to_dict(X_test)

# %%
model=PostCom2DR(args.m,args.n)
criterion=torch.nn.CrossEntropyLoss()
model.train()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam([
    {'params': params}
], lr= 0.005)

y_train=torch.from_numpy(np.array(y_train)).long()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)


# %%
# from tqdm import tqdm
import random
total_rows=len(X_train)
for epoch in (range(20)):
    loss_this_epoch=0
    num_batches=0
    #shuffling X_Train and y_train to create random batches
    c = list(zip(X_train, y_train))
    random.shuffle(c)

    X_train, y_train = zip(*c)
    y_train=torch.Tensor(y_train).long()
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
    scheduler.step()
    print('epoch {}, loss {}'.format(epoch, loss_this_epoch/num_batches))



# %%
y_test=torch.from_numpy(np.array(y_test)).long()
y_test_cap=model.forward(X_test)
loss = criterion(y_test_cap, y_test)


# %%
_, predictions = torch.max(y_test_cap, 1)
predictions = predictions.numpy()

# %%
accuracy = accuracy_score(y_true=np.array(y_test), y_pred=predictions)
print("Accuracy after training is:",accuracy)
y_test=np.array(y_test)

# %%
report = classification_report(y_true=np.array(y_test), y_pred=predictions, target_names=['Real', 'Fake'])
conf_matrix = confusion_matrix(y_true=np.array(y_test), y_pred=predictions)

# %%
report

# %%
print("Confusion matrix:",conf_matrix)

# %%
torch.save(model, "model_lr0.005_step4_gamma0.5_acc88.pkl")

# %%



