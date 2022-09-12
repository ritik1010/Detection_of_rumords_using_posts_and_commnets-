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
import gensim

# %%

class args:
    m=200 #Num of  words in tweets
    n=200#Max comments in posts and comments
    d=300#dimension of vector after word2vec
    hidden_lstm_dim=80
    l=80#lstm dimensionality
    m_dash=80
    T=3 #num of comments being compared to calculate topic drift
    k=80 #number of filters
    padd=1#padding size
    BatchSize=8 #batch size

# %%
def read_comments(comments_dict, adj_list,parent_ind,pres_ind,all_comments_list):
    global list_of_all_sentences
    if type(comments_dict)==type(None):
        return all_comments_list,adj_list
    if len(adj_list) > args.n or len(comments_dict['text'])==0:
        return all_comments_list,adj_list
    # print(parent_ind, pres_ind)
    adj_list[parent_ind].append(pres_ind)
    adj_list[pres_ind]=[]
    adj_list[pres_ind].append(parent_ind)

    all_comments_list.append(comments_dict['text'].split(" "))
    
    if 'engagement' in comments_dict.keys() and 'tweet_replies' in comments_dict['engagement'].keys():
        for pres_comment_dict in comments_dict['engagement']['tweet_replies']:
            all_comments_list,adj_list= read_comments(pres_comment_dict,adj_list,pres_ind,len(adj_list),all_comments_list)
    return all_comments_list,adj_list
    


# %%


# %%
list_of_all_sentences=[]

# %%
def get_adj_matrix(data):
    n=len(data['embeded_data'])
    # print("n:",n)
    adj_mat=np.zeros((n,n))
    adj_list=data['adj']
    # print(adj_list)
    for fro  in range(n):
        
        adj_mat[fro][fro]=1
        pres_list=adj_list[fro]
        # print(pres_list)
        for to in pres_list:
            # print("from:",fro,"to:",to)
            adj_mat[fro][to]=1
            
    return adj_mat
        

# %%
word_2_vec_model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz",limit=100000, binary=True)

# %%
vocab=list(word_2_vec_model.key_to_index.keys())

# %%

# vocab[:5]

# %%
def perform_word_2_vec(tweet):
    # tweet=tweet.split(" ")
    if(len(tweet))>args.m:
        tweet=tweet[:args.m]
    else:
        pad=["o"]*(args.m-len(tweet))
        # print(tweet)
        # print(pad)
        tweet.append(pad)
        
    encoded_sentence=[]
    for word in tweet:
        if word in  vocab:
            encoded_sentence.append(word_2_vec_model[word])
        else:
            encoded_sentence.append([0]*300)

    return encoded_sentence

def get_post_comments_encoding(data):
    # print(data)
    # print(len(data))
    # print("starting encoding")
    post_comments_list=[]
    # return "h"
    print(type(data))
    # print(data[0])
    for tweets in data:
        # print("getting encoded")
        # print("tweet",tweets)
        post_comments_list.append((tweets))
    # print(post_comments_list)
    return post_comments_list



# %%
def get_list_of_data(tweet_json,replies_json,label,pres_all_post_comments_list,pres_adj_list):
    #helper function to read all tweets and comments for the given news article
    # res=[]
    tweet_list=tweet_json['tweets']#get list of all tweets for given news article
    for tweet in tweet_list:
        #iterate over all tweets for the post
        # pres_dict={}
        tweet_text=tweet['text'].split(" ")
        # print("tt:",tweet_text)
        tweet_id=tweet['tweet_id']
        reply_list=replies_json[str(tweet_id)]#get list of replies for given tweet id
        pres_adj_list[len(pres_all_post_comments_list)]=[0]
        pres_adj_list[0].append(len(pres_all_post_comments_list))
        # all_comments_list=[]
        # pres_adj_list=[]
        # pres_all_comments_list=[]
    
        pres_all_post_comments_list.append( tweet_text)
        # print("appended in list")
        # if len(reply_list)==0:
        #     # print("rl becoomes 0")
        #     continue
        # print("rl",reply_list)
        # print("sTARINTN LOOP")
        for comments_dict in reply_list:
            pres_all_post_comments_list,pres_adj_list=read_comments(comments_dict,pres_adj_list,len(pres_adj_list)-1,len(pres_adj_list),pres_all_post_comments_list)
        # print(pres_all_comments_list)
        # if len(pres_all_post_comments_list)==1:
        #     continue
        
        
        # break
    return pres_all_post_comments_list,pres_adj_list
            

            
            
        # break



# %%
def perform_bfs(pres_all_post_comments_list,pres_adj_list,new_pres_all_post_comments_list,new_pres_adj_list,old_ind_list,old_to_new_ind_dict,parent_ind_list):
    if len(new_pres_all_post_comments_list)>= args.m:
        return new_pres_all_post_comments_list,new_pres_adj_list,old_to_new_ind_dict
    #add this node to the new graph
    # print("type:",type(pres_adj_list))
    next_old_ind_list=[]
    next_parent_ind_list=[]
    # print("1len:",len(new_pres_all_post_comments_list), " ",len(new_pres_adj_list))
    # print(old_ind_list,parent_ind_list)
    for old_ind,parent_ind in zip(old_ind_list,parent_ind_list):
        if len(new_pres_all_post_comments_list)>= args.m:
            return new_pres_all_post_comments_list,new_pres_adj_list,old_to_new_ind_dict
        if old_ind in old_to_new_ind_dict.keys():
            continue

        new_pres_all_post_comments_list.append(pres_all_post_comments_list[old_ind])
        old_to_new_ind_dict[old_ind]=len(new_pres_all_post_comments_list)-1
        new_pres_adj_list[old_to_new_ind_dict[old_ind]]=[parent_ind]
        new_pres_adj_list[parent_ind].append(old_to_new_ind_dict[old_ind])
        list_of_old_neighbors=pres_adj_list[old_ind]
        next_old_ind_list=next_old_ind_list+list_of_old_neighbors
        next_parent_ind_list+=[old_ind]*len(list_of_old_neighbors)
        # print("1len:",len(new_pres_all_post_comments_list), " ",len(new_pres_adj_list))
        
        # print(next_old_ind_list,next_parent_ind_list)
    # print("1len:",len(new_pres_all_post_comments_list), " ",len(new_pres_adj_list))
    new_pres_all_post_comments_list,new_pres_adj_list,old_to_new_ind_dict=perform_bfs(pres_all_post_comments_list,pres_adj_list,new_pres_all_post_comments_list,new_pres_adj_list,next_old_ind_list,old_to_new_ind_dict,next_parent_ind_list)
    return new_pres_all_post_comments_list,new_pres_adj_list,old_to_new_ind_dict

                



    



def get_top_n_comments(pres_all_post_comments_list,pres_adj_list):
    new_pres_all_post_comments_list=[]
    new_pres_adj_list={}
    old_to_new_ind_dict={}
    new_pres_adj_list[0]=[]
    new_pres_all_post_comments_list,new_pres_adj_list,old_to_new_ind_dict=perform_bfs(pres_all_post_comments_list,pres_adj_list,new_pres_all_post_comments_list,new_pres_adj_list,[0],old_to_new_ind_dict,[0])
    # print(old_to_new_ind_dict)
    return new_pres_all_post_comments_list,new_pres_adj_list,old_to_new_ind_dict

# %%
# list_of_all_init_data=[]
def read_folder_save_csv(path,df_name,label):
    #helper function to iterate over all news article of subfolder 
    global list_of_all_sentences
    list_of_all_init_data=[]
    subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
    for folder in subfolders:
        tweets_json=[]
        likes_json=[]
        try:
            with open(os.path.join(folder,"tweets.json")) as tfile:
                with  open(os.path.join(folder,"replies.json")) as lfile:
                    pres_news_text=""
                    with open(os.path.join(folder,"news_article.json")) as nfile:
                        news_json=json.load(nfile)
                        pres_news_text=news_json['text'].split(" ")
                        # list_of_all_sentences.append(pres_news_text)


                    tweets_json=json.load(tfile)
                    replies_json=json.load(lfile)
                    pres_dict={}#initalize empty dictionary for present row
                    pres_all_post_comments_list=[]
                    pres_all_post_comments_list.append(pres_news_text)#insert news article as 0th element in the list 
                    pres_adj_list={}#init empty adjacency list 
                    pres_adj_list[0]=[]#init 0th adj list empty
                    # print(tweets_json)pres_adj_matrix=get_adj_matrix(pres_adj_list)
                    pres_all_post_comments_list,pres_adj_list=get_list_of_data(tweets_json,replies_json,label,pres_all_post_comments_list,pres_adj_list)
                    pres_dict['embeded_data']=pres_all_post_comments_list
                    pres_dict['adj']=pres_adj_list
                    pres_dict['label']=label
                    # pres_adj_matrix=get_adj_matrix(pres_dict)
                    
                    # print(pres_dict['embeded_data'])
                    # print(len(pres_dict['embeded_data']))
                   
                    if args.n< len(pres_dict['embeded_data']):
                        # print(len(pres_all_post_comments_list),len(pres_adj_list))
                        pres_all_post_comments_list,pres_adj_list,old_to_new_ind=get_top_n_comments(pres_all_post_comments_list,pres_adj_list)
                        pres_dict['embeded_data']=pres_all_post_comments_list
                       
                        pres_dict['adj']=pres_adj_list
                    pres_dict['adj']=get_adj_matrix( pres_dict)
                    if  len(pres_dict['embeded_data'])>3:
                        list_of_all_init_data.append(pres_dict)



                        
                        # print("fin")
                        # print(len(pres_all_post_comments_list),len(pres_adj_list))
                        # break
                    #  list_of_all_sentences+=pres_all_post_comments_list
                    
                    # print(pres_dict)
                    # break
                # break
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print (message,"folder :",folder)
    # print(list_of_all_init_data)
    df=pd.DataFrame(list_of_all_init_data)
    print("Saving data...")
    df.to_pickle(df_name)
    return df
    # break
    

# %%


# %%
df_f=read_folder_save_csv("FakeNewsNet_Dataset/gossipcop_fake","gossipcop_fake.pkl",0)

# %%
df_f.shape

# %%
df_f.tail

# %%
len((df_f.iloc[0,:]['embeded_data']))

# %%
df_r=read_folder_save_csv("FakeNewsNet_Dataset/gossipcop_real","gossipcop_real.pkl",1)

# %%
df_f1=read_folder_save_csv("FakeNewsNet_Dataset/politifact_real","politifact_real.pkl",1)

# %%
df_r1=read_folder_save_csv("FakeNewsNet_Dataset/politifact_fake","politifact_fake.pkl",0)

# %%
df=pd.concat([df_f,df_r,df_f1,df_r1],ignore_index=True)

# %%
del df_f
del df_r
del df_f1
del df_r1

# %%


# %%
# from gensim.models import Word2Vec
# word2vec_custom = Word2Vec(list_of_all_sentences, min_count=1,vector_size=300)

# %%
# len(word2vec_custom.wv.key_to_index)

# %%
# len(word2vec_custom.wv['the'])

# %%
y=df['label']

# %%
from sklearn.model_selection import train_test_split
y=df['label']
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=.2, random_state=0, shuffle=True,
    stratify=y
)

# %%
X_train.to_pickle("train.pkl")
X_test.to_pickle("test.pkl")

# %%
df.head(1)['embeded_data'][0]

# %%
# X_train=pd.read_pickle("train.pkl")
X_test=pd.read_pickle("test.pkl")

# %%
# vocab=word2vec_custom.wv.key_to_index.keys()

# %%
from csv import writer
from tqdm import tqdm
import pandas as pd

# %%
def perform_word_2_vec(tweet):
    # tweet=tweet.split(" ")
    if(len(tweet))>args.m:
        tweet=tweet[:args.m]
    else:
        pad=["o"]*(args.m-len(tweet))
        # print(tweet)
        # print(pad)
        tweet=tweet+pad
        # print(tweet)
        
    encoded_sentence=[]
    # print("padded:",tweet)
    for word in tweet:
        if word in  vocab:
            # print(word)
            encoded_sentence.append(word_2_vec_model[word])
            # print(encoded_sentence[-1].shape)
        else:
            encoded_sentence.append([0]*args.d)

    return encoded_sentence

def get_post_comments_encoding(data):
    # print(data)
    # print(len(data))
    post_comments_list=[]
    # return "h"
    # print(type(data))
    # print(data[0])
    for tweets in data:
        # print("getting encoded")
        # print("tweet",tweets)
        post_comments_list.append(perform_word_2_vec(tweets))
    # print(post_comments_list)
    return post_comments_list



# %%
len(X_train)

# %%
X_train['label'].value_counts()

# %%
X_train['word2vec']=""

# %%
import time
header_req=True
batch_size=100
start=0
start_time=time.time()
batch_num=0
# open('df_train_embedded.pkl', 'w').close()
for ind in range(start,len(X_train),batch_size):
    # print("start")
    # print(chunks)
    # X_train.at[ind,'word2vec']=get_post_comments_encoding(chunks['
    # print(len(temp))
    # print(len(temp[0]))
    # print("h")
    # print(chunks['embeded_data'])
    # print(X_train.index(ind))
    
    row_df=pd.DataFrame(X_train.iloc[ind:ind+batch_size,],columns=X_train.columns)
    # print(row_df)
    # print(row_df.columns)
    row_df['embedded_data']=row_df['embeded_data'].apply(lambda x:  get_post_comments_encoding(x) )
    
    row_df.to_pickle('Train_pkl/df_train_embedded_'+str(batch_num)+'.pkl')
    batch_num=batch_num+1
    
    # header_req=False
    # with open('df_train_embeded.csv', 'a') as f_object:
  
    #     # Pass this file object to csv.writer()
    #     # and get a writer object
    #     writer_object = writer(f_object)
    
    #     # Pass the list as an argument into
    #     # the writerow()
    #     writer_object.writerow(chunks)
    
    #     #Close the file object
    #     f_object.close()
    
    print("Time elapsed at index {} is {}".format(ind+batch_size,(time.time()-start_time)/60))
    # if batch_num==10:
    #  break

    

# %%
X_test.shape

# %%
import time
header_req=True
batch_size=100
start=0
start_time=time.time()
batch_num=32
# open('df_train_embedded.pkl', 'w').close()
# for ind in range(32,len(X_test),batch_size):
for ind in range(3200,3500,batch_size):
    # print("start")
    # print(chunks)
    # X_train.at[ind,'word2vec']=get_post_comments_encoding(chunks['
    # print(len(temp))
    # print(len(temp[0]))
    # print("h")
    # print(chunks['embeded_data'])
    # print(X_train.index(ind))
    
    row_df=pd.DataFrame(X_test.iloc[ind:ind+batch_size,],columns=X_test.columns)
    # print(row_df)
    # print(row_df.columns)
    row_df['embedded_data']=row_df['embeded_data'].apply(lambda x:  get_post_comments_encoding(x) )
    
    row_df.to_pickle('Test_pkl/df_test_embedded_'+str(batch_num)+'.pkl')
    batch_num=batch_num+1
    
    # header_req=False
    # with open('df_train_embeded.csv', 'a') as f_object:
  
    #     # Pass this file object to csv.writer()
    #     # and get a writer object
    #     writer_object = writer(f_object)
    
    #     # Pass the list as an argument into
    #     # the writerow()
    #     writer_object.writerow(chunks)
    
    #     #Close the file object
    #     f_object.close()
    
    print("Time elapsed at index {} is {}".format(ind+batch_size,(time.time()-start_time)/60))
    # if batch_num==10:
    #  break

    

# %%
X_train.dtypes

# %%
temp_df.dtypes

# %%
temp_df.head()

# %%


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
        if label=='true':
            intlabel=1
        else:
            # print("int label becomes 0")
            intlabel=0
        y.append(intlabel)
        all_dict_list.append(pres_dict)
    return all_dict_list,y


# %%
x,y=pandas_to_dict(temp_df)

# %%
len(x)

# %%
len(x[0]['embedded_data'][0][0])

# %%
df_temp['embeded_data']

# %%
df_temp.columns

# %%
res=df_temp['embeded_data'].apply(lambda x:get_post_comments_encoding(x))

# %%
len(res[0][0][0])

# %%
# df['word2vec']=df.swifter.apply(lambda x: get_post_comments_encoding(x['embeded_data']))

# %%
X_train['word2vec']=""
X_train['word2vec']

# %%
X_train.shape

# %%


# %%
X_train.to_pickle("final_train.pkl")

# %%
# del df_train

# %%
# df_test=pd.read_csv("test.csv")
word_2_vec_model['hi'].shape

# %%
df_embeded=pd.read_csv("df_train_embedded.csv")

# %%
def pandas_to_dict(input_dataframe):
    all_dict_list=[]
    y=[]
    for emb_data,adj,label in zip(input_dataframe['embeded_data'],input_dataframe['adj'],input_dataframe['label']):
        pres_dict={}
        pres_dict['embeded_data']=emb_data
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
model.train()
params = [p for p in model.parameters() if p.requires_grad]
print(len(params))
params=params+[model.W0]

optimizer = torch.optim.Adam([
    {'params': params}
], lr= 0.005)

y_train=torch.from_numpy(np.array(y_train)).long()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

# %%
print("X_train ",len(X_train)," X_test:",len(X_test))

# %%
from tqdm import tqdm
import time
import random
total_rows=len(X_train)
# total_batch

for epoch in (range(20)):
    loss_this_epoch=0
    num_batches=0
    c = list(zip(X_train, y_train))
    random.shuffle(c)

    X_train, y_train = zip(*c)
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
    scheduler.step()
    end = time.time()

    print('epoch {}, loss {} time taken this epoch {}'.format(epoch, loss_this_epoch/num_batches))



