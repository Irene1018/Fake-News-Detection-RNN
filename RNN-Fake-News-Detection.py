#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import library
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
#RNN Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
#LSTM Model
from keras.layers.recurrent import LSTM


# In[2]:


#load data
df_train = pd.read_csv("train.csv", "\t",encoding='utf-8',header=(0)) 
df_test = pd.read_csv("test.csv", "\t",encoding='utf-8',header=(0))
df_sub = pd.read_csv("sample_submission.csv",encoding='utf-8',header=(0))


# In[3]:


#check remove of df_train row#1615
df_train.drop(axis=0, index=1615, inplace= True)
df_train = df_train.reset_index(drop=True)


# In[4]:


df_train[1614:1617]


# In[5]:


#set x&y train and test
x_train = df_train['text']
y_train_r = df_train['label']
x_test = df_test['text']
y_test=pd.to_numeric(df_sub['label'])


# In[6]:


print(x_train.shape)
print(y_train_r.shape)
print(x_test.shape)
print(y_test.shape)


# In[7]:


y_train = []
def strtoint(y_r, y_n):
    for i in range(len(y_r)):
        if y_r[i] =='0':
            y_n.append(0)
        if y_r[i] =='1':
            y_n.append(1)

strtoint(y_train_r, y_train)
y_train = np.array(y_train)


# In[8]:


#set stop words
stopwords= text.ENGLISH_STOP_WORDS


# In[9]:


#transform text to vector by Tfidf
vectorizer = TfidfVectorizer(
            norm='l2',                      
            stop_words=stopwords,
            max_features=1800               
            )

X_train = vectorizer.fit_transform(x_train).toarray()
X_test = vectorizer.fit_transform(x_test).toarray()

RNN Model
# In[10]:


#set keras sequential model
modelRNN = Sequential()
modelRNN.add(Embedding(output_dim = 32,
                      input_dim = 2800,
                      input_length = 1800))
modelRNN.add(Dropout(0.2))


# In[11]:


#set keras NN model
modelRNN.add(SimpleRNN(units = 16, return_sequences=True))
modelRNN.add(Dense(units = 256, activation = 'relu'))
modelRNN.add(Dropout(0.35))

modelRNN.add(Dense(units = 1, activation = 'sigmoid'))


# In[12]:


#model summary
modelRNN.summary()


# In[13]:


#define training model
modelRNN.compile(loss = 'binary_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

train_history = modelRNN.fit(X_train, y_train,
                            epochs = 10,
                            batch_size = 100,
                            verbose = 2)


# In[14]:


#evaluate model with test data
scores = modelRNN.evaluate(X_test, y_test, verbose = 1)


# LSTM Model

# In[18]:


#set keras sequential model
modelLSTM = Sequential()
modelLSTM.add(Embedding(output_dim = 32,
                      input_dim = 2000,
                      input_length = 1800))
modelRNN.add(Dropout(0.2))


# In[19]:


#set keras NN model
modelLSTM.add(LSTM(units = 32, return_sequences=True))
modelLSTM.add(Dense(units = 256, activation = 'relu'))
modelLSTM.add(Dropout(0.35))
#output layer
modelLSTM.add(Dense(units = 1, activation = 'sigmoid'))


# In[20]:


#model summary
modelLSTM.summary()


# In[21]:


#define training model
modelLSTM.compile(loss = 'binary_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

train_history = modelLSTM.fit(X_train, y_train,
                            epochs = 10,
                            batch_size = 100,
                            verbose = 2)


# In[22]:


#evaluate model with test data
scores = modelLSTM.evaluate(X_test, y_test, verbose = 1)


# In[ ]:




