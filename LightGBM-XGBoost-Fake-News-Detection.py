#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import library
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import text
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


#load datasets
df_train = pd.read_csv("train.csv", "\t",encoding='utf-8',header=(0))
df_test = pd.read_csv("test.csv", "\t",encoding='utf-8',header=(0))
df_sub = pd.read_csv("sample_submission.csv",encoding='utf-8',header=(0))
df_train


# In[3]:


#set x&y train and test
x_train = df_train['text']
y_train = df_train['label'].tolist()
x_test = df_test['text']
y_test=pd.to_numeric(df_sub['label']).tolist()


# In[4]:


#set stop words
stopwords= text.ENGLISH_STOP_WORDS


# In[5]:


#transform text to vector by Tfidf
vectorizer = TfidfVectorizer(
            norm='l2',                      
            stop_words=stopwords,
            max_features=1800               
            )

X_train = vectorizer.fit_transform(x_train).toarray()
X_test = vectorizer.fit_transform(x_test).toarray()


# In[6]:


#applying Xgboost model

#set paramaters
XGB_Classfier = xgb.XGBClassifier(learning_rate=0.5,                   
                              n_estimators=100,         
                              max_depth=6,                  
                              gamma=5,                               
                              objective='binary:logistic',
                              random_state=99            
                              )
#training model
XGB_Classfier = XGB_Classfier.fit(X_train, y_train)
#predicting
Xgb_pred = XGB_Classfier.predict(X_test).astype(int)


# In[7]:


#reviewing model performance
Xgb_accuracy = accuracy_score(y_test, Xgb_pred)
Xgb_precision = metrics.precision_score(y_test, Xgb_pred)
Xgb_recall = metrics.recall_score(y_test, Xgb_pred)
Xgb_F_measure = metrics.f1_score(y_test, Xgb_pred)

print("Accuracy: %f" % Xgb_accuracy)
print("Precision: %f" % Xgb_precision)
print("Recall: %f" % Xgb_recall)
print("F_measure: %f" % Xgb_F_measure)


# In[8]:


XGBC_report = classification_report(y_test, Xgb_pred)
print(XGBC_report)


# In[18]:


#applying LightGBM model

#set paramaters
LGB_Classifier = lgb.LGBMClassifier( 
                      learning_rate=0.5, 
                      num_leaves=50,
                      n_estimators=120,
                      max_bin=200,
                      random_state=99,          
                      device='cpu'
                      )
#training model
LGB_Classfier = LGB_Classifier.fit(X_train, y_train)
#predicting
Lgb_pred = LGB_Classfier.predict(X_test).astype(int)


# In[19]:


#reviewing model performance
Lgb_accuracy = accuracy_score(y_test, Lgb_pred)
Lgb_precision = metrics.precision_score(y_test, Lgb_pred)
Lgb_recall = metrics.recall_score(y_test, Lgb_pred)
Lgb_F_measure = metrics.f1_score(y_test, Lgb_pred)

print("Accuracy: %f" % Lgb_accuracy)
print("Precision: %f" % Lgb_precision)
print("Recall: %f" % Lgb_recall)
print("F_measure: %f" % Lgb_F_measure)


# In[20]:


LGBC_report = classification_report(y_test, Lgb_pred)
print(LGBC_report)


# In[ ]:




