#!/usr/bin/env python
# coding: utf-8

# In[132]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("E:\\pandas")


# In[133]:


df=pd.read_csv("diabetes.csv")
df.head()


# In[134]:


import sklearn
from sklearn.neural_network import MLPClassifier


# In[135]:


x=df[["Glucose"]]
y=df[["Outcome"]]


# In[136]:


from sklearn.model_selection import train_test_split


# In[137]:


x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3)


# In[138]:


x_train.shape


# In[139]:


y_train.shape


# In[140]:


clf=MLPClassifier(hidden_layer_sizes=(10),solver="lbfgs", alpha=1e-5)


# In[141]:


clf.fit(x,y)


# In[142]:


y_pred=clf.predict(x_train)
y_pred[1]


# In[143]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,y_pred))


# In[144]:


from sklearn.metrics import classification_report
print(classification_report(y_train,y_pred))


# In[145]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train,y_pred))


# In[146]:


a=clf.predict(([[185]]))
print(a)


# In[147]:


if a==0:
    print("You are diabetic free")
else:
    print("You have diabetic, concern to doctor")


# In[148]:


y_pred=clf.predict(x_test)


# In[149]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


# In[ ]:




