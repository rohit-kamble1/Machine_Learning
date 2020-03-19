# Machine-Learning-Project
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn
os.chdir("E:\\pandas")


# In[2]:


df=pd.read_csv("winequality-red.csv")
df.head()


# In[3]:


x=df[['pH']]
y=df[['quality']]


# # MLP Classifier

# In[4]:


from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(hidden_layer_sizes=(11,11,11),  max_iter=500)


# In[5]:


clf.fit(x,y)


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[7]:


y_pred=clf.predict(x_train)
y_pred[0]


# In[23]:


wine_quality=clf.predict(([[3]]))
wine_quality


# In[9]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,y_pred))


# In[10]:


from sklearn.metrics import classification_report
print(classification_report(y_train,y_pred))


# # Decison tree Classifier

# In[12]:


from sklearn.tree import DecisionTreeClassifier


# In[13]:


clf=DecisionTreeClassifier()
clf.fit(x,y)


# In[14]:


y_pred=clf.predict(x_train)
y_pred[0]


# In[15]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,y_pred))


# In[16]:


from sklearn.metrics import classification_report
print(classification_report(y_train,y_pred))


# In[22]:


wine_quality=clf.predict(([[3]]))
wine_quality


# # Nearest Neighbour Classifier

# In[32]:


from sklearn.neighbors import KNeighborsClassifier


# In[69]:


clf=KNeighborsClassifier(n_neighbors=24)
clf.fit(x,y)


# In[77]:


y_pred=clf.predict(x_train)
y_pred


# In[71]:


from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_pred)


# # Support Vector 

# In[72]:


from sklearn.svm import SVC


# In[109]:


clf=SVC()
clf.fit(x,y)


# In[110]:


y_pred=clf.predict(x_train)
y_pred


# In[111]:


from sklearn.metrics import accuracy_score


# In[113]:


accuracy_score(y_train, y_pred)


# # SGD Classifier

# In[118]:


from sklearn.linear_model import SGDClassifier
clf=SGDClassifier()
clf.fit(x,y)


# In[119]:


y_pred=clf.predict(x_train)


# In[121]:


from sklearn.metrics import accuracy_score
accuracy_score(y_train,y_pred)


# # Random Forest Classifier

# In[124]:


from sklearn.ensemble import RandomForestClassifier


# In[126]:


clf=RandomForestClassifier()
clf.fit(x,y)


# In[127]:


y_pred=clf.predict(x_train)


# In[129]:


from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_pred)
