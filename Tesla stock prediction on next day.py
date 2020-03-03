#!/usr/bin/env python
# coding: utf-8

# In[574]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
os.chdir("E:\\pandas")


# In[575]:


df=pd.read_csv("TSLA.csv")
df.head()


# In[576]:


x=df[["Open"]]
y=df[["Close"]]
y.shape


# In[577]:


plt.scatter(x,y)
plt.xlabel("Open Price in US $")
plt.ylabel("Closing Price in US $")
plt.title("Tesla stock data")
plt.xlim(200,240)
plt.ylim(200,240)


# In[578]:


from sklearn.tree import DecisionTreeRegressor


# In[579]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.32)


# In[580]:


y_train.shape


# In[581]:


tree=DecisionTreeRegressor()
tree.fit(x, y)


# In[582]:


y_pred=tree.predict(x_train)
y_pred


# In[583]:


plt.scatter(x,y,color='green')
plt.scatter(x_train, y_pred, color="red")
plt.xlim(100,200)
plt.ylim(100,200)
plt.xlabel("Open Price in US $")
plt.ylabel("Closing Price in US $")
plt.title("Tesla stock data")


# In[584]:


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_train, y_pred))


# In[585]:


from sklearn.metrics import r2_score
print(r2_score(y_train, y_pred))


# In[586]:


tree.predict(([[140]]))


# In[587]:


y_pred=tree.predict(x_test)


# # Test data

# In[588]:


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))


# In[589]:


from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))


# In[590]:


tree.predict(([[140]]))

