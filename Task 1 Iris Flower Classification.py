#!/usr/bin/env python
# coding: utf-8

# ## Name: Shubham Vilas Gaiwkad

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


df=pd.read_csv("Iris.csv")


# In[24]:


df


# In[25]:


df=df.drop(columns=["Id"])


# In[26]:


df


# In[27]:


df.describe()


# In[28]:


df.info()


# In[29]:


df['Species'].value_counts()


# ## Preprocessing of dataset

# In[30]:


df.isnull()


# In[31]:


df.isnull().sum()


# 
# ## Exploratory Data Analysis

# In[32]:


df['SepalLengthCm'].hist()


# In[33]:


df['SepalWidthCm'].hist()


# In[34]:


df['PetalLengthCm'].hist()


# In[35]:


df['PetalWidthCm'].hist()


# In[36]:


colors=['red','orange','blue']
species=['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[37]:


for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'], c=colors[i], label=species[i])
plt.xlabel("SepalLength")
plt.ylabel("SepalWidth")
plt.legend()


# In[38]:


for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'], c=colors[i], label=species[i])
plt.xlabel("PetalLength")
plt.ylabel("PetalWidth")
plt.legend()


# In[39]:


for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalLengthCm'], c=colors[i], label=species[i])
plt.xlabel("SepalLength")
plt.ylabel("PetalLength")
plt.legend()


# In[40]:


for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['SepalWidthCm'],x['PetalWidthCm'], c=colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# ## Correlation Matrix

# In[41]:


df.corr()


# In[42]:


corr=df.corr()
fig, ax=plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True,ax=ax, cmap='coolwarm')


# ## ModelTraining

# In[43]:


from sklearn.model_selection import train_test_split

X=df.drop(columns=['Species'])
Y=df['Species']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)


# ## Model 1-Logistic Regression
# 

# In[44]:


from sklearn.linear_model import LogisticRegression
model_LR=LogisticRegression()


# In[45]:


model_LR.fit(x_train,y_train)


# In[46]:


model_LR.predict(x_test)


# In[47]:


y_test


# In[48]:


print("Accuracy of Logistic Regression: ",model_LR.score(x_test,y_test)*100)


# ## Model 2- K-Nearest Neighbors 

# In[49]:


from sklearn.neighbors import KNeighborsClassifier
model_KNN= KNeighborsClassifier()


# In[50]:


model_KNN.fit(x_train,y_train)


# In[51]:


model_KNN.predict(x_test)


# In[52]:


y_test


# In[53]:


print("Accuracy of K-Nearest Neighbour: ",model_KNN.score(x_test,y_test)*100)


# ## Model 3.Support Vector Machine

# In[54]:


from sklearn.svm import SVC
model_SVC= SVC()


# In[55]:


model_SVC.fit(x_train,y_train)


# In[56]:


model_SVC.predict(x_test)


# In[57]:


y_test


# In[58]:


print("Accuracy of Support Vector Machine: ",model_SVC.score(x_test,y_test)*100)


# ## New data for Prediction

# In[59]:


X_new=np.array([[3,2,1,0.2],[4.9,2.2,3.8,1.1],[5.3,2.5,4.6,1.9]])


# In[60]:


model_SVC.predict(X_new)


# In[61]:


model_LR.predict(X_new)


# In[62]:


model_KNN.predict(X_new)


# In[ ]:





# In[ ]:





# In[ ]:




