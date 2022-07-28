#!/usr/bin/env python
# coding: utf-8

# # import the libraries

# In[13]:


import warnings
warnings.simplefilter("ignore")


# # import pandas and numpy

# In[14]:


import pandas as pd 
import numpy as np


# # import  data visualization library

# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # import the dataset

# In[16]:


dataset = pd.read_csv('Admission_Predict.csv')


# In[21]:


dataset


# # understanding the dataset

# In[22]:


dataset.shape


# In[23]:


dataset.head()


# # slicing the dataset-removing the columns

# In[24]:


dataset = dataset.drop(['Serial No.','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research'],axis = 1)


# In[25]:


dataset


# # reashaping

# In[26]:


x = dataset.iloc[:,0]


# In[27]:


x.shape


# In[28]:


x = dataset.iloc[:,0].values.reshape(-1,1)


# In[29]:


x.shape


# In[30]:


x


# In[31]:


y = dataset.iloc[:,-1].values.reshape(-1,1)


# In[32]:


y.shape


# In[33]:


y


# # scatter plot

# In[34]:


plt.scatter(x,y)
plt.show()


# # dividing dataset into training set and test set

# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


x_train,x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)


# In[38]:


x_train.shape


# In[39]:


x_test.shape


# In[40]:


y_train.shape


# In[41]:


y_test.shape


# # perform the linear regression

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[43]:


lm = LinearRegression()


# In[45]:


lm.fit(x_train,y_train)


# # predict the chances of admit

# In[46]:


y_pred = lm.predict(x_test)


# In[51]:


check = pd.DataFrame(x_test,columns = ['GRE Score'])


# In[48]:


check


# In[52]:


check["COA Actual"]=y_test


# In[53]:


check


# In[54]:


check['COA Predicated']=y_pred


# In[55]:


check


# # visualize the regressor line

# In[56]:


plt.scatter(x,y, color = 'blue')
plt.plot(x_test,y_pred,color ='red')


# In[ ]:




