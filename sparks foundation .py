#!/usr/bin/env python
# coding: utf-8

# # Author: A.H.Harshith

# # Data Science and business analytics 

# # Prediction using Supervised ML

# # Predict the percentage of an student based on the no. of study hours.

# # TASK -1

# # --------------------------------------------------------------------------

# # importing the libraries

# In[5]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# #    loading the database

# In[6]:


url = 'http://bit.ly/w-data'
df= pd.read_csv(url)
df.head()


# #the data is imported successfully 

# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df.describe()


# #   plotting the distribution scores

# In[20]:


df.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# # Preperation of data

# In[25]:


X = df.iloc[:,:-1].values
y = df.iloc[:,1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# # Training the model

# In[27]:


from sklearn.linear_model import LinearRegression

regression=LinearRegression()
regression.fit(X_train,y_train)

print("training completed")


# In[28]:


line = regression.coef_*X+regression.intercept_

plt.scatter(X,y)
plt.plot(X,line);
plt.show()


# In[29]:


print(X_test)
y_pred = regression.predict(X_test)


# In[30]:


dataf = pd.DataFrame({'Actual':y_test,'Predicted': y_pred})
dataf


# # Testing, predicted score after 9.25 hours of studying 

# In[31]:


hours=9.25
pred=regression.predict(np.array(hours).reshape(-1,1))
print("No of Hours = {}".format(hours))
print("Predicted Score={}".format(pred[0]))


# # Printing the mean Absolute error

# In[37]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:




