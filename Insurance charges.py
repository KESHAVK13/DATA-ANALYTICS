#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt # visualization
import matplotlib as mpl # visualization tuning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df=pd.read_csv("insurance.csv")
df.head() 


# In[6]:


df.isna().sum() #checking for any null values


# In[7]:


df.info() #checking datatypes (later we have to convert string/objects to int/float type to be able to use models)


# In[8]:


df.describe() #descriptive statistics of features


# In[9]:


region_values={"southwest":1,
             "southeast":2,
              "northwest":3,
              "northeast":4} #creating a dictionary to replace values in region column


# In[10]:


df["region_encodeing"]=df.region.map(region_values)


# In[11]:


df.head() #region column can now be substituted with region_encodeing


# In[13]:


df.drop(['region'],inplace=True,axis=1)


# In[15]:


sex_values={"female":0,'male':1}
smoker_values={'yes':1,'no':0}
df["sex_values"]=df.sex.map(sex_values)
df['smoker_values']=df.smoker.map(smoker_values)


# In[16]:


df.head() #now we can remove smoker and sex columns as we have replaced them with binary 


# In[20]:


df.drop(['sex'],inplace=True,axis=1)


# In[21]:


df.drop(['smoker'],inplace=True,axis=1)


# In[22]:


df.head()


# In[27]:


plt.hist(df['charges'],bins=2000)
plt.show()


# In[31]:


#scaling and normalizing features 
scaler=MinMaxScaler()
df_scaled=scaler.fit_transform(df)


# In[37]:


df_scaled_df= pd.DataFrame(df_scaled, columns=['age','bmi','children','charges','region_encoding','sex_type','smoker_or_no'])


# In[38]:


df_scaled_df.head()


# In[39]:


x = df_scaled_df.drop(['charges'], axis = 1)
y = df_scaled_df['charges']


# In[41]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[42]:


Lin_reg = LinearRegression()
Lin_reg.fit(x_train, y_train)


# In[43]:


print(Lin_reg.intercept_)
print(Lin_reg.coef_)
print(Lin_reg.score(x_test, y_test))


# In[ ]:




