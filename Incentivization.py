#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize']=(12,6)


# In[2]:


df=pd.read_csv('driver-data.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


from sklearn.cluster import KMeans


# In[6]:


kmeans=KMeans(n_clusters=2)
df_analyze=df.drop('id',axis=1)


# In[8]:


kmeans.fit(df_analyze)


# In[9]:


kmeans.cluster_centers_


# In[11]:


print(kmeans.labels_)
print(len(kmeans.labels_))


# In[12]:


print(type(kmeans.labels_))
unique,counts =np.unique(kmeans.labels_,return_counts=True)
print(dict(zip(unique,counts)))


# In[15]:


df_analyze['cluster']=kmeans.labels_
sns.set_style('whitegrid')
sns.lmplot('mean_dist_day','mean_over_speed_perc', data=df_analyze,
          hue='cluster',palette='coolwarm',size=6,aspect=1, fit_reg=False)


# In[16]:


kmeans_4=KMeans(n_clusters=4)
kmeans_4.fit(df.drop('id',axis=1))
kmeans_4.fit(df.drop('id',axis=1))
print(kmeans_4.cluster_centers_)
unique,counts=np.unique(kmeans_4.labels_,return_counts=True)

kmeans_4.cluster_centers_
print(dict(zip(unique,counts)))


# In[19]:


df_analyze['cluster']=kmeans_4.labels_
sns.set_style('whitegrid')
sns.lmplot('mean_dist_day','mean_over_speed_perc', data=df_analyze,
          hue='cluster',palette='coolwarm',size=6,aspect=1, fit_reg=False)


# In[ ]:




