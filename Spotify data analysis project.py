#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[32]:


df = pd.read_csv(r'C:\Users\shanmukha varma\Downloads\SpotifyFeatures.csv.zip')
df.head()


# In[33]:


df.head()


# In[34]:


df.tail()


# In[35]:


pd.isnull(df).sum()


# In[63]:


df


# In[36]:


df.info()


# In[39]:


sorteddf =df.sort_values('popularity' ,ascending=True).head(10)
sorteddf


# In[40]:


df.describe().transpose()


# In[41]:


most_popular = df.query('popularity>90',inplace = False).sort_values('popularity', ascending = False)
most_popular[:10]


# In[47]:


#Changing duration from milliseconds to seconds
df['duration']= df['duration_ms'].apply(lambda x: round(x/1000))
df.drop('duration_ms', inplace=True, axis=1)


# In[51]:


df.duration.head()


# In[54]:


#Correlation
corr_df = df.drop(['key','mode'], axis=1).corr(method='pearson')
plt.figure(figsize=(14,6))
heatmap= sns.heatmap(corr_df, annot=True, fmt='.1g', vmin=-1, vmax=1,cmap ='inferno', center=0, linewidths=1, linecolor="Black")
heatmap.set_title("Correlation HeatMap between Variable")
heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=90)


# In[55]:


sample_df = df.sample(int(0.004*len(df)))


# In[56]:


len(sample_df)


# In[64]:


#Regression plot between loudness & enenrgy
#Regression plots as the name suggests creates a regression line between 2 parameters and helps to visualize their linear relationships.
plt.figure(figsize=(10,6))
sns.regplot(data=sample_df,y='loudness', x='energy', color='c').set(title = "Loudness vd Energy")


# In[58]:


#Regression plot between popularity & acousticness
#Regression plots as the name suggests creates a regression line between 2 parameters and helps to visualize their linear relationships.
plt.figure(figsize=(10,6))
sns.regplot(data=sample_df,y='popularity', x='acousticness', color='g').set(title = "popularity vs acousticness")


# # Analysis on genre of the song

# In[67]:


df_tracks = pd.read_csv(r'C:\Users\shanmukha varma\Downloads\SpotifyFeatures.csv.zip')


# In[68]:


df_tracks.head(10)


# In[70]:


plt.title('Duration of th songs from Different Genre')
sns.color_palette('rocket', as_cmap=True)
sns.barplot(y='genre', x='duration_ms', data=df_tracks)
plt.xlabel('Duration in Milliseconds')
plt.ylabel("Genre of songs")


# In[72]:


#Top 5 genre by popularity
sns.set_style(style = 'darkgrid')
plt.figure(figsize=(10,5))
famous = df_tracks.sort_values('popularity', ascending = False).head(10)
sns.barplot(y='genre', x= 'popularity', data= famous).set(title="Top 5 genre by popularity")


# In[73]:


df_tracks.info()


# In[ ]:




