#!/usr/bin/env python
# coding: utf-8

# In[130]:

# Import libraries
import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
# Clustering libraries and methods used
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
# Libraries to create 3D plot using seaborn cmap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
# PCA libraries and methods used 
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/nick-edu/dmmldl/master/MobilePrice.csv")

#Displays vital information about our dataset, includes all feature names and 4 rows with data
df.head()


# <h2>Description of the features</h2>

# **battery_power**: Total energy a battery can store in one time measured in mAh
# 
# **bluteooth**: If the phone has bluetooth or not
# 
# **clock_speed**: The speed at which microprocessor executes instructions
# 
# **dual_sim**: If the phone has dual sim support or not
# 
# **front_camera**: Front Camera mega pixels
# 
# **four_g**: If the phone has 4G capability or not
# 
# **internal_memory**: Internal memeory in gigabytes
# 
# **depth**: Mobile depth in terms of object
# 
# **width**: Mobile width in terms of object
# 
# **n_cores**: The number of cores in the processors
# 
# **primary_camera**: Primary Camera Megapixels
# 
# **px_height**: Pixel resolution height of the phone image display
# 
# **px_width**: Pixel resolution width of the phone image display
# 
# **ram**: RAM memory in Megabytes
# 
# **screen_height**: The height measurement of the screen in cm
# 
# **screen_width**: The width measurement of the screen in cm
# 
# **talk_time**: Longest time that a single battery charge will last when you are talking
# 
# **three_g**: If the phone has 3G capability
# 
# **touch_screen**: If the phone has touch screen or not
# 
# **wifi**: If the phone has wifi or not
# 
# **price_range**: low price (0) and high price(1)

# In[131]:


#  Display number of columns
print('Number of rows in the dataset:', df.shape[0])
# Display number of rows
print('Number of columns in the dataset:', df.shape[1])


# In[132]:


# Checks if there is any missing values
df.isnull().sum()
# No missing values are detected in any of the columns


# In[133]:


# Correlation matrix to check if there is any relationship between the variables 
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True)
# We can see that few variables are correlated and the one explaining the price range difference is the ram


# In[134]:


# Pairplot to see relationship and correlations among features
df1 = df[["battery_power", "ram", "n_cores", "clock_speed", "price_range"]]
sns.pairplot(df1, hue="price_range", palette='rocket')


# In[138]:


# Analysis on the price_range and RAM relationship
g = sns.FacetGrid(df, col="price_range", hue="price_range", palette='rocket')
g.map(sns.scatterplot, "ram", "battery_power", alpha=.7)
g.add_legend()


# In[124]:


sns.kdeplot(data=df, x="battery_power", hue="price_range", multiple="stack", palette='rocket')


# In[125]:


sns.violinplot(x="price_range", y="battery_power", data=df, palette='rocket')


# In[23]:


# We can see there seems to be higher number of processors in the phones in the higher price category
sns.boxplot(x='price_range', y='n_cores', data=df, palette='rocket')


# In[118]:


#We can clearly see that more phones in the higher price range tends to have higher battery power
sns.boxplot(x='price_range', y='battery_power', data=df, palette='rocket')


# In[24]:


#We can see pixel height seem to affect price
sns.barplot(data=df, x='price_range', y='px_height', palette='rocket')
plt.show()


# In[25]:


'''We can clearly see that the distribution is skewed based on the RAM, meaning a higher amount of RAM
correlates to a higher price range and viceversa'''
sns.displot(data=df, x='ram', hue='price_range', col='price_range', palette='rocket')
plt.show()


# <h1>1.2 Clustering</h1>
# <p>As part of this sub-question you will perform clustering on your chosen dataset from the above links. Choose one of the clustering algorithms that were discussed during the lecture for the application on your chosen dataset. Then choose few columns from your dataset that you think suitable for performing your chosen clustering. Describe and reflect on the clustering results and you are free to use the graphs/images and any other sort of visualizations also.
# </p>

# In[75]:


# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/nick-edu/dmmldl/master/MobilePrice.csv")

# Select the three most relevant features from the dataset
df_clustering = df[["battery_power", "ram", "n_cores"]]

# Scaling variables
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_clustering)

# Define a set of clusters
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 10]


# <h2>Silhouette Score</h2>
# <p>In order to find the most accurate number of clusters</p>

# In[76]:


score_list = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters = n_clusters)
    preds = clusterer.fit_predict(df_scaled)
    centers = clusterer.cluster_centers_

    score = silhouette_score(df_scaled, preds)
    score_list.append(score)
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
    
sns.lineplot(range_n_clusters, score_list)


# <p>The above graph suggests using from 6 to 8 clusters as they are the ones maximizing the silhouette score. However, the value of the silhouette score highlights the fact that the clusters are not very clearly separated, and some overlapping may occur in some instances.</p>
# <p>We decide to carry on building a model using 6 clusters</p>

# In[81]:


# define the kmeans model
kmeans_clustering_model = KMeans(n_clusters=6)


# In[140]:


# get the six clusters labels and store it in the dataframe
df_clustering["cluster"] = kmeans_clustering_model.fit_predict(df_scaled)
df_clustering.head()


# In[151]:


# 3D scatterplot of the clusters
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_clustering['battery_power'],df_clustering['ram'],df_clustering['n_cores'], marker="s", c=df_clustering["cluster"], s=40, cmap="RdBu")
plt.title('3D Visualization of the Six Clusters')
ax.set_xlabel('battery_power')
ax.set_ylabel('ram')
ax.set_zlabel('n_cores')
plt.show()


# <p>From the analysis of the chart we can see there is still some overlapping among the clusters. However, we can see the distinction among the clusters, centering mainly at the extremeties of each features distributions</p>
