#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# In[2]:


# Parameters
n_samples = 300
n_features = 2
centers = 4


# In[3]:


# Generate data
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=3, random_state=42)


# In[4]:


# Visualize the data
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.title("Generated Dataset for K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# In[5]:


from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_init = 10, n_clusters = i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[6]:


plt.plot(range(1,11),wcss)


# In[ ]:


kmeans = KMeans(n_init = 10, n_clusters = 4)
y_kmeans = kmeans.fit_predict(X)


# In[ ]:


plt.scatter(X[y_kmeans==0, 0], X[y_kmeans == 0,1], s = 60, c= 'red', label = 'Cluster1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans == 1,1], s = 60, c= 'blue', label = 'Cluster2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans == 2,1], s = 60, c= 'green', label = 'Cluster3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans == 3,1], s = 60, c= 'yellow', label = 'Cluster4')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s= 100, c= 'black', label = 'Centroids')


# In[ ]:




