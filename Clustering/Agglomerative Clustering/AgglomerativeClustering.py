# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:20:56 2021

@author: abhis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\abhis\Music\THM.csv')

data = data[['Share Capital (in Rs. Crore)','Share price (Rs.)']]
df2 = {'Share Capital (in Rs. Crore)': 21, 'Share price (Rs.)': 44}

data = data.append(df2, ignore_index = True)
print(data)
from sklearn.preprocessing import normalize
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
data_scaled.head()

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.axhline(y=6, color='r', linestyle='--')

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data_scaled)


print(data_scaled)
print(cluster.fit_predict(data_scaled))




from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='ward')

plt.figure(figsize=(10, 7))  
plt.scatter(data_scaled['Share Capital (in Rs. Crore)'], data_scaled['Share price (Rs.)'], c=cluster.labels_) 