
# coding: utf-8

import numpy as np
from scipy import stats
from scipy import io as spio
from scipy import misc
from scipy import special
from scipy import integrate
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg

import pandas as pd

#get_ipython().magic(u'matplotlib inline')

from sklearn import cluster, datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Load data into DataFrame
df = pd.read_csv('FT_ALL.csv')
stats = df[['stat_1', 'stat_2', 'stat_3']].copy()

# Run the clustering alogrithm to get labels for each datapoint
k_means = cluster.KMeans(n_clusters=6)
k_means.fit(stats) 
labels = k_means.labels_ # Store the cluster labels
centroids = k_means.cluster_centers_

# Visualization the result
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

#ax.imshow(stats, interpolation='nearest',
#           cmap=plt.cm.Paired,
#           aspect='auto', origin='lower')

x = stats['stat_1']
y = stats['stat_2']
zs = stats['stat_3']

# Scatter without clustering
#ax.scatter(x, y, zs)
#ax.clf()

ax.scatter(x, y, zs, c=labels.astype(np.float))
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],  marker='x', s=120, linewidths=3, color='red', zorder=10)

ax.set_title('Student Slider')
ax.set_xlim3d(0, 100)
ax.set_ylim3d(0, 100)
ax.set_zlim3d(0, 100)
ax.set_xlabel('Collaboration')
ax.set_ylabel('Project/Concept Quality')
ax.set_zlabel('Industry Engagement')

plt.show()
