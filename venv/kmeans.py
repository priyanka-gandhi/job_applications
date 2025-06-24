import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

data = make_blobs(n_samples=200)
X = (data[0])

#initialize the centroids randomly
import random
init_centroids = random.sample(range(0, len(X)), 3)
centroids = []
for i in init_centroids:
    centroids.append(X[i])
centroids = np.array(centroids)
print(centroids)

# cluster assignment
def ClosestCentroids(cent, X):
    closest_centroid = []
    for i in X:
        distance=[]
        for j in cent:
            dist = (sum((i - j)**2))**0.5
            distance.append(dist)
        closest_centroid.append(np.argmin(distance))
    return closest_centroid

#update_centroid = ClosestCentroids(centroids, X)
#print(update_centroid)

# move the centroids based on the mean of the data points
def calc_centroids(clusters, X):
    new_centroids = []
    new_df = pd.concat([pd.DataFrame(X), pd.DataFrame(clusters, columns=['cluster'])],axis=1)
    for c in set(new_df['cluster']):
        current_cluster = new_df[new_df['cluster'] == c][new_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids

update_centroid = ClosestCentroids(centroids, X)
centroids = calc_centroids(update_centroid, X)
print(centroids)
    #plt.figure()
    #plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='black')
    #plt.scatter(X[:, 0], X[:, 1], alpha=0.1)
    #plt.show()
