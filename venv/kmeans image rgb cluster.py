import scipy.misc
'''
X = scipy.misc.imread('/Users/priyanka/Desktop/lstm/3.png')

#STEP 1
print(X.shape) # if png then 4th is the transparency layer so remove it
X = X[...,:3]
print(X.shape)      #   h * w * 3

# Some variables
NUM_CLUSTERS = 3     # k
NUM_ITER = 3          # n
NUM_ATTEMPTS = 5      # m

from sklearn.cluster import KMeans
km = KMeans(n_clusters=NUM_CLUSTERS, init='random', max_iter=1, n_init=1)#, verbose=1)
km.fit(X).
print('Pre-clustering metrics')
print('----------------------')
print('Inertia:', km.inertia_)
print('Centroids:', km.cluster_centers_)
'''


import numpy as np
import matplotlib.pyplot as plt
import cv2

k = int(input("Enter number of clusters: "))

X=cv2.imread('IMG_20210303_195735.jpg')
X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)  #image to vector

plt.figure(figsize=(10,10))
plt.imshow(X)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#print(X.shape)      #h * w * 3

X1 = np.float32(X.reshape((-1,3)))     #converting to 2d matrix


centers_matrix=[]
compactness1=[]
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)       #iterative kmeans set to 100 iterations
for i in  range(2,k+1):     #runs loop from 2 clusters to the user input k value
    compactness,labels,centers = cv2.kmeans(X1,i,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)  #kmeans with randomly chosen centroids.append(
    centers = np.uint8(centers) #final cluster centroids
    processed_image = centers[labels.flatten()].reshape((X.shape))
    compactness1.append(compactness)
    centers_matrix.append(processed_image)
print(centers_matrix)
print(compactness1)

for j in range(len(centers_matrix)):
    plt.imshow(centers_matrix[j])
    plt.title('Processed Image \n K = %i' % (j+2)), plt.xticks([]), plt.yticks([])
    plt.show()

