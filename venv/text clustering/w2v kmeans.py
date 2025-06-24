import nltk
import pandas
import pprint
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

from sklearn.metrics import classification_report

def vectors(sentence, model):
    vector=[]
    count=0
    for word in sentence:
        try:
            if count == 0:
                vector = model[word]
            else:
                vector = np.add(vector,model[word])
                count=count+1
        except:
            pass
    return np.asarray(vector)

dataset = pandas.read_excel('/Users/priyanka/Downloads/reinvitationtointerviewmpulsemobilebdsdeveloperr/Sample Message Data.xlsx', encoding = 'utf-8')
df = pandas.read_csv('/Users/priyanka/Downloads/reinvitationtointerviewmpulsemobilebdsdeveloperr/StanfordCoreNLP_results.csv', encoding = 'utf-8')


comments = dataset['content']
y = df['Stanford CoreNLP Result1']

min_count = 1

#window = 4

model = Word2Vec(comments, min_count=min_count)

vector = []
for i in comments:
    vector.append(vectors(i, model))
X= np.array(vector)

kclusterer = KMeans(3, max_iter=5000)
kclusterer.fit(X)
labels = kclusterer.labels_
print(labels)
dataset['labels'] = labels

print(classification_report(y,labels))
#dataset.to_csv("/Users/priyanka/Downloads/reinvitationtointerviewmpulsemobilebdsdeveloperr/w2v kmeans.csv", index=False)


'''
X = model[model.wv.vocab]

clusters_number = 3
kclusterer = KMeansClusterer(clusters_number,  distance=nltk.cluster.util.cosine_distance)

assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

words = list(model.wv.vocab)
for i, word in enumerate(words):
    print (word + ":" + str(assigned_clusters[i]))

kmeans = cluster.KMeans(n_clusters = clusters_number)
kmeans.fit(X)

labels = kmeans.labels_
print(labels)
centroids = kmeans.cluster_centers_


clusters = {}
for commentaires, label in zip(comments, labels):
    try:
        clusters[str(label)].append(comments)
    except:
       clusters[str(label)] = [comments]
pprint.pprint(clusters)

df_new=pd.DataFrame()
df_new =list(clusters)
df_new.to_csv("/Users/priyanka/Downloads/reinvitationtointerviewmpulsemobilebdsdeveloperr/w2v kmeans.csv", index=False)
'''
