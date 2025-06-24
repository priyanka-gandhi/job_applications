import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD



df = pd.read_excel("/Users/priyanka/Downloads/reinvitationtointerviewmpulsemobilebdsdeveloperr/Sample Message Data.xlsx")

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(df['content'])

#generating embeddings
truncated_svd = TruncatedSVD(300)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(truncated_svd, normalizer)
Xnew = lsa.fit_transform(X)

#k-means clustering
model = KMeans(n_clusters=3, max_iter=500, n_init=1).fit(Xnew)
df['kmeans'] = (list(model.labels_))


print((df[df.kmeans == 0]))
print((df[df.kmeans == 1]))
print((df[df.kmeans == 2]))


#df.to_csv("/Users/priyanka/Downloads/reinvitationtointerviewmpulsemobilebdsdeveloperr/kmeans_output.csv")
