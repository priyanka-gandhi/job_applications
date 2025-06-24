import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

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


f = open("data.txt")
trainingdata = f.readlines()[1:]
f.close()
trainingdata = np.array([[int(x),y.rstrip()] for x,y in (line.split(' ',1) for line in trainingdata)])

corpus = trainingdata[:,1]
Y = trainingdata[:,0]

min_count = 1

model = Word2Vec(corpus, min_count=min_count)

vector = []
for i in corpus:
    vector.append(vectors(i, model))
X= np.array(vector)

train,test, train_labels, test_labels = train_test_split(corpus,Y, test_size=0.2)

rf = RandomForestClassifier()
model = rf.fit(train, train_labels)

predict = model.predict(test)
print("Accuracy using is " ,accuracy_score(test_labels, predict))
