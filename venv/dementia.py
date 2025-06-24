import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix, precision_score, recall_score

df = pd.read_csv('/Users/priyanka/Downloads/ce_cogs_dx.txt',  sep=" ")
#df.to_csv("/Users/priyanka/Downloads/ce_cogs_dx.csv", index=False)

df = df.drop(['projid','cycle','cycle_f'], axis=1)

print(df.describe())


df = df[df.dx != 1]

#Correlation Matrix
fig, ax = plt.subplots(figsize=(14,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',linecolor='white',linewidths=.5)
plt.title("Correlation Matrix")
#plt.show()

#df = df[['mmsetot', 'ebdrtot', 'ebmttot', 'sdmtcor','dx']]
y = df['dx']
X = df.drop(["dx"], axis=1)


# Split our data
train, test, train_labels, test_labels = train_test_split(X,y, test_size=0.25,random_state=50, stratify=y)

#Naive Bayes
gnb = GaussianNB()
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print("Accuracy using Naive Bayes is " , accuracy_score(test_labels, preds))
print(classification_report(test_labels, preds))


df1 = pd.DataFrame(test)
df2 = pd.DataFrame(test_labels)

df3 = pd.concat((df1,df2), axis =1)

df3['Predicted'] = preds
#print(df3)


#Decision Tree Model
clf = DecisionTreeClassifier()
clf = clf.fit(train, train_labels)
preds = clf.predict(test)
print ("Accuracy using Decision Tree is " , accuracy_score(test_labels, preds))

#K Neighbors Classifier
knn = KNeighborsClassifier()
knn = knn.fit(train, train_labels)
preds = knn.predict(test)
print("Accuracy using K Neighbors Classifier is " , accuracy_score(test_labels, preds))

#using MLPClassifier
mlpc = MLPClassifier()
mlpc = mlpc.fit(train, train_labels)
preds = mlpc.predict(test)
print("Accuracy using MLPC Classifier is " ,accuracy_score(test_labels, preds))

#Using Random Forest Classifier
rfor = RandomForestClassifier()
rfor = rfor.fit(train, train_labels)
preds = rfor.predict(test)
print("Accuracy using Random Forest is " ,accuracy_score(test_labels, preds))

#Using SVM
svc_model = SVC()
svc_model = svc_model.fit(train, train_labels)
preds = svc_model.predict(test)
print("Accuracy using SVM is " ,accuracy_score(test_labels, preds))

#Using Logistic Regression
logreg = LogisticRegression()
logreg = logreg.fit(train, train_labels)
preds = logreg.predict(test)
print("Accuracy using Logistic Regression is " ,accuracy_score(test_labels, preds))







