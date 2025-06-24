'''
Please find the below dataset:
https://data.world/ian/2013-zip-code-income

Assume only 15% of income can be spent on a car and three cars were available with 35K, 60K and 90K.
Predict which zip code will have more customers to which car.
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

df= pd.read_csv("/Users/priyanka/Downloads/13zpallagi.csv")

df["car allowance"] = df["N02650"]*(0.15)

# 0 - No car, 1 - 30k, 2 - 60k, 3 - 90k
car=[]
for i in df["car allowance"]:
    i = int(i)
    if i > 30000 and i < 60000:
        car.append('1')
    elif i > 60000 and i < 90000:
        car.append('2')
    elif i > 90000:
        car.append('3')
    else:
        car.append('0')
X = df.drop(["STATE"], axis=1)
y = [int(i) for i in car]
df["car"] = car

# Split our data
train, test, train_labels, test_labels = train_test_split(X,y, test_size=0.25,random_state=50, stratify=y)

#Naive Bayes
gnb = GaussianNB()
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print("Accuracy using Naive Bayes is " , accuracy_score(test_labels, preds))

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


#print(classification_report(test_labels, preds))
#print(confusion_matrix(test_labels, preds))
