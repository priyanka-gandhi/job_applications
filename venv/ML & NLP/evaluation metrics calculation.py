import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score
import numpy as np

actual = [1,0,1,1,1,0,0,1,0,1]
pred = [1,0,1,1,1,1,0,1,0,0]

tp,tn,fp,fn = [0]*4
for i in range(len(actual)):
    if (actual[i] == pred[i] == 1):
        tp+=1
    elif (actual[i] == 0) and (pred[i] == 1):
        fp+=1
    elif (actual[i] == 1) and (pred[i] == 0):
        fn+=1
    else:
        tn+=1

print("Accuracy:    ",((tp+tn)/(tp+fp+fn+tn)),accuracy_score(actual,pred))
print("Precision:   ",(tp/(tp+fp)), precision_score(actual,pred))
print("Recall:  ",(tp/(tp+fn)), recall_score(actual,pred))

actual = np.array([-3, -1, -2, 1, -1, 1, 2, 1, 3, 4, 3, 5])
pred = np.array([-2, 1, -1, 0, -1, 1, 2, 2, 3, 3, 3, 5])

d = actual - pred

mae = np.mean(abs(d))

mse = np.mean(d**2)
rmse = np.sqrt(mse)

r2 = 1-(sum(d**2)/sum((actual-np.mean(actual))**2))

#Reading files

#Text file
text_file = open("text.txt", "r")
lines = text_file.read()

#json
df = pd.read_json("/home/kunal/Downloads/Loan_Prediction/train.json")
