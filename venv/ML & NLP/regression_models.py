import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

df= pd.read_csv("data.txt", sep=",", names =['val','labels'])

X = pd.DataFrame(df['val'])
y= pd.DataFrame(df['labels'])

x_train,test,train_labels, test_labels = train_test_split(X,y, test_size=0.2)

lm = LinearRegression()
model = lm.fit(x_train,train_labels)
predict = model.predict(test)

print("MAE: ", mean_absolute_error(test_labels,predict))
print("MSE: ", mean_squared_error(test_labels,predict))
