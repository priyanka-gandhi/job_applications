import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

import holidays
us_holidays = holidays.US()

def fast_mode(df, key_cols, value_col):
    return (df.groupby(key_cols + [value_col]).size().to_frame('counts').reset_index().sort_values('counts', ascending=False).drop_duplicates(subset=key_cols)).drop(columns='counts')

df = pd.read_csv("/Users/priyanka/Desktop/Resume/interviews/doordash/historical_data.csv")
print(df.columns)

#print(df.nunique())    #checking for variance of each column data
#print(df.duplicated().any())       #checking if there are any duplicates


#dealing with null values
df.loc[df.store_primary_category.isnull(), 'store_primary_category'] = df.store_id.map(fast_mode(df, ['store_id'], 'store_primary_category').set_index('store_id').store_primary_category)
df.loc[df.market_id.isnull(), 'market_id'] = df.store_id.map(fast_mode(df, ['store_id'], 'market_id').set_index('store_id').market_id)

df['market_id'].fillna(value=0, inplace=True)       #unknown markets are set to 0
df['store_primary_category'].fillna(value='Unknown', inplace=True)      #unknown store category is set to Unknown
df['order_protocol'].fillna(value=0, inplace=True)      #unknown order protocol are set to 0

df = df.dropna(subset=['actual_delivery_time','estimated_store_to_consumer_driving_duration'])     #dropping rows with NA delivery time and store-to-consumer as there is no way for us to estimate the time since address is not given


#print((df['created_at'] > df['actual_delivery_time']).sum())    #to check if there was noise where created date is after Delivery
#df['Month'] = df['created_at'].dt.month    #historical data has only 3 months of data so not considering it as a feature


#---------------------------------------------

print(df.isnull().sum())    # to view features with null values


df['created_at'] = pd.to_datetime(df['created_at'],format = '%Y-%m-%d %H:%M:%S',errors = 'raise')
df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'],format = '%Y-%m-%d %H:%M:%S',errors = 'raise')


#to get the day of the week, hour and min when the order was created
df['created_at_day'] = df['created_at'].dt.dayofweek
df['created_at_hour'] = df['created_at'].dt.hour
df['created_at_min'] = df['created_at'].dt.minute


df['duration'] = (df['actual_delivery_time']-df['created_at']).dt.total_seconds()
df['is_holiday'] = [1 if i in us_holidays else 0 for i in df['created_at']]     #considering US holiday

df['store_primary_category'] = labelencoder.fit_transform(df['store_primary_category'].astype(str))         #encoding string features
print(df['store_primary_category'])

#df['preparation_time'] = df['duration'] - (df['estimated_order_place_duration']+df['estimated_store_to_consumer_driving_duration'])
#X = df[['store_primary_category','total_items','num_distinct_items','estimated_order_place_duration','estimated_store_to_consumer_driving_duration','created_at_day','created_at_hour','is_holiday']]

#X = df[['store_primary_category','subtotal','min_item_price','max_item_price','estimated_store_to_consumer_driving_duration','created_at_day','created_at_hour']]
y= df['duration']

#x = df[['store_primary_category','total_items','num_distinct_items','estimated_order_place_duration','estimated_store_to_consumer_driving_duration','created_at_day','created_at_hour','is_holiday']]

x= df[['store_primary_category','num_distinct_items','created_at_day','is_holiday','created_at_hour','estimated_store_to_consumer_driving_duration']]
#PCA
from sklearn.preprocessing import StandardScaler

class convers_pca():
    def __init__(self, no_of_components):
        self.no_of_components = no_of_components
        self.eigen_values = None
        self.eigen_vectors = None
        
    def transform(self, x):
        return np.dot(x - self.mean, self.projection_matrix.T)
    
    def inverse_transform(self, x):
        return np.dot(x, self.projection_matrix) + self.mean
    
    def fit(self, x):
        self.no_of_components = x.shape[1] if self.no_of_components is None else self.no_of_components
        self.mean = np.mean(x, axis=0)
        
        cov_matrix = np.cov(x - self.mean, rowvar=False)
        
        self.eigen_values, self.eigen_vectors = np.linalg.eig(cov_matrix)
        self.eigen_vectors = self.eigen_vectors.T
        
        self.sorted_components = np.argsort(self.eigen_values)[::-1]
        
        self.projection_matrix = self.eigen_vectors[self.sorted_components[:self.no_of_components]]
        self.explained_variance = self.eigen_values[self.sorted_components]
        self.explained_variance_ratio = self.explained_variance / self.eigen_values.sum()


        print(self.eigen_vectors)
        print(self.eigen_values)
        print(self.sorted_components)

std = StandardScaler()
transformed = StandardScaler().fit_transform(x)
pca = convers_pca(no_of_components=6)
pca.fit(transformed)

x_std = pca.transform(transformed)





#print(X)
#print(df[df.columns].corr()['duration'][:])

#corr = df.corr()
#sns.set(font_scale=0.5)
#sns.heatmap(corr, annot=True)
#plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict

'''
n_estimators = [50, 100, 150, 200, 250, 300, 350]

for val in n_estimators:
    score = cross_val_score(RandomForestRegressor(n_estimators= val, random_state= 42), X, y, cv= 5)
    print(f'Average score({val}): {"{:.3f}".format(score.mean())}')

'''

X = df[['store_primary_category','num_distinct_items','created_at_hour','estimated_store_to_consumer_driving_duration']]

# Split our data
train, test, train_labels, test_labels = train_test_split(X,y, test_size=0.2,random_state=50)


rf = RandomForestRegressor(n_estimators = 150, random_state = 42, max_depth=50)
#model = rf.fit(train, train_labels)


'''
#Feature Selection
#Correlation Matrix
corr = df.corr()
sns.set(font_scale=0.5)
sns.heatmap(corr, annot=True, fmt=".2f")
plt.savefig("corr.png",bbox_inches='tight')

#PCA 
class convers_pca():
    def __init__(self, no_of_components):
        self.no_of_components = no_of_components
        self.eigen_values = None
        self.eigen_vectors = None
        
    def transform(self, x):
        return np.dot(x - self.mean, self.projection_matrix.T)
    
    def inverse_transform(self, x):
        return np.dot(x, self.projection_matrix) + self.mean
    
    def fit(self, x):
        self.no_of_components = x.shape[1] if self.no_of_components is None else self.no_of_components
        self.mean = np.mean(x, axis=0)
        
        cov_matrix = np.cov(x - self.mean, rowvar=False)
        
        self.eigen_values, self.eigen_vectors = np.linalg.eig(cov_matrix)
        self.eigen_vectors = self.eigen_vectors.T
        
        self.sorted_components = np.argsort(self.eigen_values)[::-1]
        
        self.projection_matrix = self.eigen_vectors[self.sorted_components[:self.no_of_components]]
        self.explained_variance = self.eigen_values[self.sorted_components]
        self.explained_variance_ratio = self.explained_variance / self.eigen_values.sum()


        print(self.eigen_vectors)
        print(self.eigen_values)
        print(self.sorted_components)

std = StandardScaler()
transformed = StandardScaler().fit_transform(x)
pca = convers_pca(no_of_components=6)
pca.fit(transformed)

x_std = pca.transform(transformed)

#RFE with RF
model = rf.fit(train, train_labels)
rfe = RFE(model, 4)
fit = rfe.fit(X, y)
#print("Feature Ranking: %s" % (fit.ranking_))  #the least ranked columns were dropped from dataframe X

'''

'''
#for feature selection
rfe = RFE(model, 4)
fit = rfe.fit(X, y)
#print("Feature Ranking: %s" % (fit.ranking_))  #the least ranked columns were dropped from dataframe X

# example of mutual information feature selection for numerical input data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot

# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_regression, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

'''

cv_score = cross_val_score(rf, train, train_labels, cv = 5)
#print("CV mean score: ", cv_score.mean())


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

predictions = cross_val_predict(rf, test, test_labels, cv = 5)
# Calculate the absolute errors
#errors = abs(predictions - test_labels)
errors = mean_absolute_error(test_labels, predictions)
# Print out the mean absolute error (mae)
#print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
print('Mean Absolute Percentage Error:', round(np.mean(mape), 2), '%.')


# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

print("R2:  ", r2_score(test_labels, predictions))




mse = mean_squared_error(test_labels, predictions)

print("MAE: ", mean_absolute_error(test_labels, predictions))
print("MSE: ", mse)
print("RMSE: ", mse*(1/2.0))
#print(math.sqrt(mse))


df_new = pd.DataFrame()
df_new["Labels"] = test_labels
df_new["Predicted"] = predictions
df_new.to_csv("output1.csv",index=False)

print("--------------")
'''
predictions = model.predict(test)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
print('Mean Absolute Percentage Error:', round(np.mean(mape), 2), '%.')


# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

from sklearn.metrics import mean_squared_error, mean_absolute_error




mse = mean_squared_error(test_labels, predictions)

print("MAE: ", mean_absolute_error(test_labels, predictions))
print("MSE: ", mse)
print("RMSE: ", mse*(1/2.0))
#print(math.sqrt(mse))



x_ax = range(len(test_labels))
plt.plot(x_ax, test_labels, linewidth=1, label="original")
plt.plot(x_ax, predictions, linewidth=1.1, label="predicted")
plt.axis([min(test_labels),(test_labels.mean())/2,min(predictions),(predictions.mean())/2])
plt.title("y-test and y-predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()
'''

'''
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


score = sgdr.score(xtrain, ytrain)
print("R-squared:", score)
cv_score = cross_val_score(sgdr, X, y, cv = 10)
print("CV mean score: ", cv_score.mean())

ypred = nsvr.predict(xtest)

mse = mean_squared_error(ytest, ypred)
print("MSE: ", mse)
print("RMSE: ", mse*(1/2.0))

x_ax = range(len(ytest))
plt.plot(x_ax, ytest, linewidth=1, label="original")
plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()
'''

'''
CV mean score:  [-1.49470266e-04 -9.16980773e+00 -1.76854668e+00 -1.05830866e+00
 -1.70411265e+02]
Mean Absolute Error: 765.52 degrees.
Accuracy: 71.13 %.


CV mean score:  -132.99529802817466
Mean Absolute Error: 782.51 degrees.
Mean Absolute Percentage Error: 56879     21.15
38051     13.49
71406     14.38
148242    12.71
116175     7.38
          ...  
118178    46.49
167179    24.19
171085     0.84
151527    13.88
82277     18.22
Name: duration, Length: 39379, dtype: float64 %.
Accuracy: 70.37 %.
MSE:  5622975.455289934
RMSE:  2811487.727644967

'''
