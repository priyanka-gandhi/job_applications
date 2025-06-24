import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import holidays
us_holidays = holidays.US()

def fast_mode(df, key_cols, value_col):
    return (df.groupby(key_cols + [value_col]).size().to_frame('counts').reset_index().sort_values('counts', ascending=False).drop_duplicates(subset=key_cols)).drop(columns='counts')

df = pd.read_csv("/Users/priyanka/Desktop/Resume/interviews/doordash/historical_data.csv")


#df.nunique()    #checking for variance of each column data
#df.duplicated().any()       #checking if there are any duplicates
#df['created_at'] > df['actual_delivery_time'].sum()    #to check if there was noise where created date is after Delivery
#df['total_busy_dashers'] > df['total_onshift_dashers'].sum()    #more than half of the records had null values or errors. since busy dashers is a subset of onshift dashers. Data not reliable for prediction


#dealing with null values
df.loc[df.store_primary_category.isnull(), 'store_primary_category'] = df.store_id.map(fast_mode(df, ['store_id'], 'store_primary_category').set_index('store_id').store_primary_category)
df.loc[df.market_id.isnull(), 'market_id'] = df.store_id.map(fast_mode(df, ['store_id'], 'market_id').set_index('store_id').market_id)
df['market_id'].fillna(value=0, inplace=True)       #unknown markets are set to 0
df['store_primary_category'].fillna(value='Unknown', inplace=True)      #unknown store category is set to Unknown
df['order_protocol'].fillna(value=0, inplace=True)      #unknown order protocol are set to 0


df.update(df[['total_onshift_dashers','total_busy_dashers','total_outstanding_orders']].fillna(0))     #not depending on store_id/time/market_id
df = df.dropna(subset=['actual_delivery_time','estimated_store_to_consumer_driving_duration'])     #dropping rows with NA delivery time and store-to-consumer as there is no way for us to estimate the time since address is not given

df.isnull().sum()   # to view features with null values


df['created_at'] = pd.to_datetime(df['created_at'],format = '%Y-%m-%d %H:%M:%S',errors = 'raise')
df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'],format = '%Y-%m-%d %H:%M:%S',errors = 'raise')


#Feature Engineering
df['created_at_day'] = df['created_at'].dt.dayofweek
df['created_at_hour'] = df['created_at'].dt.hour
df['created_at_min'] = df['created_at'].dt.minute
df['is_holiday'] = [1 if i in us_holidays else 0 for i in df['created_at']]     #considering US holiday
#df['Month'] = df['created_at'].dt.month    #historical data has only 3 months of data so not considering it as a feature

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['store_primary_category'] = labelencoder.fit_transform(df['store_primary_category'].astype(str))         #encoding string features


#df['preparation_time'] = df['duration'] - (df['estimated_order_place_duration']+df['estimated_store_to_consumer_driving_duration'])

X = df[['store_primary_category','num_distinct_items','created_at_day','created_at_hour','is_holiday','estimated_store_to_consumer_driving_duration']]
df['duration'] = (df['actual_delivery_time']-df['created_at']).dt.total_seconds()
y= df['duration']



#Correlation Matrix
corr = df.corr()
sns.set(font_scale=0.5)
sns.heatmap(corr, annot=True, fmt=".2f")
plt.savefig("corr.png",bbox_inches='tight')

'''
# to view the loss/error to minimize cost function
n_estimators = [25, 50, 100, 150, 200, 250, 300, 350]

for val in n_estimators:
    score = cross_val_score(RandomForestRegressor(n_estimators= val, random_state= 42), X, y, cv= 5)
    print(f'Average score({val}): {"{:.3f}".format(score.mean())}')

'''


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train, test, train_labels, test_labels = train_test_split(X,y, test_size=0.2,random_state=50)
rf = RandomForestRegressor(n_estimators = 150, random_state = 42, max_depth=50)

model = rf.fit(train, train_labels)
rfe = RFE(model, 4)
fit = rfe.fit(X, y)
print("Feature Ranking: %s" % (fit.ranking_))  #the least ranked columns were dropped from dataframe X

cv_score = cross_val_score(rf, train, train_labels, cv = 5)
predictions = cross_val_predict(rf, test, test_labels, cv = 5)




model = rf.fit(train, train_labels)


#Evaluation Metrics
errors = mean_absolute_error(test_labels, predictions)
mape = 100 * (errors / test_labels)
print('Mean Absolute Percentage Error:', round(np.mean(mape), 2), '%.')
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
print("R2:  ", r2_score(test_labels, predictions))

df_new = pd.DataFrame()
df_new["Labels"] = test_labels
df_new["Predicted"] = predictions
df_new.to_csv("output1.csv",index=False)

df_test = pd.read_csv("/Users/priyanka/Desktop/Resume/interviews/doordash/predict_data.csv")

df_test.loc[df_test.store_primary_category.isnull(), 'store_primary_category'] = df_test.store_id.map(fast_mode(df_test, ['store_id'], 'store_primary_category').set_index('store_id').store_primary_category)
df_test['store_primary_category'].fillna(value='Unknown', inplace=True)      #unknown store category is set to Unknown
df_test['store_primary_category'] = labelencoder.fit_transform(df_test['store_primary_category'].astype(str))         #encoding string features
df_test['created_at'] = pd.to_datetime(df_test['created_at'],format = '%Y-%m-%d %H:%M:%S',errors = 'raise')
df_test['created_at_day'] = df_test['created_at'].dt.dayofweek
df_test['created_at_hour'] = df_test['created_at'].dt.hour
df_test = df_test.dropna(subset=['estimated_store_to_consumer_driving_duration'])     #dropping rows with NA store-to-consumer as there is no way for us to estimate the time since address is not given
df_test['is_holiday'] = [1 if i in us_holidays else 0 for i in df_test['created_at']]

X_test = df_test[['store_primary_category','num_distinct_items','created_at_day','created_at_hour','is_holiday','estimated_store_to_consumer_driving_duration']]


new_predictions = model.predict(X_test)

df_new = pd.DataFrame()
df_new["delivery_id"] = df_test['delivery_id']
df_new["Predicted"] = new_predictions
df_new.to_csv("/Users/priyanka/Desktop/Resume/interviews/doordash/predictions.csv",index=False)
