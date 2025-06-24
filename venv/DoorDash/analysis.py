import pandas as pd
import matplotlib.pyplot as plt

'''
df= pd.read_csv("output.csv")
print(df.columns)
#print(len(df))
#data = (df['Delay (> 10mins)'].value_counts())
data = df['Delay (> 5mins) '].value_counts().rename_axis('Category').reset_index(name='Count')
data['Percentage'] = data['Count'].div(sum(data['Count'])).mul(100)
print(data)
#print(df['Delay '].value_counts())


import pandas as pd
import re

#data = pd.read_csv('testdata.csv')
#print(data.head())

import matplotlib.pyplot as plt

fig = plt.figure(figsize =(10, 7))
plt.pie(data['Percentage'], labels = data['Category'], autopct='%.2f%%')
plt.title("Distribution Considering Delay within 5 Minutes")
# show plot
plt.show()


'''
df= pd.read_csv("/Users/priyanka/Desktop/Resume/interviews/doordash/historical_data.csv")
df['created_at'] = pd.to_datetime(df['created_at'],format = '%Y-%m-%d %H:%M:%S',errors = 'raise')

df['created_at_day'] = df['created_at'].dt.dayofweek
print((df['created_at_day'].value_counts()))
