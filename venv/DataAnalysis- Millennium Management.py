import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

df = pd.read_excel("/Users/priyanka/Desktop/Resume/interviews/Millennium Management/Sample Dataset.xlsx")
print(df)


print(df.isna().sum())  #checking for missing values
print(df[df.duplicated()])      #checking for duplicate records

df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', inplace=True)

#statistical charateristics of the datsets
#df[2:].describe().to_csv("Data Description.csv",index=False)
cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
for i, col in enumerate(cols):
    print(df[col].describe())


'''
#The new dataframe fills the missing weekdays (business days) dates records with previous day's recordings
df1 = df.merge(
    pd.DataFrame({'Date':df['Date'] + pd.offsets.BDay()}), on='Date', how='outer'
).sort_values('Date').ffill().dropna().reset_index(drop=True)

#print(pd.concat([df,df1]).drop_duplicates(keep=False))     #records that were added in the new dataframe (df1)
'''

#Correlation Matrix
fig, ax = plt.subplots(figsize=(14,8))
df=df[["Open", "High","Low","Close","Adj Close"]]
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',linecolor='white',linewidths=.5)
plt.title("Correlation Matrix")
plt.show()


#Visualization and Outliers
df.plot(x="Date", y=["High", "Low","Open","Close","Adj Close"])

df1 = pd.DataFrame(df['Close'])
df1.rename(columns={'Close':'close'}, inplace=True)
df1['percentage_change']=df1.close.pct_change()

df1_mean = df1['percentage_change'].agg(['mean', 'std'])
fig, ax = plt.subplots(figsize=(10,6))
df1['percentage_change'].plot(label='percentage_change', legend=True, ax = ax)
plt.axhline(y=df1_mean.loc['mean'], c='r', label='mean')
plt.axhline(y=df1_mean.loc['std'], c='c', linestyle='-.',label='std')
plt.axhline(y=-df1_mean.loc['std'], c='c', linestyle='-.',label='std')
plt.legend(loc='lower right')
plt.title("Percentage Change of Close")
plt.show()


#Visualizing Closing Price wrt Date
fig = plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'])
plt.show()

#candle stick plot
df = pd.read_excel("/Users/priyanka/Desktop/Resume/interviews/Millennium Management/Sample Dataset.xlsx")
import plotly.graph_objects as go
fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

fig.show()


