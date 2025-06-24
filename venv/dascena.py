import os, sys, time, re, random, math, json
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#with open('berlin_ards.json') as json_file:
#    berlin_ards = json.load(json_file)

import json
import pandas as pd
from pandas.io.json import json_normalize


#json to csv
df=pd.read_json (r'/Users/priyanka/Downloads/dascena/berlin_ards.json')
df = df.transpose()
print(df)

df = pd.concat([df.drop(['Info'], axis=1), df['Info'].apply(pd.Series)], axis=1)

names = ['SysABP', 'DiasABP', 'HR', 'Temp', 'RespRate', 'SpO2', 'Creatinine', 'WBC', 'Platelets']


df['SysABP'] = [i[0] for i in df['feature_matrix']]
df['DiasABP'] = [i[1] for i in df['feature_matrix']]
df['HR'] = [i[2] for i in df['feature_matrix']]
df['Temp'] = [i[3] for i in df['feature_matrix']]
df['RespRate'] = [i[4] for i in df['feature_matrix']]
df['SpO2'] = [i[5] for i in df['feature_matrix']]
df['Creatinine'] = [i[6] for i in df['feature_matrix']]
df['WBC'] = [i[7] for i in df['feature_matrix']]
df['Platelets'] = [i[8] for i in df['feature_matrix']]

df = df.drop(['feature_matrix'], axis=1)


df = df.drop(['feature_matrix_row_names'], axis=1)


df.to_csv('/Users/priyanka/Downloads/dascena/berlin_ards.csv', sep=',', encoding='utf-8') #save as csv
exit(0)


df=pd.read_csv("/Users/priyanka/Downloads/dascena/berlin_ards.csv")
values=  np.array(df['feature_matrix'])

sysabp =[]


for i in range(len(values)):
    dummy = values[i].split("],")[0]
    dummy = dummy.strip("[")
    sysabp.append(dummy)

#sysabp = sysabp.interpolate(limit=2, limit_direction="forward");
sysabp = [i.replace("None","0.0") for i in sysabp]
#sysabp = [np.array(j).astype(np.float) for j in i.split(",") for i in sysabp]
sysabp1=[]
'''
for i in sysabp:
    temp=[]
    i=str(i).strip("[")
    i=i.strip("]")
    print(i)
    for j in i.split(","):
        temp.append(float(j))
    sysabp.append(temp)
'''
print(sysabp[0])
print(sysabp[1])
#plt.boxplot(sysabp)
#plt.show()
