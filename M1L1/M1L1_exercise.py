import os,sys
import datetime
import math
import scipy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib
import random
np.random.seed(0)##固定随机种子，使实验可复现

df=pd.read_csv('CQF_January_2023_M1L1_Excel.csv')
df=df[['Date','Adj Close']]
df=df.dropna()
df.columns=['date','price']

##1
df['D1_rtn']=(df['price'].shift(-1)-df['price'])/df['price']
df['D2_rtn']=(df['price'].shift(-2)-df['price'])/df['price']
df['D5_rtn']=(df['price'].shift(-5)-df['price'])/df['price']
df=df.dropna()

D1_rtn_std=df['D1_rtn'].std() ##0.00603
Adjust_D2_rtn_std=df['D2_rtn'].std()/(2**0.5) ##0.00609
Adjust_D5_rtn_std=df['D5_rtn'].std()/(5**0.5) ##0.00609
print('1 ##########')
print('D1_rtn_std:',D1_rtn_std)
print('Adjust_D2_rtn_std:',Adjust_D2_rtn_std)
print('Adjust_D5_rtn_std:',Adjust_D5_rtn_std)
##结论：三者相近

##2
df2=df['D1_rtn']
df2=df2.sample(frac=1) ##Re-shuffle
mid=len(df2)//2
A=df2.iloc[:mid] ##上半部分
B=df2.iloc[mid:] ##下半部分

A_u=A.mean()##1.9e-5
A_std=A.std()## 0.0062
B_u=B.mean()##1.2e-4
B_std=B.std()##0.0058
print('2 ##########')
print('A_u:',A_u)
print('B_u:',B_u)
print('A_std:',A_std)
print('B_std:',B_std)
##结论：上下部分的u差很多，std较相似
##但是理论上服从同一分布的数据打乱分成2部分后，还是服从同一分布，猜测可能数据量不够大，或者存在离群值

