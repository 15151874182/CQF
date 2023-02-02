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

##3 手动QQplot
df3=df[['D1_rtn','D5_rtn']]
df3['scaled_D1']=(df3['D1_rtn']-df3['D1_rtn'].mean())/df3['D1_rtn'].std()
df3['scaled_D5']=(df3['D5_rtn']-df3['D5_rtn'].mean())/df3['D5_rtn'].std()

df3['scaled_D1']=df3['scaled_D1'].sort_values().values ##不影响其它列情况下单独排序
df3['scaled_D5']=df3['scaled_D5'].sort_values().values
df3.index=[i for i in range(1,len(df3)+1)]
df3['density']=df3.index/len(df3)

from scipy.stats import norm
# q = norm.cdf(1.96)  #累计密度函数
# norm.ppf(q)  #累计密度函数的反函数
df3['Standard']=norm.ppf(df3['density'])
plt.plot(df3['Standard'],df3['scaled_D1'],label='scaled_D1')
plt.plot(df3['Standard'],df3['scaled_D5'],label='scaled_D5')
plt.title("scaled-D1-D5-QQplot", fontsize=22)
plt.xlabel('theoretical')
plt.ylabel('empirical')
plt.legend(fontsize=12)   
###直接调包QQplot
import statsmodels.api as sm
fig1 = sm.qqplot(df3['scaled_D1'], line='45')
fig2 = sm.qqplot(df3['scaled_D5'], line='45')
plt.show()

##histogram
df3=df3.iloc[:-1,:] ##最后一行的Standard是inf，要去掉
df3['scaled_D1'].plot(kind="hist",bins=20,color="blue",edgecolor='black',density=True,label="histogram")
#加核密度图
df3['scaled_D1'].plot(kind="kde",color="red",label="scaled_D1")
df3['Standard'].plot(kind="kde",color="green",label="Normal")

plt.xlabel("scaled_D1")
plt.ylabel("density")
plt.title("scaled_D1_distribution")
plt.legend()
plt.show()

