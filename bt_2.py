#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:57:51 2023

@author: guangdafei
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cal_WMA(df: pd.DataFrame, index, n, col): #calculating weighted moving avg
    #df is the dataframe, index is the index (trade day), n is the rolling days, col is the column name
    wma = 0
    if index < n:
        n = index+1
    for i in range(n):
        w = (i+1)/(0.5*n*(n+1))
        wma += w * df.loc[index-n+1+i, col]
    
    return wma

standard = '自由流通市值加权连板比率剪刀差'

df21 = pd.read_csv('tradedate21-22.csv')
# df21 = pd.read_csv('tradedate23.csv')

print(df21)

df21['HMA_raw30'] = 0
df21['HMA_raw100'] = 0

for i in range(df21.index[-1]+1):
        n = 30;
        df21.loc[i,'HMA_raw30'] = 2 * cal_WMA(df21, i, int(n/2), standard) - cal_WMA(df21, i, n, standard)
        # HMA_raw100
        n = 100;
        df21.loc[i,'HMA_raw100'] = 2 * cal_WMA(df21, i, int(n/2), standard) - cal_WMA(df21, i, n, standard)

df21['HMA30'] = 0
df21['HMA100'] = 0
df21['return'] = 1
count = 0
for i in range(df21.index[-1]+1):
        n = 30;
        df21.loc[i,'HMA30'] = cal_WMA(df21, i, n, 'HMA_raw30')

        # HMA_raw100
        n = 100;
        df21.loc[i,'HMA100'] = cal_WMA(df21, i, n, 'HMA_raw100')
        
        if df21.loc[i,'HMA30'] / df21.loc[i,'HMA100'] > 1.15 and df21.loc[i,'HMA30'] > 0:
            df21.loc[i,'return'] = df21.loc[i-1,'return'] * (1+df21.loc[i,'pct_chg']/100)
            if i>300 and i < 400:
                print(df21.loc[i,'pct_chg'],'  ' ,df21.loc[i,'return'])
        
print(df21.loc[:,[standard,'HMA30','HMA100','return']])
plt.plot(df21.index, df21.loc[:,'return'])
plt.savefig(standard+'.png')
plt.show()

    


    
        

'''
0 1 2 3 4 5

i = 5, n = 30
n = 6



'''