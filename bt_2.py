#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:57:51 2023

报告原文
https://pdf.dfcfw.com/pdf/H3_AP202011181430494641_1.pdf?1605686153000.pdf

只在HMA30/HMA100>1.15且HMA30和HMA100都大于0的情形下持有多仓，剩下空仓

HMA计算方法：
https://school.stockcharts.com/doku.php?id=technical_indicators:hull_moving_average

运行时间：一分钟内

@author: guangdafei
"""

import math
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

standard= '涨跌停比率剪刀差'
# standard = '连板比率剪刀差'
# standard = '自由流通市值加权涨跌停比率剪刀差'
# standard = '自由流通市值加权连板比率剪刀差'

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

returns = 1
for i in range(df21.index[-1]+1):
        if i < 99:
            continue;
        n = 30;
        df21.loc[i,'HMA30'] = cal_WMA(df21, i, int(math.sqrt(n)), 'HMA_raw30')

        # HMA_raw100
        n = 100;
        df21.loc[i,'HMA100'] = cal_WMA(df21, i, int(math.sqrt(n)), 'HMA_raw100')
        
        if df21.loc[i,'HMA30'] / df21.loc[i,'HMA100'] > 1.15 and df21.loc[i,'HMA30'] > 0:
            returns = returns * (1 + df21.loc[i,'pct_chg']/100)
        # otherwise returns not change
        
        df21.loc[i,'return'] = returns

        
print(df21.loc[:,[standard,'HMA30','HMA100','return']])
plt.plot(df21.index[99:], df21.loc[99:df21.index[-1]+1,'return'])
plt.savefig(standard+'2.png')
plt.show()

    


    
        

'''
0 1 2 3 4 5

i = 5, n = 30
n = 6



'''