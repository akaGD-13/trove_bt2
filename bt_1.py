#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:21:09 2023

当指标大于x, 做多。小于y，做空
运行时间：取决于 maxi, step 的值
train = 2021, 2022
test = 2023

@author: guangdafei
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

for z in range(4):
    if z == 0:
        standard= '涨跌停比率剪刀差'
    elif z == 1:
        standard = '连板比率剪刀差'
    elif z==2:
        standard = '自由流通市值加权涨跌停比率剪刀差'
    else:
        standard = '自由流通市值加权连板比率剪刀差'
        
    print(standard)
    df21 = pd.read_csv('tradedate21-22.csv')
    # df21 = pd.read_csv('tradedate23.csv')
    
    maxi = 0.3
    step = 0.0009
    result = pd.DataFrame(np.arange(0, maxi, step)).merge(pd.DataFrame(np.arange(-maxi, 0, step)), how='cross')
    result = result.set_axis(['x','y'], axis=1)
    result['return'] = ''
    '''         x      y return
    0       0.000 -0.500       
    1       0.000 -0.499            
          ...    ...    ...     
    249998  0.499 -0.002       
    249999  0.499 -0.001     
    '''  
    for i in range(result.index[-1]+1):
        x = result.loc[i,'x']
        y = result.loc[i,'y']
        returns = 1
        for j in range(df21.index[-1]+1):
            if (df21.loc[j,standard] > x): # 做多
                returns = returns * (df21.loc[j,'pct_chg']/100 + 1)
            elif (df21.loc[j,standard] < y): # 做空
                returns = returns * -1 * (df21.loc[j,'pct_chg']/100 - 1)
        #ends inner for loop
        result.loc[i, 'return'] = returns
        if i%10000 == 0:
            print('Processing: i =', i)
        
    #ends outer for loop
    
    #find the best x and best y
    result = result.sort_values(by=['return'])
    print(result)
    # print(result.iloc[-1, 0:2])
    best_x = result.iloc[-1,0]
    best_y = result.iloc[-1,1]
    
    # test using 2023 data:
    df21 = pd.read_csv('tradedate23.csv')
    df21['return'] = ''
    df21['value'] = ''
    value = 1
    returns = 1
    for j in range(df21.index[-1]+1):
        if (df21.loc[j,standard] > best_x): # 做多
            returns = returns * (df21.loc[j,'pct_chg']/100 + 1)
        elif (df21.loc[j,standard] < best_y): # 做空
            returns = returns * -1 * (df21.loc[j,'pct_chg']/100 - 1)
        
        value = value * (df21.loc[j,'pct_chg']/100 + 1)
        df21.loc[j,'value'] = value
        
        df21.loc[j,'return'] = returns
    #ends for j loop
    plt.plot(df21.index, df21.loc[:,'return'], label=standard)
    plt.plot(df21.index, df21.loc[:,'value'], label='300')
    t = 'x, y are ' + str(best_x) + ', ' + str(best_y)
    plt.title(t)
    plt.legend()
    plt.savefig(standard+'1.png')
    plt.clf()


        