#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:24:57 2023

前一天的数据：推波助澜index_weight.csv (用来继承index_weight和计算连板) 需要update
从2023年1月3日到现在的数据：推波助澜.csv 需要添加一行

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


def cal_HMA30_HMA100(df21: pd.DataFrame):
    for z in range(5):
        if z == 0:
            standard= '涨跌停比率剪刀差'
        elif z == 1:
            standard = '连板比率剪刀差'
        elif z==2:
            standard = '自由流通市值加权涨跌停比率剪刀差'
        elif z==3:
            standard = '自由流通市值加权连板比率剪刀差'
        else:
            standard = '自由流通市值加权地天与天地板比率剪刀差'
        
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
        df21['value'] = 1
        df21['30/100' + standard] = 0;
        df21['1.15' + standard] = False;
        # 'return' 是净值，不是rate of return

        for i in range(df21.index[-1]+1):
                if i < 99:
                    continue;
                n = 30;
                df21.loc[i,'HMA30'] = cal_WMA(df21, i, int(math.sqrt(n)), 'HMA_raw30')
        
                n = 100;
                df21.loc[i,'HMA100'] = cal_WMA(df21, i, int(math.sqrt(n)), 'HMA_raw100')
                
                if  df21.loc[i,'HMA100'] != 0:
                    df21.loc[i, '30/100' + standard] = df21.loc[i-1,'HMA30'] / df21.loc[i-1,'HMA100']
                else:
                    df21.loc[i, '30/100' + standard] = 0
                    
                if df21.loc[i,'HMA100'] > 0 and df21.loc[i, '30/100' + standard] > 1.15:
                    df21.loc[i,'1.15' + standard] = True
    # all HMA30/HMA100 are calculated and stored in column: '30/100'+standard
    return df21




# df21 = pd.read_csv('tradedate09-23_tdb.csv')
# df23 = pd.read_csv('tradedate23_tdb.csv') # for tesitng
# # df23 = df21

# df21 = cal_HMA30_HMA100(df21)
# df23 = cal_HMA30_HMA100(df23)

# df21.to_csv('tradedate09-23_hma.csv')
# df23.to_csv('tradedate23_hma.csv')
df23 = pd.read_csv('推波助澜.csv')
df23 = df23.iloc[:,1:]

df23 = cal_HMA30_HMA100(df23)
df23.to_csv('推波助澜.csv')
df21 = df23
size = df21.index[-1] + 1
  

X2 = df21.loc[:, ['1.15'+'自由流通市值加权连板比率剪刀差',]].drop(index=size-1).reset_index(drop=True).fillna(0)

X5 = df21.loc[:, ['1.15'+'连板比率剪刀差']].drop(index=size-1).reset_index(drop=True).fillna(0)


df23['return'] = 1
df23['hs300'] = 1
value = 1
hs300 = 1
kai_or_ping = False # True：开仓，False： 平仓中
trade_count = 0
v_count = 0
temp_value = 0

for i in range(df23.index[-1]+1):
    if i < 100:
        continue
     
    if X2.iloc[i-1,0]: # 涨跌停1.15
        temp = value # 储存前一天的
        # if X2.iloc[i-1,0] == True: # 连板1.15
        #     if X3.iloc[i-1,0] == True: # 地天板1.15
        #         value = value * (1 + df23.loc[i, 'pct_chg']/100 * 1.0)
        #     else:
        #         value = value * (1 + df23.loc[i, 'pct_chg']/100 * 0.9)
        # else:          
        #     value = value * (1 + df23.loc[i, 'pct_chg']/100 * 0.8)
        value = value * (1 + df23.loc[i, 'pct_chg']/100 * 1.0)  
        if kai_or_ping == False: #还未开仓
            kai_or_ping = True
            trade_count += 1 
            temp_value = temp # 记录开仓时的value
        if i == df23.index[-1] and kai_or_ping == True: # last day 并且 开仓中
            # print(value,'-', temp_value,'=', value-temp_value)
            if value - temp_value > 0:
                v_count += 1
    else:
        if kai_or_ping == True: # 如果没平仓
            kai_or_ping = False # 平仓
            # print(value,'-', temp_value,'=', value-temp_value) # 还有一行print在上面，请一起comment/uncomment
            if value - temp_value > 0: #平仓 - 开仓 > 0 盈利
                v_count +=1
    #     if df23.loc[i, 'pct_chg'] != -1: # short
    #         value = value * (1 - df23.loc[i, 'pct_chg']/100)
    df23.loc[i, 'return'] = value;
    hs300 = hs300 * (1 + df23.loc[i, 'pct_chg']/100)
    df23.loc[i, 'hs300'] = hs300

df23.to_csv('推波助澜.csv')

plt.plot(df23.index[100:], df23.loc[100:,'return'], label='model')
plt.plot(df23.index[100:], df23.loc[100:,'hs300'], label='HS300')
plt.title('Market Timing')
plt.legend()
plt.savefig('推波助澜.png')
plt.clf()    

# 计算指标
max_ = -999
dd = 0
max_drawdown = 0;
profit_sum = 0
loss_sum = 0
df23['rate_of_return'] = 0

for index in range(df23.index[-1]+1):
    if index < 100:
        continue
    if df23.loc[index, 'return'] > max_:
        max_ = df23.loc[index, 'return']
    dd = (max_ - df23.loc[index, 'return'])/max_
    if dd > max_drawdown:
        max_drawdown = dd;
    if df23.loc[index, 'return'] > df23.loc[index-1, 'return']:
        profit_sum += df23.loc[index, 'return'] - df23.loc[index-1, 'return']
    else:
        loss_sum -=  df23.loc[index, 'return'] - df23.loc[index-1, 'return']
    
    df23.loc[index, 'rate_of_return'] =  (df23.loc[index, 'return'] - df23.loc[index-1, 'return']) / df23.loc[index-1, 'return']

#年化收益0
days = len(df23.index[100:])
year = days/242
yearly_return = pow(df23.loc[99+days, 'return']/df23.loc[100,'return'], 1/year) - 1
# 最大回撤
if max_drawdown < 0: # although impossible due to the calculation process
    max_drawdown = 0;

# 胜率 （
v_ratio = v_count / trade_count
# 盈亏比 = avg_profit/avg_loss
profit_loss_ratio = profit_sum /loss_sum
#夏普比率
risk_free_return = 0 # 无风险利率为0 可调整为其他
yearly_volatility = np.std(df23.loc[100:, 'rate_of_return']) * np.sqrt(250)
sharpe_ratio = yearly_return / yearly_volatility
# 年均交易次数 
yearly_trade_count = trade_count / year
  
title = 'yearly_return = ' + str(yearly_return) + '\n'
# title += 'yearly_volatility = ' + str(yearly_volatility) + '\n'
title += 'max_drawdown: ' + str(max_drawdown) + '\n' 
title += '胜率: ' + str(v_ratio) + '\n' 
title += '盈亏比: ' + str(profit_loss_ratio) + '\n'
title += 'sharp ratio: ' + str(sharpe_ratio) + '\n'
title += 'yearly trade count: ' + str(yearly_trade_count)

print(title)
