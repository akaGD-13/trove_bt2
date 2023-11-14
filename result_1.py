#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 23:31:38 2023

calculate the indexs of method 1 (bt_1.py)

@author: guangdafei
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('result from bt_1')
for z in range(4):
    if z == 0:
        standard= '涨跌停比率剪刀差'
    elif z == 1:
        standard = '连板比率剪刀差'
    elif z==2:
        standard = '自由流通市值加权涨跌停比率剪刀差'
    else:
        standard = '自由流通市值加权连板比率剪刀差'
        
    df21 = pd.read_csv(standard+'return.csv')
    
    max = -999
    min = 999
    v_count = 0 #获胜次数
    profit_sum = 0 #盈利金额
    loss_sum = 0 #亏损金额
    df21['rate_of_return'] = 0
    # 计算指标
    for i in range(len(df21.index[1:])):
        index = i + 1
        if df21.loc[index, 'return'] > max:
            max = df21.loc[index, 'return']
        if df21.loc[index, 'return'] < min:
            min = df21.loc[index, 'return']
        if df21.loc[index, 'return'] > df21.loc[index-1, 'return']:
            v_count +=1
            profit_sum += df21.loc[index, 'return'] - df21.loc[index-1, 'return']
        else:
            loss_sum -=  df21.loc[index, 'return'] - df21.loc[index-1, 'return']
        
        df21.loc[index, 'rate_of_return'] =  (df21.loc[index, 'return'] - df21.loc[index-1, 'return']) / df21.loc[index-1, 'return']

    #年化收益0
    days = len(df21.index[1:])
    year = days/242
    yearly_return = pow(df21.loc[days, 'return']/df21.loc[1,'return'], 1/year) - 1
    # 最大回撤
    max_drawdown = (max - min)/max

    # 胜率 （暂时不清楚计算方法 不确定什么是交易次数 连续两天都是做多 算两次还是一次？
    v_ratio = v_count / days
    
    # 盈亏比 = avg_profit/avg_loss
    if loss_sum == 0:
        profit_loss_ratio = 0
    else:
        profit_loss_ratio = profit_sum /loss_sum
    #夏普比率
    risk_free_return = 0 # 无风险利率为0 可调整为其他
    yearly_volatility = np.std(df21.loc[1:, 'rate_of_return']) * np.sqrt(250)
    if yearly_volatility == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = yearly_return / yearly_volatility
    #交易次数
        
    title = 'yearly_return = ' + str(yearly_return) + '\n'
    title += 'yearly_volatility = ' + str(yearly_volatility) + '\n'
    title += '盈亏比: ' + str(profit_loss_ratio) + '\n'
    title += 'max_drawdown: ' + str(max_drawdown) + '\n' 
    # title += '胜率: ' + str(v_ratio) + '(not accurate, calculating method needs verification)\n' 
    title += 'sharp ratio: ' + str(sharpe_ratio)
    
    print(standard + ":")
    print(title)
    print()
        
    