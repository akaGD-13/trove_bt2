#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:31:29 2023

machine learning on HMA30/HMA100
folliowng the report:
    https://mp.weixin.qq.com/s?__biz=MzI1Mjg3MzYyOA==&mid=2247502995&idx=1&sn=d899e3efca1cc1b4a41307e90110f306&chksm=e9df9260dea81b769ed30cac53b49e1b8a14bdaabea7a008d4890846aaf1aabca955751d4374&mpshare=1&scene=1&srcid=110685Q1vJViDZWbq95WfJMs&sharer_shareinfo=172efbe8769055c3805337fb7d2e895f&sharer_shareinfo_first=172efbe8769055c3805337fb7d2e895f&version=4.1.9.90740&platform=mac#rd

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
    for z in range(4):
        if z == 0:
            standard= '涨跌停比率剪刀差'
        elif z == 1:
            standard = '连板比率剪刀差'
        elif z==2:
            standard = '自由流通市值加权涨跌停比率剪刀差'
        else:
            standard = '自由流通市值加权连板比率剪刀差'
    
        # df21 = pd.read_csv('tradedate21-22.csv')
       
        
        # print(df21)
        
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
                    
                if df21.loc[i, '30/100' + standard] > 1.15 and df21.loc[i,'HMA100'] > 0:
                    df21.loc[i,'1.15' + standard] = True
    # all HMA30/HMA100 are calculated and stored in column: '30/100'+standard
    return df21




df21 = pd.read_csv('tradedate09-23.csv')
# df23 = pd.read_csv('tradedate23.csv') # for tesitng
df23 = df21

df21 = cal_HMA30_HMA100(df21)
df23 = cal_HMA30_HMA100(df23)

#regression
import statsmodels.api as sm
from statsmodels.formula.api import ols


# regression
size = df21.index[-1] + 1
X = df21.loc[:, ['30/100'+'自由流通市值加权涨跌停比率剪刀差', '30/100'+'自由流通市值加权连板比率剪刀差']].drop(index=size-1).reset_index(drop=True).fillna(0)
y = df21.loc[:, 'pct_chg'].drop(index=0).reset_index(drop=True).fillna(0)
lm_fit_linear1 = sm.OLS(y, sm.add_constant(X), missing='drop').fit()
print('Regression: ')
print(lm_fit_linear1.summary())
print()

# Decision Tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

df21['up_or_down'] = False

for i in range(df21.index[-1]+1):
    if df21.loc[i, 'pct_chg'] > 0:
        df21.loc[i, 'up_or_down'] = True

# decision tree 1
y = df21.loc[:, 'up_or_down'].drop(index=0).reset_index(drop=True).fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.7, stratify=y)

model_tree = DecisionTreeClassifier(criterion='gini', random_state=42)
model_tree.fit(X_train, y_train)

y_pred = model_tree.predict(X_test)


print('Decision Tree 1 : ')
print('accuracy:', model_tree.score(X_test, y_test))
print(classification_report(y_test, y_pred))
print('depth:', model_tree.get_depth())
print()



#logistic regression
from sklearn.linear_model import LogisticRegression


model_logreg = LogisticRegression(max_iter=500)
model_logreg.fit(X_train, y_train)

y_pred = model_logreg.predict(X_test)
print('Logistic Regression result: ')
print(model_logreg.score(X_test, y_test))
print()


# decision tree 2
X = df21.loc[:, ['1.15'+'自由流通市值加权涨跌停比率剪刀差', '1.15'+'自由流通市值加权连板比率剪刀差']].drop(index=size-1).reset_index(drop=True).fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.7, stratify=y)

model_tree2 = DecisionTreeClassifier(criterion='gini', random_state=42)
model_tree2.fit(X_train, y_train)

y_pred = model_tree2.predict(X_test)

print('Decision Tree 2 : ')
print('accuracy:', model_tree2.score(X_test, y_test))
print(classification_report(y_test, y_pred))
print('depth:', model_tree.get_depth())
print()




#testing
df23['tree_result'] = model_logreg.predict(df23.loc[:,  ['30/100自由流通市值加权涨跌停比率剪刀差', '30/100自由流通市值加权连板比率剪刀差']].fillna(0))

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
    if df23.loc[i-1,'tree_result'] == True: # long
        temp = value # 储存前一天的
        value = value * (1 + df23.loc[i, 'pct_chg']/100)
        if kai_or_ping == False: #还未开仓
            kai_or_ping = True
            trade_count += 1 
            temp_value = temp # 记录开仓时的value
        if i == df23.index[-1] and kai_or_ping == True: # last day 并且 开仓中
            # print(value,'-', temp_value,'=', value-temp_value)
            if value - temp_value > 0:
                v_count += 1
    else:
        if kai_or_ping == True: # 如果没平仓 or last day
            kai_or_ping = False # 平仓
            # print(value,'-', temp_value,'=', value-temp_value) # 还有一行print在上面，请一起comment/uncomment
            if value - temp_value > 0: #平仓 - 开仓 > 0 盈利
                v_count +=1
    #     if df23.loc[i, 'pct_chg'] != -1: # short
    #         value = value * (1 - df23.loc[i, 'pct_chg']/100)
    df23.loc[i, 'return'] = value;
    hs300 = hs300 * (1 + df23.loc[i, 'pct_chg']/100)
    df23.loc[i, 'hs300'] = hs300

plt.plot(df23.index[100:], df23.loc[100:,'return'], label='model')
plt.plot(df23.index[100:], df23.loc[100:,'hs300'], label='HS300')
plt.title('Logistic Regression')
plt.legend()
plt.savefig('Logistic_Regression2_09-23.png')
plt.clf()    

# 计算指标
max_ = -999
min_ = 999
profit_sum = 0
loss_sum = 0
df23['rate_of_return'] = 0

for index in range(df23.index[-1]+1):
    if index < 100:
        continue
    if df23.loc[index, 'return'] > max_:
        max_ = df23.loc[index, 'return']
    if df23.loc[index, 'return'] < min_:
        min_ = df23.loc[index, 'return']
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
max_drawdown = (max_ - min_)/max_

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