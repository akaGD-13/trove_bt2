#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:00:53 2023

Maching Learning: Regression, Decision Tree

@author: guangdafei
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#regression
import statsmodels.api as sm
from statsmodels.formula.api import ols

df21 = pd.read_csv('tradedate09-23_tdb.csv')
df23 = pd.read_csv('tradedate23_tdb.csv')

# regression
size = df21.index[-1] + 1
# X = df21.loc[:, ['自由流通市值加权涨跌停比率剪刀差', '自由流通市值加权连板比率剪刀差','自由流通市值加权地天与天地板比率剪刀差']].drop(index=size-1).reset_index(drop=True)
X = df21.loc[:, ['自由流通市值加权涨跌停比率剪刀差', '自由流通市值加权连板比率剪刀差']].drop(index=size-1).reset_index(drop=True)
y = df21.loc[:, 'pct_chg'].drop(index=0).reset_index(drop=True)
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

y = df21.loc[:, 'up_or_down'].drop(index=0).reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)

model_tree = DecisionTreeClassifier(criterion='gini')
model_tree.fit(X_train, y_train)

y_pred = model_tree.predict(X_test)



print('Decision Tree: ')
print('training set accuracy: ', model_tree.score(X_train, y_train))
print('testing accuracy:', model_tree.score(X_test, y_test))
print(classification_report(y_test, y_pred))
print('depth:', model_tree.get_depth())
print()

#logistic regression
from sklearn.linear_model import LogisticRegression


model_logreg = LogisticRegression(max_iter=500)
model_logreg.fit(X_train, y_train)

y_pred = model_logreg.predict(X_test)
print('Logistic Regression result: ')
print('training set accuracy: ', model_logreg.score(X_train, y_train))
print('testing accuracy: ',model_logreg.score(X_test, y_test))
print()


# testing
# df23['tree_result'] = model_tree.predict(df23.loc[:,  ['自由流通市值加权涨跌停比率剪刀差', '自由流通市值加权连板比率剪刀差','自由流通市值加权地天与天地板比率剪刀差']])
df23['tree_result'] = model_logreg.predict(df23.loc[:,  ['自由流通市值加权涨跌停比率剪刀差', '自由流通市值加权连板比率剪刀差']])

# print(y_pred)

df23['return'] = 1
df23['hs300'] = 1
value = 1
hs300 = 1

# 指标相关
max_ = -999
dd = 0
max_drawdown = 0
profit_sum = 0
loss_sum = 0
df23['rate_of_return'] = 0

for i in range(df23.index[-1]+1):
    if i == 0:
        continue
    if df23.loc[i-1,'tree_result'] == True: # long
        value = value * (1 + df23.loc[i, 'pct_chg']/100)
    else:
        if df23.loc[i, 'pct_chg'] != -1: # short
            value = value * (1 - df23.loc[i, 'pct_chg']/100)
    df23.loc[i, 'return'] = value;
    hs300 = hs300 * (1 + df23.loc[i, 'pct_chg']/100)
    df23.loc[i, 'hs300'] = hs300
    
    # 计算指标
    if df23.loc[i, 'return'] > max_:
        max_ = df23.loc[i, 'return']
    dd = (max_ - df23.loc[i, 'return'])/max_
    if dd > max_drawdown:
        max_drawdown = dd;
    if df23.loc[i, 'return'] > df23.loc[i-1, 'return']:
         profit_sum += df23.loc[i, 'return'] - df23.loc[i-1, 'return']
    else:
         loss_sum -=  df23.loc[i, 'return'] - df23.loc[i-1, 'return']
     
    df23.loc[i, 'rate_of_return'] =  (df23.loc[i, 'return'] - df23.loc[i-1, 'return']) / df23.loc[i-1, 'return']
    

plt.plot(df23.index, df23.loc[:,'return'], label='model')
plt.plot(df23.index, df23.loc[:,'hs300'], label='HS300')
plt.title('result_ML1_09-23')
plt.legend()
plt.savefig('result_ML1_09-23.png')
plt.clf()      

#年化收益0
days = len(df23.index[100:])
year = days/242
yearly_return = pow(df23.loc[99+days, 'return']/df23.loc[100,'return'], 1/year) - 1
# 最大回撤
if max_drawdown < 0: # although impossible due to the calculation process
    max_drawdown = 0;

# 胜率 （
# v_ratio = v_count / trade_count
# 盈亏比 = avg_profit/avg_loss
profit_loss_ratio = profit_sum /loss_sum
#夏普比率
risk_free_return = 0 # 无风险利率为0 可调整为其他
yearly_volatility = np.std(df23.loc[100:, 'rate_of_return']) * np.sqrt(250)
sharpe_ratio = yearly_return / yearly_volatility
# 年均交易次数 
# yearly_trade_count = trade_count / year
    
title = 'yearly_return = ' + str(yearly_return) + '\n'
# title += 'yearly_volatility = ' + str(yearly_volatility) + '\n'
title += 'max_drawdown: ' + str(max_drawdown) + '\n' 
# title += '胜率: ' + str(v_ratio) + '\n' 
title += '盈亏比: ' + str(profit_loss_ratio) + '\n'
title += 'sharp ratio: ' + str(sharpe_ratio) + '\n'
# title += 'yearly trade count: ' + str(yearly_trade_count)

print(title)