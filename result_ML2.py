#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:31:29 2023

machine learning on HMA30/HMA100

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




df21 = pd.read_csv('tradedate21-22.csv')
df23 = pd.read_csv('tradedate23.csv') # for tesitng

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

for i in range(df23.index[-1]+1):
    if i < 100:
        continue
    if df23.loc[i-1,'tree_result'] == True: # long
        value = value * (1 + df23.loc[i, 'pct_chg']/100)
    else:
        if df23.loc[i, 'pct_chg'] != -1: # short
            value = value * (1 - df23.loc[i, 'pct_chg']/100)
    df23.loc[i, 'return'] = value;
    hs300 = hs300 * (1 + df23.loc[i, 'pct_chg']/100)
    df23.loc[i, 'hs300'] = hs300

plt.plot(df23.index[100:], df23.loc[100:,'return'], label='model')
plt.plot(df23.index[100:], df23.loc[100:,'hs300'], label='HS300')
plt.title('Logistic Regression')
plt.legend()
plt.savefig('Logistic_Regression2.png')
plt.clf()    