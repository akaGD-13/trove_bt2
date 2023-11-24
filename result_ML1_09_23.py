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

df21 = pd.read_csv('tradedate09-23.csv')
df23 = pd.read_csv('tradedate23.csv')

# regression
size = df21.index[-1] + 1
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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.7, stratify=y)

model_tree = DecisionTreeClassifier(criterion='gini', random_state=42)
model_tree.fit(X_train, y_train)

y_pred = model_tree.predict(X_test)

df23['tree_result'] = model_tree.predict(df23.loc[:,  ['自由流通市值加权涨跌停比率剪刀差', '自由流通市值加权连板比率剪刀差']])

# print(y_pred)

print('Decision Tree: ')
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

df23['return'] = 1
df23['hs300'] = 1
value = 1
hs300 = 1

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

plt.plot(df23.index, df23.loc[:,'return'], label='model')
plt.plot(df23.index, df23.loc[:,'hs300'], label='HS300')
plt.title('decision tree')
plt.legend()
plt.savefig('decision_tree1_09-23.png')
plt.clf()      