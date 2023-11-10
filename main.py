#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 00:38:09 2023

@author: guangdafei
"""

import tushare as ts
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']


token = "13bb0841c8a377221b39d9142f42bae2e2e9a897b9f692c75dd90d65"
ts.set_token(token)
pro = ts.pro_api()



def pre_process_stocklist():
    df = pd.read_csv('stocklist.csv')
    a = list(df.columns)
    a[0] = '1'
    # print(a)
    df = df.set_axis(a, axis=1).drop(columns='1')
    # print(df)
    
    to_remove = []
    for i in range(len(df.index)):
        if ('ST' in df.iloc[i,2] or 'PT' in df.iloc[i,2] ):
            to_remove.append(i)
    n = len(to_remove)
    for i in range(n):
        df = df.drop(index=to_remove[-i-1])     
    
    return df

def pro_process_open():
    return

stocklist = pre_process_stocklist();
# print(stocklist)

code = '000001.SZ';

df1 = pd.read_csv('temp');
df = df1.iloc[:, 2:4]
df = df.set_axis(['日期', code], axis=1)


for i in range(len(stocklist.index)):
    if i==0:
        continue
    code = stocklist.iloc[i,0]
    print(code)
    df1 = pro.query('daily', ts_code=code, start_date='20040429', end_date='20161130')
    df1 = df1.iloc[:, 2:4]
    df1 = df1.set_axis(['日期', code], axis=1)
    df.merge(df1, how='left', on='日期')
    # 失败, ValueError: Length of values (2865) does not match length of index (2910)
    # 决定用数据库里的数据

df.to_csv('open1.csv')