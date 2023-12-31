#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:46:38 2023

草稿

@author: guangdafei
"""

import tushare as ts
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']


token = "13bb0841c8a377221b39d9142f42bae2e2e9a897b9f692c75dd90d65"
ts.set_token(token)
pro = ts.pro_api()

# start = '20091009'
# end = '20230928'
# # remember to also change the file path when saving at the end of this program
date = datetime.now().date().strftime("%Y%m%d")
index_weight = pd.read_csv('index_weight.csv')
index_weight = index_weight.iloc[:,1:]

df23 = pd.read_csv('推波助澜.csv')
df23 = df23.iloc[:,1:]

print(df23)

while int(df23.iloc[-1,0]) >= int(date):
    df23 = df23.iloc[:-1,:]
    
print(df23)


# # start = '20220801' # approximately 100 trade date before 20230101 (to calculate HMA100)
# # end = '20230928'

# start = '20220801' #  100 trade date before 20230103 （first tradedate of 2023) (to calculate HMA100)
# end = '20231228'
# index_name= '399300.SZ'
# # df = pro.index_weight(index_code=index_name, start_date=start, end_date=start)
# # print(df)

# df23 = pd.read_csv('推波助澜.csv')
# df23 = df23.iloc[:,1:]
# print(len(df23.loc[0,:]))

# date = '20231228'
# returns = pro.index_daily(ts_code=index_name, start_date=date, end_date=date)
# change = returns.loc[0,'pct_chg']
# print(returns)
# print(change)
# zt = 0 #涨停比率 
# dt = 0# 跌停比率
# zt_c = 0 #连板涨停比率
# dt_c = 0 # 连板跌停比率
# tdb = 0 #地天板比率
# dtb = 0#天地板比率
# zt_w = 0# 加权涨停比率
# dt_w = 0# 加权跌停比率
# zt_cw = 0 # 加权连板涨停比率
# dt_cw = 0 # 加权连板跌停比率
# dtb_w = 0# 加权地天板比率
# tdb_w = 0# 加权天地板比率

# df23.loc[len(df23.index)] = [date, zt, dt, dtb, tdb, dtb-tdb, zt-dt, zt_c-dt_c, zt_w-dt_w, zt_cw-dt_cw, dtb_w-tdb_w, change, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# print(df23.loc[len(df23.index)-1,:])
#获取日期信息
# tradedate = pro.query('daily', ts_code='000425.SZ' , start_date="20220804", end_date="20220804")
# print(tradedate.empty())
# tradedate = pro.query('daily', ts_code='600519.SH' , start_date='20161029', end_date='20161130')
# df = df.set_index('con_code')
# print(df)
# df['pct_chg'] = 0
# df_temp = pro.index_weight(index_code='399300.SZ', start_date='20230201', end_date='20230201')
# # print(df_temp)
# df = df_temp.merge(df.loc[:,['pct_chg']], how='left', right_on=df.index, left_on='con_code')
# df.fillna(0, inplace=True)
# print(df.iloc[:,[1,2,3,4,5]])

# start_date = datetime(2023, 1, 3)  # 假设2023年1月3日是一个交易日

# # 计算向前25*7个自然日的日期
# result_date = start_date - timedelta(days=20 * 7)

# print("向前20*7个自然日的日期是:", result_date.strftime("%Y-%m-%d"))

# tradedate = pro.query('daily', ts_code='600519.SH' , start_date='20161029', end_date='20161130')
# tradedate = tradedate.iloc[:,1:3]
# size = tradedate.index[-1]
# tradedate['涨停比率'] = ''
# tradedate = tradedate.iloc[:,[0,2]]
# tradedate.iloc[0,1] = 1
# print(tradedate)

# df = pro.index_weight(index_code='399300.SZ', start_date='20141030', end_date='20141030')
# print(df)
# if df.empty or df.index[-1] != 299:
    # print('yes')
#     #     index_code   con_code trade_date  weight
#     # 0    399300.SZ  600519.SH   20230901  6.2310
#     # 1    399300.SZ  300750.SZ   20230901  3.3419
#     # 2    399300.SZ  601318.SH   20230901  2.9086
#     # ..         ...        ...        ...     ...
#     # 299  399300.SZ  001289.SZ   20230901  0.0162
# codelist = df.iloc[:,1]

# for j in range(300):
#     code = codelist[j]
#     print(code)

# tradedate = pro.query('daily', ts_code='600519.SH' , start_date='20161129', end_date='20161130')
# print(tradedate)
# start = '20161029'
# end = '20161130'
# tradedate = pro.query('daily', ts_code='600519.SH' , start_date=start, end_date=end)
# tradedate = tradedate.iloc[:,1:3]
# tradedate = tradedate.sort_values(by=['trade_date'])
# tradedate = tradedate.reset_index()
# tradedate['涨停比率'] = '' #1
# tradedate = tradedate.iloc[:,[1,3]]
# print(tradedate)
# print(tradedate.iloc[0,0])

# start = '20210101'
# end = '20221231'
# # code = '600519.SH'
# # returns = pro.daily(ts_code='399300.SZ', start_date=start, end_date=end)
# returns = pro.index_daily(ts_code='399300.SZ', start_date=start, end_date=end)
# returns = returns.loc[:,['trade_date','pct_chg']]
# for i in range(100):
#     j = i + 300
#     print(returns.loc[j,'pct_chg'])
# print(returns)
# df =pro.index_weight(index_code='399300.SZ', start_date=start, end_date=start)
#         #     index_code   con_code trade_date  weight
#         # 0    399300.SZ  600519.SH   20230901  6.2310
#         # 1    399300.SZ  300750.SZ   20230901  3.3419
#         # 2    399300.SZ  601318.SH   20230901  2.9086
#         # ..         ...        ...        ...     ...
#         # 299  399300.SZ  001289.SZ   20230901  0.0162
# # print(df)
# df['prev'] = 0
# df['pct_chg'] = 0
# print(df)



# codelist = df.iloc[:,1]
# allcode = ''

# for x in range(300):
#     allcode = codelist[x] + ',' + allcode
# # print(allcode)
# df1 = ts.pro_bar(ts_code=code, adj='qfq', start_date=start, end_date=start)
# # df2 = pro.limit_list_d(ts_code=allcode, trade_date=start, fields='ts_code,limit') #涨停/跌停
# # df1 = pro.bak_daily(ts_code=allcode, trade_date='20211012', fields='ts_code,pct_change,swing') # 涨幅，振幅
# df1 = pro.daily(ts_code=allcode, start_date=start, end_date=start) # 涨幅
# df1 = df1.loc[:,['ts_code','pct_chg']]

# df1 = pd.DataFrame(codelist).merge(df1, how='left', left_on='con_code', right_on='ts_code')
# # df1 = df1.merge(df2, how='left', left_on='con_code', right_on='ts_code')
# df1.fillna(0,inplace=True)
# df1= df1.set_index('con_code')
# df1 = df1.loc[:,['pct_chg']]
# # df1 = pro.bak_daily(trade_date='20211012', fields='trade_date,ts_code,name,close,open')
# # df_copy = df1.copy()
# # for i in range(300):
# #     print(list(df1.iloc[i,:]))
# print(df1)