#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 00:45:24 2023

提取数据
运行时间：两年 = 10分钟左右

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

index_name= '399300.SZ'

start = '20091009'
end = '20230928'
# remember to also change the file path when saving at the end of this program

# start = '20220801' # approximately 100 trade date before 20230101 (to calculate HMA100)
# end = '20230928'

#获取日期信息
tradedate = pro.query('daily', ts_code='600519.SH' , start_date=start, end_date=end)
tradedate = tradedate.iloc[:,1:3]
tradedate = tradedate.sort_values(by=['trade_date']) #排序
tradedate = tradedate.reset_index()
tradedate['涨停比率'] = '' #1
tradedate = tradedate.iloc[:,[1,3]] #只要‘trade_date'和’涨停比率‘
tradedate['跌停比率'] = '' #2
tradedate['地天板比率'] = '' #3
tradedate['天地板比率'] = '' #4
tradedate['天地板比率剪刀差'] = '' #5
tradedate['涨跌停比率剪刀差'] = '' #6
tradedate['连板比率剪刀差'] = '' #7
tradedate['自由流通市值加权涨跌停比率剪刀差'] = '' #8
tradedate['自由流通市值加权连板比率剪刀差'] = '' #9
tradedate['自由流通市值加权地天与天地板比率剪刀差'] = '' #10
size = tradedate.index[-1]+1

df = pro.index_weight(index_code=index_name, start_date=start, end_date=start)
# df = df.set_index('con_code')
df['pct_chg'] = 0
df['prev'] = 0
print(df)

for i in range(size): # loop through all date 
    #都去除百分号
    zt = 0 #涨停比率 
    dt = 0# 跌停比率
    zt_c = 0 #连板涨停比率
    dt_c = 0 # 连板跌停比率
    tdb = 0 #地天板比率
    dtb = 0#天地板比率
    zt_w = 0# 加权涨停比率
    dt_w = 0# 加权跌停比率
    zt_cw = 0 # 加权连板涨停比率
    dt_cw = 0 # 加权连板跌停比率
    dtb_w = 0# 加权地天板比率
    tdb_w = 0# 加权天地板比率
    
    date = tradedate.iloc[i,0] #日期
    print('date is: ' + date)
    # 获取index weighth
    df_temp =pro.index_weight(index_code=index_name, start_date=date, end_date=date)
        #     index_code   con_code trade_date  weight
        # 0    399300.SZ  600519.SH   20230901  6.2310
        # 1    399300.SZ  300750.SZ   20230901  3.3419
        # 2    399300.SZ  601318.SH   20230901  2.9086
        # ..         ...        ...        ...     ...
        # 299  399300.SZ  001289.SZ   20230901  0.0162
    
    if df_temp.empty or df_temp.index[-1] != 299: # size is not 300: empty or missing stocks #use the original df, update prev
        print("empty")
    else: # not empty, update df
        print('not empty')
        df = df_temp.merge(df.loc[:,['con_code','prev','pct_chg']], how='left', left_on = 'con_code', right_on='con_code')
        # df = df.set_index('con_code')
        
    df.fillna(0, inplace=True) # if some stock remove/added set their pct_chg and prev as 0
    df.loc[:,'prev'] = df.loc[:,'pct_chg']  #update prev  
    codelist = df.loc[:,'con_code']
    allcode = ''
    
    for x in range(300): #提取所有股票代码
        allcode = codelist[x] + ',' + allcode
    #ends this for loop
        
    # 提取这些stock的pct_change, swing, limit(涨跌停类型)
    # df1 = pro.bak_daily(ts_code=allcode, trade_date=date, fields='ts_code,pct_change,swing') # 涨幅，振幅
    # df2 = pro.limit_list_d(ts_code=allcode, trade_date=date, fields='ts_code,limit')  #涨停/跌停
    df1 = pro.daily(ts_code=allcode, start_date=date, end_date=date) # 涨幅
    df1 = df1.loc[:,['ts_code','pct_chg']]
    
    df1 = pd.DataFrame(codelist).merge(df1, how='left', left_on='con_code', right_on='ts_code') 
    
    # df1 = df1.merge(df2, how='left', left_on='con_code', right_on='ts_code')
    df1.fillna(0,inplace=True)
    df1= df1.set_index('con_code')
    df1 = df1.loc[:,['pct_chg']] # 'limit'
    df1['swing'] = 0

    for j in range(300): #loop through all stocks
        code = codelist[j]
        # 获取和他们对应涨跌幅
        prev_pct = df.loc[j,'prev']
        
        pct = df1.loc[code, 'pct_chg']
        swing = df1.loc[code, 'swing']
        
        df.loc[j,'pct_chg'] = pct
        
        if (j == 1):
            print(code, ':', prev_pct, pct)
        if pct > 9.5:
            zt += 1/300
            zt_w += df.iloc[j,3]/100
            if prev_pct > 9.5: #连板涨停
                    zt_c += 1/300
                    zt_cw += df.iloc[j,3]/100
        elif pct < -9.5:
            dt += 1/300
            dt_w += df.iloc[j,3]/100
            if prev_pct < -9.5: # 连续跌停
                dt_c += 1/300
                dt_cw += df.iloc[j,3]/100
                
    # ends inner for loop
    # print(df)
    tradedate.iloc[i,1] = zt
    tradedate.iloc[i,2] = dt
    tradedate.iloc[i,3] = dtb
    tradedate.iloc[i,4] = tdb
    tradedate.iloc[i,5] = dtb - tdb
    tradedate.iloc[i,6] = zt - dt
    tradedate.iloc[i,7] = zt_c - dt_c
    tradedate.iloc[i,8] = zt_w - dt_w
    tradedate.iloc[i,9] = zt_cw - dt_cw
    tradedate.iloc[i,10] = dtb_w - tdb_w
    
    
    print(df)
#ends outer for loop

#add return of the 399300.SZ
returns = pro.index_daily(ts_code=index_name, start_date=start, end_date=end)
returns = returns.loc[:,['trade_date','pct_chg']]
tradedate = tradedate.merge(returns, how='left', left_on='trade_date', right_on='trade_date')
print(tradedate);
# tradedate.to_csv('tradedate21-22.csv') # 2021-2022
# tradedate.to_csv('tradedate23.csv') # 2023
tradedate.to_csv('tradedate09-23.csv') # 2009-2023
            
# V1
#     提取涨跌幅 得出涨停比率与跌停比率 9.5%为分界
#     通过比率择时
# V2： 涨跌停比率剪刀差、连板比率剪刀差、地天板与天地板比率剪刀差
#     需要保存前一天的涨停与跌停的股票
#     与今天的取交集 得出连续涨停比率和连续跌停比率
#     涨跌停比率剪刀差 = 涨停比率 - 跌停比率
#     连板比率剪刀差 = 连续涨停比率 - 连续跌停比率
#     地天板比率 = 地天板走势的个股数量/成分股总数
            #前一天跌停 今天涨停
#     天地板比率 = 天地板走势的个股数量/成分股总数
        #前一天涨停 今天跌停
#     天地板比率剪刀差 = 地天板比率 - 天地板比率
# V3
#     自由流通市值加权涨跌停比率剪刀差
#     自由流通市值加权连板比率剪刀差 
#     自由流通市值加权地天与天地板比率剪刀差