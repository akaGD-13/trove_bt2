#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:19:51 2023
天地板/地天板数据

@author: guangdafei
"""

import tushare as ts
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import requests

def tdb_or_dtb(code, date, Open):
    # import requests
    
    formatted_date = date[:4]+'-'+date[4:6]+'-'+date[6:]
    
    # 通过超级命令客户端 “工具-refresh_token查询" 得到 refresh_token
    refreshToken = "eyJzaWduX3RpbWUiOiIyMDIzLTEyLTAxIDEwOjU3OjAzIn0=.eyJ1aWQiOiI2OTk2OTQzNTYiLCJ1c2VyIjp7ImFjY291bnQiOiJ3bHpoMDcyIiwiYXV0aFVzZXJJbmZvIjp7ImFwaUZvcm1hbCI6IjEifSwiY29kZUNTSSI6W10sImNvZGVaekF1dGgiOltdLCJoYXNBSVByZWRpY3QiOmZhbHNlLCJoYXNBSVRhbGsiOmZhbHNlLCJoYXNDSUNDIjpmYWxzZSwiaGFzQ1NJIjpmYWxzZSwiaGFzRXZlbnREcml2ZSI6ZmFsc2UsImhhc0ZUU0UiOmZhbHNlLCJoYXNGdW5kVmFsdWF0aW9uIjpmYWxzZSwiaGFzSEsiOnRydWUsImhhc0xNRSI6ZmFsc2UsImhhc0xldmVsMiI6ZmFsc2UsImhhc1VTIjpmYWxzZSwiaGFzVVNBSW5kZXgiOmZhbHNlLCJtYXJrZXRDb2RlIjoiMTY7MzI7MTQ0Ozk2OzE3NjsxMTI7ODg7NDg7MTI4OzE2OC0xOzE4NDsyMDA7MjE2OzEwNDsxMjA7MTM2OzIzMjs1Njs2NDsiLCJtYXhPbkxpbmUiOjEsInByb2R1Y3RUeXBlIjoiU1VQRVJDT01NQU5EUFJPRFVDVCIsInJlZnJlc2hUb2tlbkV4cGlyZWRUaW1lIjoiMjAyMy0xMi0zMCAxMzoyMzo1OCIsInNlc3NzaW9uIjoiNzAxZjg1NDAyY2M1MDJiMTBmNWFkOGZlNjBjNjYyMDQiLCJzaWRJbmZvIjp7fSwidWlkIjoiNjk5Njk0MzU2IiwidXNlclR5cGUiOiJPRkZJQ0lBTCIsIndpZmluZExpbWl0TWFwIjp7fX19.D06E7950E9543F6765FA7C7464F76F35F826AD8D2225B369A72203469A6A665F"
    
    para = {"Content-Type": "application/json", "refresh_token": refreshToken}
    # 利用 refresh_token 得到 access_token
    accessToken = requests.get(
        "https://ft.10jqka.com.cn/api/v1/get_access_token", params=para).json()["data"]["access_token"]
    
    # 利用 access_token 得到 分钟级数据
    
    url = "https://ft.10jqka.com.cn/api/v1/high_frequency"
    
    header = {"Content-Type": "application/json", "access_token": accessToken}
    
    # 此para的构造参考《HTTP20230404用户手册》 P12 “高频序列”
    para = {
        "codes": code,
        "indicators": "open,high,low,close,avgPrice,volume,amount,change",
        "starttime": formatted_date + " 09:00:00",
        "endtime": formatted_date + " 17:00:00",
    }
    
    # 返回包含数据的字典
    dataDict = requests.get(url, params=para, headers=header).json()
    if 'tables' not in dataDict:
        print("tdb_or_dtb function: No tables, return -1")
        return -1;
    # print(dataDict)
    open_list = dataDict['tables'][0]['table']['open']
    # print(open_list)
    for i in range(len(open_list)):
        if open_list[i]/Open >= 1.095:
            return 1;
        elif open_list[i]/Open <= 0.905:
            return 0;
    return -1;

    
tdb_count = 0 # for debugging
no_data_count = 0

plt.rcParams['font.sans-serif'] = ['SimHei']

token = "13bb0841c8a377221b39d9142f42bae2e2e9a897b9f692c75dd90d65"
ts.set_token(token)
pro = ts.pro_api()

index_name= '399300.SZ'

# start_weight = '20091009'
# start = '20091009'
# end = '20230928'
# remember to also change the file path when saving at the end of this program

start_weight = '20220801'
# start = '20220801' # approximately 100 trade date before 20230101 (to calculate HMA100)
# end = '20230928'

start = '20220804' #  100 trade date before 20230103 （first tradedate of 2023) (to calculate HMA100)
end = '20231228'

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

df = pro.index_weight(index_code=index_name, start_date=start_weight, end_date=start_weight)
# df = df.set_index('con_code')
df['pct_chg'] = 0
df['prev'] = 0
df.to_csv('index_weight.csv')
# print(df)

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
    if i%3 == 0:
        df_temp =pro.index_weight(index_code=index_name, start_date=date, end_date=date)
            #     index_code   con_code trade_date  weight
            # 0    399300.SZ  600519.SH   20230901  6.2310
            # 1    399300.SZ  300750.SZ   20230901  3.3419
            # 2    399300.SZ  601318.SH   20230901  2.9086
            # ..         ...        ...        ...     ...
            # 299  399300.SZ  001289.SZ   20230901  0.0162
        
        if df_temp.empty or df_temp.index[-1] != 299: # size is not 300: empty or missing stocks #use the original df, update prev
            # print("empty, 继承之前的index weight")
            df = pd.read_csv('index_weight.csv')
            df = df.iloc[:,1:]
        else: # not empty, update df
            # print('not empty，更新index weight')
            df = df_temp.merge(df.loc[:,['con_code','prev','pct_chg']], how='left', left_on = 'con_code', right_on='con_code')
            df.to_csv('index_weight.csv')
            # df = df.set_index('con_code')
        
    df.fillna(0, inplace=True) # if some stock remove/added set their pct_chg and prev as 0
    df.loc[:,'prev'] = df.loc[:,'pct_chg']  #update prev  
    codelist = df.loc[:,'con_code']
    allcode = ''
    
    for x in range(300): #提取所有股票代码
        allcode = codelist[x] + ',' + allcode # 这里的顺序不能反
    #ends this for loop
        
    # 提取这些stock的pct_change, swing, limit(涨跌停类型)
    # df1 = pro.bak_daily(ts_code=allcode, trade_date=date, fields='ts_code,pct_change,swing') # 涨幅，振幅
    # df2 = pro.limit_list_d(ts_code=allcode, trade_date=date, fields='ts_code,limit')  #涨停/跌停
    df1 = pro.daily(ts_code=allcode, start_date=date, end_date=date) # 涨幅
    df1 = df1.loc[:,['ts_code','pct_chg','high','low', 'open']]
    
    df1 = pd.DataFrame(codelist).merge(df1, how='left', left_on='con_code', right_on='ts_code') 
    
    # df1 = df1.merge(df2, how='left', left_on='con_code', right_on='ts_code')
    df1.fillna(0,inplace=True)
    df1= df1.set_index('con_code')
    df1 = df1.loc[:,['pct_chg','high','low', 'open']] # 'limit'

    for j in range(300): #loop through all stocks
        code = codelist[j]
        # 获取和他们对应涨跌幅
        prev_pct = df.loc[j,'prev']
        
        
        pct = df1.loc[code, 'pct_chg']
        
        df.loc[j,'pct_chg'] = pct
        
        # if (j == 1):
        #     print(code, ':', prev_pct, pct) # testing
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
        
        #天地板/地天板
        high = float(df1.loc[code, 'high'])
        low = float(df1.loc[code, 'low'])
        Open = float(df1.loc[code, 'open'])
        if Open == 0: # empty dataframe
            # print(code, date,"Open is 0", Open)
            iii = 1
        # print("high low and open are: "+ str(high) + " " + str(low) + " " + str(Open))
        elif float(high/Open) >= 1.095 and float(low/Open) <= 0.905:
            result_tdb_dtb = tdb_or_dtb(code, date, Open) # insert function here
            tdb_count += 1
            print(date, code, "tdb / dtb 出现：" + str(result_tdb_dtb))
            
            if result_tdb_dtb == 0: #tdb
                tdb += 1/300
                tdb_w += df.iloc[j,3]/100
            elif result_tdb_dtb == 1: #dtb
                dtb += 1/300
                dtb_w += df.iloc[j,3]/100
            else:
                no_data_count += 1
        
                
    # ends inner for loop (stocks)
    
    df.to_csv('index_weight.csv')
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
    
    
    # print(df)
#ends outer for loop (date)

#add return of the 399300.SZ
returns = pro.index_daily(ts_code=index_name, start_date=start, end_date=end)
returns = returns.loc[:,['trade_date','pct_chg']]
tradedate = tradedate.merge(returns, how='left', left_on='trade_date', right_on='trade_date')
print(tradedate);
# tradedate.to_csv('tradedate21-22.csv') # 2021-2022
# tradedate.to_csv('tradedate23.csv') # 2023
tradedate.to_csv('推波助澜.csv') # 2009-2023

print("tdb or dtb count: " + str(tdb_count))
print("no date: " + str(no_data_count))
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
