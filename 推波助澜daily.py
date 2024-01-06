#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:24:57 2023

前一天的数据：推波助澜index_weight.csv (用来继承index_weight和计算连板) 需要update
从2023年1月3日到现在的数据：推波助澜.csv 需要添加一行

@author: guangdafei
"""

import math
import time #计算运行时间
import pandas as pd
import numpy as np
import requests #同花顺
from datetime import datetime # 获取当前日期
import tushare as ts
token = "13bb0841c8a377221b39d9142f42bae2e2e9a897b9f692c75dd90d65"
ts.set_token(token)
pro = ts.pro_api()
import matplotlib.pyplot as plt

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


start_time = time.time();

date = datetime.now().date().strftime("%Y%m%d") # 获取当前日期
# date = '20240104'
index_weight = pd.read_csv('index_weight.csv')
index_weight = index_weight.iloc[:,1:]

df23 = pd.read_csv('推波助澜.csv')
df23 = df23.iloc[:,1:]

while int(df23.iloc[-1,0]) >= int(date): # 防止同一天多次运行 如果csv里最近的日期比当前日期靠后，删除最后一行
    df23 = df23.iloc[:-1,:]

df = index_weight


index_name= '399300.SZ'
# 提取数据，更新index_weight.csv 和 推波助澜.csv
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

# 获取index weight
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
    # df = df.set_index('con_code')
    
df.fillna(0, inplace=True) # if some stock remove/added set their pct_chg and prev as 0
df.loc[:,'prev'] = df.loc[:,'pct_chg']  #update prev  
codelist = df.loc[:,'con_code']
allcode = ''

for x in range(300): #提取所有股票代码
    allcode = codelist[x] + ',' + allcode # 这里的顺序不能反
#ends this for loop
    
# 提取这些stock的pct_change, swing, limit(涨跌停类型)
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
        print(date, code, "tdb / dtb 出现：" + str(result_tdb_dtb))
        
        if result_tdb_dtb == 0: #tdb
            tdb += 1/300
            tdb_w += df.iloc[j,3]/100
        elif result_tdb_dtb == 1: #dtb
            dtb += 1/300
            dtb_w += df.iloc[j,3]/100
    
            
# ends inner for loop (stocks)

df.to_csv('index_weight.csv')
# print(index_weight)
# print(df)

returns = pro.index_daily(ts_code=index_name, start_date=date, end_date=date)
change = returns.loc[0,'pct_chg']

df23.loc[len(df23.index)] = [date, zt, dt, dtb, tdb, dtb-tdb, zt-dt, zt_c-dt_c, zt_w-dt_w, zt_cw-dt_cw, dtb_w-tdb_w, change, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


# 计算HMA相关，更新 推波助澜.csv

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
# print(df23)

plt.plot(df23.index[100:], df23.loc[100:,'return'], label='model')
plt.plot(df23.index[100:], df23.loc[100:,'hs300'], label='HS300')
plt.title('20230103 -- ' + date)
plt.legend()
plt.savefig('推波助澜daily.png')
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


title = '推波助澜中期择时模型表现: (沪深300 2023-01-03 -- 今)' + '\n'
title += 'yearly_return = ' + str(yearly_return) + '\n'
# title += 'yearly_volatility = ' + str(yearly_volatility) + '\n'
title += 'max_drawdown: ' + str(max_drawdown) + '\n' 
title += '胜率: ' + str(v_ratio) + '\n' 
title += '盈亏比: ' + str(profit_loss_ratio) + '\n'
title += 'sharp ratio: ' + str(sharpe_ratio) + '\n'
title += 'yearly trade count: ' + str(yearly_trade_count)

print(df23)

print()
print('date is: ' + date)
if X2.iloc[-1,0]:
    print("今日持仓")
else:
    print('今日空仓')
print()
print(title)

end_time = time.time()
elapsed_time = end_time - start_time
print()
print(f"运行时间: {elapsed_time} seconds")

