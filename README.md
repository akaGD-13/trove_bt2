# trove_bt2


[推波助澜中期择时](https://mp.weixin.qq.com/s?__biz=MzI1Mjg3MzYyOA==&mid=2247502995&idx=1&sn=d899e3efca1cc1b4a41307e90110f306&chksm=e9df9260dea81b769ed30cac53b49e1b8a14bdaabea7a008d4890846aaf1aabca955751d4374&mpshare=1&scene=1&srcid=110685Q1vJViDZWbq95WfJMs&sharer_shareinfo=172efbe8769055c3805337fb7d2e895f&sharer_shareinfo_first=172efbe8769055c3805337fb7d2e895f&version=4.1.9.90740&platform=mac#rd)

天地板数据暂时无法提取

V1

    提取涨跌幅 得出涨停比率与跌停比率 9.5%为分界
    通过比率择时
V2

    涨跌停比率剪刀差、连板比率剪刀差、地天板与天地板比率剪刀差
    需要保存前一天的涨停与跌停的股票
    与今天的取交集 得出连续涨停比率和连续跌停比率
    涨跌停比率剪刀差 = 涨停比率 - 跌停比率
    连板比率剪刀差 = 连续涨停比率 - 连续跌停比率
     
V3

     自由流通市值加权涨跌停比率剪刀差
     自由流通市值加权连板比率剪刀差 
     
沪深300

# notes.py:

  gathering data of 沪深300, and calculating indexs:
  
      tradedate['涨停比率'] = '' #1
      tradedate['跌停比率'] = '' #2
      tradedate['地天板比率'] = '' #3
      tradedate['天地板比率'] = '' #4
      tradedate['天地板比率剪刀差'] = '' #5
      tradedate['涨跌停比率剪刀差'] = '' #6
      tradedate['连板比率剪刀差'] = '' #7
      tradedate['自由流通市值加权涨跌停比率剪刀差'] = '' #8
      tradedate['自由流通市值加权连板比率剪刀差'] = '' #9
      tradedate['自由流通市值加权地天与天地板比率剪刀差'] = '' #10

# bt_1.py
  当指标大于x, 做多。小于y，做空
  
        运行时间：取决于 maxi, step 的值
        standard= '涨跌停比率剪刀差'
        standard = '连板比率剪刀差'
        standard = '自由流通市值加权涨跌停比率剪刀差'
        standard = '自由流通市值加权连板比率剪刀差'
    四个图：以 1.png 结尾

# bt_2.py:
报告原文
  
    https://pdf.dfcfw.com/pdf/H3_AP202011181430494641_1.pdf?1605686153000.pdf  
只在HMA30/HMA100>1.15且HMA30和HMA100都大于0的情形下持有多仓，剩下空仓

HMA计算方法：
  
    https://school.stockcharts.com/doku.php?id=technical_indicators:hull_moving_average
四个图：以 2.png 结尾
