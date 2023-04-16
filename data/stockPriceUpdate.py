import numpy as np
import pandas as pd

# requests可能要去載包
import requests
import pandas as pd
import datetime
import time

from bs4 import BeautifulSoup

#---------------------------------------
#-------------爬蟲的code----------------
#---------------------------------------

def get_stock_history(date, stock_no):
    
    # date：輸入當個月的第一天 如'20100101'(2010年1月1日) 這樣會拿到1月的資料 
    # stock_no：股票的ID
    
    # url：這是爬蟲的網址
    url = 'http://www.twse.com.tw/exchangeReport/STOCK_DAY?date=%s&stockNo=%s' % (date, stock_no)
    
    # 這是要抓東西的CODE
    r = requests.get(url)
    r.content.decode("utf-8")
    data = r.json()
    
    stat_number = 0
    
    while data['stat'] != 'OK':
        
        if stat_number == 3:
            out_list = []
            return out_list
        
        time.sleep(10)
        
        print('The stat have problem and reset again')
        url = 'http://www.twse.com.tw/exchangeReport/STOCK_DAY?date=%s&stockNo=%s' % (date, stock_no)
        r = requests.get(url)
        r.content.decode("utf-8")
        data = r.json()
        
        stat_number += 1
    
    #進行資料格式轉換  
    return transform(data['data'])  

#---------------------------------------
#------------轉換日期的code--------------
#---------------------------------------

def transform_date(date):
    y, m, d = date.split('/')
    return str(int(y)+1911) + '/' + m  + '/' + d  #民國轉西元
    
#---------------------------------------
#------------轉換資料的code--------------
#---------------------------------------

def transform_data(data):
    
    # print(data)
    
    if data[3] == "--":
        data[3] = "No data"
    else:
        data[3] = float(data[3].replace(',', ''))
        
    if data[4] == "--":
        data[4] = "No data"
    else:
        data[4] = float(data[4].replace(',', ''))
        
    if data[5] == "--":
        data[5] = "No data"
    else:
        data[5] = float(data[5].replace(',', ''))
        
    if data[6] == "--":
        data[6] = "No data"
    else:
        data[6] = float(data[6].replace(',', ''))
        
    data[0] = datetime.datetime.strptime(transform_date(data[0]), '%Y/%m/%d')
    data[1] = int(data[1].replace(',', ''))  #把千進位的逗點去除
    data[2] = int(data[2].replace(',', ''))
    
    # data[4] = float(data[4].replace(',', ''))
    # data[5] = float(data[5].replace(',', ''))
    # data[6] = float(data[6].replace(',', ''))
    data[7] = float(0.0 if data[7].replace(',', '') == 'X0.00' else data[7].replace(',', ''))  # +/-/X表示漲/跌/不比價
    data[8] = int(data[8].replace(',', ''))
    return data

def transform(data):
    
    out_list = []
    
    for d in data:
        
        temp_list = transform_data(d)
        
        try:
            temp_list.index("No data")
            continue
        except:
            out_list.append(temp_list)
        
    return out_list

#---------------------------------------
#----------創造成一個dataframe-----------
#---------------------------------------

def create_df(date, stock_no):
    
    tp_data = get_stock_history(date, stock_no)
    
    if len(tp_data) == 0:
        tp = pd.DataFrame()
        return tp
    
    df_col = ['date', 'shares', 'amount', 'open', 'high', 'low', 'close', 'change', 'turnover']
    s = pd.DataFrame(tp_data, columns = df_col)
    

    # s.columns = ['date', 'shares', 'amount', 'open', 'high', 'low', 'close', 'change', 'turnover']
    #             "日期","成交股數","成交金額","開盤價","最高價","最低價","收盤價","漲跌價差","成交筆數" 
    
    
    stock = []
    for i in range(len(s)):
        stock.append(stock_no)
    s['stockno'] = pd.Series(stock ,index=s.index, dtype='float64')  #新增股票代碼欄，之後所有股票進入資料表才能知道是哪一張股票
    datelist = []
    for i in range(len(s)):
        datelist.append(s['date'][i])
    s.index = datelist  #索引值改成日期
    s2 = s.drop(['date'],axis = 1)  #刪除日期欄位
    mlist = []
    for item in s2.index:
        mlist.append(item.month)
    s2['month'] = mlist  #新增月份欄位
    return s2

#---------------------------------------
#---------創造要抓年限的日期格式----------
#---------------------------------------

def create_date(begin_date, end_date): 
    
    # 輸入的起始形式為 '20180101'
    # 輸入的結束形式為 '20200701'
    # 這樣就是抓2018年1月至2020年7月的資料
    
    list_date = [begin_date] 
    
    begin_year = int(begin_date[0:4])
    end_year = int(end_date[0:4])
    
    begin_month = int(begin_date[4:6])
    end_month = int(end_date[4:6])
    
    print('Begin Year %i, End Year' % begin_year, end_year)
    print('Begin Month %i, End Month' % begin_month, end_month)
    
    while True:
        
        if begin_year == end_year and begin_month == end_month:
            break
        
        if begin_month >= 10:
            temp_str = str(begin_year) + str(begin_month) + str(0) +str(1)
        else:
            temp_str = str(begin_year) + str(0) + str(begin_month) + str(0) +str(1)
        

        
        if begin_month == 12:
            begin_month = 0
            begin_year = begin_year + 1
        # print(temp_str)    
        list_date.append(temp_str)
        begin_month = begin_month + 1
            
    return list_date

def resampleData(df):
    TW_50_TICKER= df.stockno.unique()
    data_dict = {}
    df_Ta_copy = df.copy()

    # 決定要不要用日期替代index
    reset_index2date = False

    for sk_id in TW_50_TICKER:
        
        # 判斷每支股票的位置在哪裡
        bool_tp = df_Ta_copy['stockno'] == int(sk_id)
        
        # 把每支股票取出來
        temp_df = df_Ta_copy.loc[bool_tp]
        
        # 把紀錄id那項丟掉
        temp_df_1 = temp_df.drop(labels=['stockno'], axis=1)
        
        if reset_index2date == True:
            # 有些資料的date不是index 要自己加 有的話就不用
            temp_df_1.set_index('date' , inplace=True)
        
        # 存到字典裡
        data_dict[sk_id] = temp_df_1.copy()
        
    for x in TW_50_TICKER:
        data_dict[x] = data_dict[x][~data_dict[x].index.duplicated(keep='first')]
        data_dict[x] = data_dict[x].resample('B').ffill()
        #data_dict[x] = data_dict[x].drop('Days',inplace=True) #I'm thinking should drop Days
        print(f'process{x}')
    # # 用字典(dict)儲存不同支股票的data，這樣比較好做

    first_number =0

    for key, item in data_dict.items():
        
        temp_df = data_dict[key]
        temp_df['stockno'] = key

        if first_number == 0:
            final_df = temp_df
            first_number = 1
        else:
            final_df = pd.concat([final_df, temp_df], axis=0)

    return final_df


if __name__ == '__main__':

    currentDataPath = './data/latest_50_45tic_sourceData.pkl'
    currentData = pd.read_pickle(currentDataPath)

    
    startDate = str(currentData.index[-1].date() +  datetime.timedelta(days=1))
    endDate = str(datetime.datetime.now().date()).replace('-','')
    tickers =  ['0050','1101','1102','1216','1301','1303','1326','1402','1590','2002','2207',\
    '2303','2308','2317','2327','2330','2357','2379','2382','2395','2408',\
    '2409','2412','2454','2603','2609','2615','2801','2880','2881','2882',\
    '2883','2884','2885','2886','2891','2892','2912','3008','3034','3045',\
    '3481','4904','4938','6505','9910']

    if datetime.datetime.now().date()-currentData.index[-1].date() >= datetime.timedelta(days=5):
    # 要抓的日期 20100101-20201201
        if startDate.replace('-','') != endDate:
            list_month = create_date(startDate.replace('-',''),endDate)

            # 輸出日期的樣子
            print('All of month number = ', len(list_month))
            print('Month list', list_month)


            # 開始爬蟲

            final_result = currentData.copy()

            for i in range(len(tickers)):
                    
                out_result = pd.DataFrame()
                
                # 每個月抓一次資料
                for j in range(len(set(list_month))):

                    print('Stock ID--->', tickers[i], end='|')
                    print('Month--->', list_month[j])
                    
                    temp_result = create_df(list_month[j], tickers[i])
                    
                    # print(temp_result)
                    
                    if temp_result.empty:
                        time.sleep(5)
                        print('Continue')
                        continue
                    
                    # print(temp_result)
                    out_result = pd.concat([out_result, temp_result], axis=0)
                    out_result = out_result.loc[startDate:]
                    # 一定要加這個 每隔幾秒在抓一次
                    time.sleep(5)
                    
                    print('Finish')
                    
                final_result = pd.concat([final_result,out_result],axis=0)
                
            
            print(f'Done: \n {final_result}')

            final_result = resampleData(final_result)

            # 存source data
            latest_data_path = './data/latest_50_45tic_sourceData.pkl'
            final_result.to_pickle(latest_data_path)

            #存 ETF 50 data
            etf50 = final_result[final_result.stockno == 50]
            latest_ETF50_data_path = './data/latest_50ETF_data.pkl'
            etf50.to_pickle(latest_ETF50_data_path)

            #存 45 tickers data
            stockTickerData = final_result[final_result.stockno != 50]
            latest_stockTickerData_path = './data/latest_45tic_data.pkl'
            stockTickerData.to_pickle(latest_stockTickerData_path)

            # 這邊是修改每欄的名字 存price data
            price_data =final_result[['stockno','close','change']]
            price_data.rename(columns={'stockno':'Stock_ID'}, inplace=True)
            latest_price_data_path = './data/latest_45tic_priceBook.pkl'
            price_data[price_data.Stock_ID != 50].to_pickle(latest_price_data_path)
            

            # 存 Jiang npy data

            stockTickerData = stockTickerData.loc['2012-01-02':]
            tickers = tickers[1:] #remove 0050   
            list_open = np.array([])  
            open_of_the_day = np.array([])  
            list_close = np.array([])  
            list_high = np.array([])  
            list_low = np.array([])  

            for s in tickers:
                data = stockTickerData[stockTickerData.stockno == float(s)]
                data = data[~data.index.duplicated(keep='first')]
                data = data.resample('B').ffill()
                data = data[['open', 'close', 'high', 'low']]
                list_open=np.append(list_open,data.open.values[:-1], axis=0)
                open_of_the_day=np.append(open_of_the_day,data.open.values[1:], axis=0)
                list_close=np.append(list_close,data.close.values[:-1], axis=0)
                list_high=np.append(list_high,data.high.values[:-1], axis=0)
                list_low=np.append(list_low,data.low.values[:-1], axis=0)
                print(data.shape,s)
            total_days = data.shape[0] -1
            array_open=list_open.reshape(45,total_days)
            array_open_of_the_day=open_of_the_day.reshape(45,total_days)
            array_close=list_close.reshape(45,total_days)
            array_high=list_high.reshape(45,total_days)
            array_low=list_low.reshape(45,total_days)

            X = np.transpose(np.array([array_close/array_open, 
                                array_high/array_open,
                                array_low/array_open,
                                array_open_of_the_day/array_open]), axes= (0,1,2))

            print(X.shape)
            jiang_npy_path = './data/jiang_data.npy'
            np.save(jiang_npy_path, X)
        else:
            print('No need to scrawl new price, current price is latest')



