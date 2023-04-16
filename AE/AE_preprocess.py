import pandas as pd
import numpy as np
import talib as ta

TW_50_TICKER  = ['2330',\
 '2317',
 '2454',
 '6505',
 '2412',
 '2308',
 '1301',
 '1303',
 '2882',
 '2303',
 '2881',
 '1326',
 '3711',
 '2886',
 '2891',
 '3008',
 '1216',
 '2002',
 '2382',
 '3034',
 '3045',
 '2884',
 '2207',
 '2892',
 '2408',
 '2912',
 '5880',
 '2357',
 '2327',
 '2395',
 '2885',
 '5871',
 '1101',
 '2379',
 '2880',
 '2603',
 '8046',
 '6415',
 '4904',
 '3481',
 '2409',
 '4938',
 '1590',
 '5876',
 '2801',
 '9910',
 '1402',
 '2474',
 '1102',
 '2883']

def kd_indicator(df, fk_period=9, sk_period=3, sk_matype=1, sd_period=3, sd_matype=1):
    # (1) 9, 3, 3 (2) 9, 6, 6 (3) 5, 3, 3
    k, d = ta.STOCH(df['high'], df['low'], df['close'],
                     fastk_period=fk_period, slowk_period=sk_period, slowk_matype=sk_matype, slowd_period=sd_period,  slowd_matype=sd_matype)
    
    return [k,d, k.sub(d, fill_value=0)]
# print(kd_indicator(df)[2])

def kdf_indicator(df, fk_period=5, fd_period=3, fd_matype=0):
    fk, fd = ta.STOCHF(df['high'], df['low'], df['close'], fastk_period=fk_period, fastd_period=fd_period, fastd_matype=fd_matype)
    return [fk, fd]
# print(kdf_indicator(df)[0])
# print(kdf_indicator(df)[1])

def kdrsi_indicator(df, tp=14, fk_period=5, fd_period=3, fd_matype=0):
    fk, fd = ta.STOCHRSI(df['close'], timeperiod=tp, fastk_period=fk_period, fastd_period=fd_period, fastd_matype=fd_matype)
    return [fk, fd]
# print(kdrsi_indicator(df)[0])
# print(kdrsi_indicator(df)[1])

def ma_indicator(df, tp=5, mt=0):
    # 3, 5, 10, 20, 60, 120 均線設定
    # matype：0 = SMA，1 = EMA，2 = WMA，3 = DEMA，4 = TEMA，5 = TRIMA，6 = KAMA，7 = MAMA，8 = T3（默認值= SMA）
    return ta.MA(df['close'], timeperiod=tp, matype=mt)
# print(ma_indicator(df, 5, 0))
# print(ma_indicator(df, 5, 1))

def bias_indicator(df, n=5):
    # 乖離率參數3, 5, 10, 20, 60, 120, 240。
    close = df['close']
    return close / close.rolling(n, min_periods=1).mean()
# print(bias_indicator(df))

def macd_indicator(df, fp=12, sp=26, sgp=9):
    # 移動平均線參數分別為5、10、30。
    # https://zh.wikipedia.org/wiki/MACD
    macd, macdsignal, macdhist = talib.MACD(df['close'].values, fastperiod=fp, slowperiod=sp, signalperiod=sgp)
    return [macd, macdsignal, macdhist]
# macd, macdsignal, macdhist = macd_indicator(df, 12, 26, 9)
# print(macd_indicator(df, 12, 26, 9)[0])
# print(macd_indicator(df, 12, 26, 9)[1])
# print(macd_indicator(df, 12, 26, 9)[2])

def rsi_indicator(df, tp=12):
    #RSI的天数一般是6、12、24
    return ta.RSI(df['close'].values, timeperiod=tp)
# type(rsi_indicator(df, 12))

def div_indicator(df):
    return ta.DIV(df['high'], df['low'])
# print(div_indicator(df))

def beta_indicator(df, tp=5):
    return ta.BETA(df['high'], df['low'], timeperiod=tp)
# print(beta_indicator(df))

def sin_indicator(df):
    return ta.SIN(df['close'])
# print(sin_indicator(df))

def cos_indicator(df):
    return ta.COS(df['close'])
# print(cos_indicator(df))

def ultosc_indicator(df, tp1=7, tp2=14, tp3=28):    
    return ta.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=tp1, timeperiod2=tp2, timeperiod3=tp3)
# print(ultosc_indicator(df))

def ht_dcperiod_indicator(df):
    return ta.HT_DCPERIOD(df['close'])
# print(ht_dcperiod_indicator(df))

def adosc_indicator(df, fp=3, sp=10):    
    return ta.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=fp, slowperiod=sp)
# print(adosc_indicator(df))

def mfi_indicator(df, tp=14):    
    return ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=tp)
# print(mfi_indicator(df))

def ht_phasor_indicator(df):
    inphase, quadrature = ta.HT_PHASOR(df['close'])
    return [inphase, quadrature]
# print(ht_phasor_indicator(df)[1])

def cci_indicator(df, tp=14):    
    return ta.CCI(df['high'], df['low'], df['close'], timeperiod=tp)
# print(cci_indicator(df))

def plus_di_indicator(df, tp=14):    
    return ta.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=tp)
# print(plus_di_indicator(df))

def adxr_indicator(df, tp=14):    
    return ta.ADXR(df['high'], df['low'], df['close'], timeperiod=tp)
# print(adxr_indicator(df))

def aroon_indicator(df, tp=14):
    aroondown, aroonup = ta.AROON(df['high'], df['low'], timeperiod=tp)
    return [aroondown, aroonup]
# print(aroon_indicator(df)[0])

def cci(df, n):
    result = ta.CCI(df['high'], df['low'], df['close'], n)   
    return result 

def rocp(df, n):
    result = ta.ROCP(df['close'], timeperiod=n)
    return result

def scale_value(df):
    
    cl_tp = df['close'].to_numpy()
    op_tp = df['open'].to_numpy()
    hi_tp = df['high'].to_numpy()
    lo_tp = df['low'].to_numpy()
    vo_tp = df['volume'].to_numpy()

    cl_list = []
    op_list = []
    hi_list = []
    lo_list = []
    vo_list = []

    for index, cl_ele in enumerate(cl_tp):

        op_value_tp = op_tp[index] / cl_ele - 1
        hi_value_tp = hi_tp[index] / cl_ele - 1
        lo_value_tp = lo_tp[index] / cl_ele - 1

        op_list.append(op_value_tp)
        hi_list.append(hi_value_tp)
        lo_list.append(lo_value_tp)

        if index == 0:
            cl_list.append(float('Nan'))
        else:
            cl_value_tp = cl_ele - cl_tp[index-1]
            cl_list.append(cl_value_tp)


        if index < 10:
            vo_list.append(float('Nan'))
        else:
            vo_tp_10 = vo_tp[index-10:index-1].mean()
            vp_value_tp = vo_tp[index] / vo_tp_10

            vo_list.append(vp_value_tp)
            
    return np.array(cl_list), np.array(op_list), np.array(hi_list), np.array(lo_list), np.array(vo_list)

# 讀取檔案的位址(這邊是用絕對位置，一定要改，不然讀不到)
new_data_pkl = '/Users/yi-hsuanlee/Desktop/WIDM/Thesis/finRL-elegant/dashboard/data/latest_45tic_data.pkl'
df_Ta = pd.read_pickle(new_data_pkl)

# 這邊是修改每欄的名字
df_Ta.rename(columns={'stockno':'Stock_ID', 'amount':'volume'}, inplace=True)

TW_50_TICKER= df_Ta.Stock_ID.unique()
data_dict = {}
df_Ta_copy = df_Ta.copy()

# 決定要不要用日期替代index
reset_index2date = False

for sk_id in TW_50_TICKER:
    
    # 判斷每支股票的位置在哪裡
    bool_tp = df_Ta_copy['Stock_ID'] == int(sk_id)
    
    # 把每支股票取出來
    temp_df = df_Ta_copy.loc[bool_tp]
    
    # 把紀錄id那項丟掉
    temp_df_1 = temp_df.drop(labels=['Stock_ID'], axis=1)
    
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
    temp_df['Stock_ID'] = key

    if first_number == 0:
        final_df = temp_df
        first_number = 1
    else:
        final_df = pd.concat([final_df, temp_df], axis=0)


# 製作feature
out_dict = {}

for i, sk_id in enumerate(TW_50_TICKER):  
    
    print('Stock ID--->', sk_id, end='|')
    
    df_Top20_cal = data_dict[sk_id]
    
    if df_Top20_cal.empty:
        print('Skip')
        continue
    
    # 把OHCLV做縮放
    close, open1, high, low, volume = scale_value(df_Top20_cal)
    
    # 生成技術指標的數值
    dis_kd533 = kd_indicator(df_Top20_cal)[2]
    kdf_k = kdf_indicator(df_Top20_cal)[0]
    kdf_d = kdf_indicator(df_Top20_cal)[1]
    kdrsi_k = kdrsi_indicator(df_Top20_cal)[0]
    kdrsi_d = kdrsi_indicator(df_Top20_cal)[1]
    dis_ma3 = df_Top20_cal['close'].sub(ma_indicator(df_Top20_cal, 3, 0), fill_value=0)
    bias3 = bias_indicator(df_Top20_cal)
    div = div_indicator(df_Top20_cal)
    beta = beta_indicator(df_Top20_cal)
    sin = sin_indicator(df_Top20_cal)
    cos = cos_indicator(df_Top20_cal)
    ultosc = ultosc_indicator(df_Top20_cal)
    ht_dcperiod = ht_dcperiod_indicator(df_Top20_cal)
    adosc = adosc_indicator(df_Top20_cal)
    mfi = mfi_indicator(df_Top20_cal)
    ht_phasor = ht_phasor_indicator(df_Top20_cal)[1]
    cci = cci_indicator(df_Top20_cal)
    plus_di = plus_di_indicator(df_Top20_cal)
    adxr = adxr_indicator(df_Top20_cal)
    aroon = aroon_indicator(df_Top20_cal)[0]
    
    # 最後組成一個dataframe
    feature = pd.DataFrame({'close':close, 'open':open1, 'high':high, 'low':low, 'volume':volume
                           ,'dis_kd533':dis_kd533, 'kdf_k':kdf_k, 'kdf_d':kdf_d, 'kdrsi_k':kdrsi_k, 'kdrsi_d':kdrsi_d, 
                           'dis_ma3':dis_ma3, 'bias3':bias3, 'div':div, 'beta':beta, 'sin':sin,
                           'cos':cos, 'ultosc':ultosc, 'ma_ht_dcperiod5':ht_dcperiod, 'adosc':adosc,
                           'mfi':mfi, 'ht_phasor':ht_phasor, 'cci':cci, 'plus_di':plus_di, 'adxr':adxr, 'aroon':aroon})
    
    out_dict[sk_id] = feature
    print('Finish')

# 把不同股票的dataframe組成一個大個dataframe，才能一次輸出成excel

for sk_id, temp_df in out_dict.items():
    
    temp_df['Stock_ID'] = sk_id
    
    if sk_id == list(out_dict.keys())[0]:
        final_df = temp_df
    else:
        final_df = pd.concat([final_df, temp_df], axis=0)


# data path
preProcessPath = f'./data/latest-45tic-data-preprocess.pkl'
final_df.to_pickle(preProcessPath)
print(f'preprocess done!')
