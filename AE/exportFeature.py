import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import talib as ta
import math
import time
import argparse


from numpy import array

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Reshape 

from talib import abstract

from pandas import Series
from pandas.tseries.offsets import Day

from datetime import datetime
from datetime import timedelta

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import visualkeras



# LSTM input shape explanation [n_samples x timesteps x n_features]

#-------------------------
#轉換成能輸入至DNN的資料型態
#-------------------------

def transform_to_dnn_input(data, train_test_rate=0.3):
    
    # data input 要是numpy
    
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    
    train_bound = int((data.shape[0]+1) * (1-train_test_rate))
    
    # print('Train size = ', train_bound, end=' | ')
    # print('Test size = ', test_bound)
    # print('Summary = ', (train_bound+test_bound), end=' | ')
    # print('Original = ', data.shape[0] + 1, end='\n\n')
    '''
    # check 大小是否一樣
    if (train_bound + test_bound) != (data.shape[0] + 1):
        print('Not equal')
        return False, False
    '''
    x_train = data[0:train_bound]
    x_train_out = np.array(x_train)
    
    x_test = data[train_bound:]
    x_test_out = np.array(x_test)
    
    return x_train_out, x_test_out

#-------------------------
#轉換成能輸入至LSTM的資料型態
#-------------------------

def transform_to_lstm_input(timesteps, data, train_test_rate=0.3):
    # lstm
    # timesteps：一次多久時長輸入(在本論文裡是5、15、30)
    # data：numpy的資料
    
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    
    x_train = []
    x_test = []
    
    train_bound = int((data.shape[0]+1) * (1-train_test_rate))
    test_bound = data.shape[0] - train_bound + 1
    
    # print('Train size = ', train_bound, end=' | ')
    # print('Test size = ', test_bound)
    # print('Summary = ', (train_bound+test_bound), end=' | ')
    # print('Original = ', data.shape[0] + 1, end='\n\n')
    
    if (train_bound + test_bound) != (data.shape[0] + 1):
        print('Not equal')
        return False, False
    
    for reshape_i in range(timesteps, train_bound):
        x_train.append(data[reshape_i-timesteps:reshape_i])
    
    x_train_out = np.array(x_train)
    
    for reshape_i in range(train_bound, data.shape[0] + 1):
        x_test.append(data[reshape_i-timesteps:reshape_i])
    
    # print(x_test)
    
    x_test_out = np.array(x_test)
    
    return x_train_out, x_test_out

#-------------------------
#這是組合式模型預測部分的資料
#-------------------------

def transform_to_lstm_input_pred(timesteps, pred_steps, data, train_test_rate=0.3, dimen = 0):
    
    # composite
    # timesteps：一次多久時長輸入(在本論文裡是5、15、30)
    # pred_steps：要預測多長的資料(在本論文是5)
    # data：numpy的資料
    
    # data input 要是numpy
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    
    x_train = []
    x_test = []
    x_train_lowdim = []
    x_test_lowdim = []
    
    train_bound = int((data.shape[0]+1) * (1-train_test_rate))
    test_bound = data.shape[0] - train_bound + 1
    
    # print('Train size = ', train_bound, end=' | ')
    # print('Test size = ', test_bound)
    # print('Summary = ', (train_bound+test_bound), end=' | ')
    # print('Original = ', data.shape[0] + 1, end='\n\n')
    
    if (train_bound + test_bound) != (data.shape[0] + 1):
        print('Not equal')
        return False, False
    
    # 是否只取某幾個特徵出來預測
    if dimen == 0:
        data_low = data
    else:
        data_low = data[:, :dimen]
    
    # 組合成3維的訓練資料
    for reshape_i in range(timesteps, train_bound):
        
        # AE部分就是往回推
        x_train.append(data[reshape_i-timesteps:reshape_i])
        
        # 預測部分往前推
        try:
            x_train_lowdim.append(data_low[reshape_i:reshape_i+pred_steps])
        except:
            continue
        
    x_train_out = np.array(x_train)
    x_train_low = np.array(x_train_lowdim)
    
    # 訓練資料分成訓練和預測的部分
    gap_step = abs(pred_steps - timesteps)
    
    x_train_out = x_train_out.copy()[:-pred_steps]
    x_train_pred = x_train_low.copy()[:-pred_steps]
  
    # 組合成3維的測試資料
    
    # AE部分就是往回推
    for reshape_i in range(train_bound-1, data.shape[0] + 1):
        x_test.append(data[reshape_i-timesteps:reshape_i])
    
    # 預測部分往前推
    for reshape_i in range(train_bound-1, data.shape[0] + 1 - pred_steps):
        x_test_lowdim.append(data_low[reshape_i:reshape_i+pred_steps])
    
    x_test_out = np.array(x_test)
    x_test_low = np.array(x_test_lowdim)
    
    # 測試資料分成訓練和預測的部分  
    x_test_out = x_test_out.copy()[:-pred_steps]
    x_test_pred = x_test_low.copy()
    
    return x_train_out, x_test_out, x_train_pred, x_test_pred

def consturct_columns_name(number, have_date=True):
    
    out_list = []
    
    str_feature = 'feature'
    for i in range(number):
        out_list.append(str_feature + '_' + str(i+1))
    
    if have_date == True:
        out_list.append('Date')
        
    return out_list

#----------------------
#換成RMSE和MSE來計算LOSS
#----------------------

def rmse_mse(test, true):
    
    # tp = (test-true)**2
    # print(tp)
    # tp1 = tp.sum()
    # print(tp1)
    
    MSE = ((test-true)**2).sum()/true.size
    RMSE = np.sqrt(((test-true)**2).sum()/true.size)

    return RMSE, MSE


def main(if_train=False,pretrain_AEModel=''):
    print('Can not find the pretrain_AEModel file') if if_train is False and pretrain_AEModel == "" else print(f'pretrain_AEModel loaded from {pretrain_AEModel} \n prediction process starting')
    # 沒有分sheet
    drop_na = True
    All_data_dict = {}

    # 讀取資料
    training_process_data = './data/latest-45tic-data-preprocess.pkl'
    df_Top = pd.read_pickle(training_process_data)
    df=df_Top.loc['2012-02-13':,:] # retrive training data after 2012
    # df['day']=df.index.dayofweek
    df.rename({'Stock_ID':'stockno'},axis=1,inplace=True)
    TW_50_TICKER = df.stockno.unique()

    # 用字典(dict)儲存不同支股票的data，這樣比較好做

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
        

    All_data_dict = data_dict
    # 把全部股票的資料組起來

    # 這部分是紀錄了哪些資料佔了哪些位置
    every_df_length = []

    for key, items in All_data_dict.items():
        
        every_df_length.append(key)
        X_train_tp = items.to_numpy()    
        if key == 1101:
            X_train = X_train_tp
        else:
            X_train = np.vstack((X_train, X_train_tp))

    print(X_train)
    total_day = len(data_dict[2330].index.unique())
    print(f'tota day {total_day}')
    print(X_train[:total_day].shape)


    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    total_day = len(data_dict[1101].index.unique())
    print(f'tota day {total_day}')
    XX=X_train.reshape(len(TW_50_TICKER),total_day,25)
    print(XX.shape)

    stock_pool = TW_50_TICKER 

    # 設定訓練的長度和預測的長度
    timesteps = 30
    pred_steps = 5

    # 為算其他評分方式儲存(僅有在非單一訓練自己的AE才有)
    X_cla_dict = {}

    # 為最後輸出儲存
    X_save_dict = {}

    # 這是給訓練單一股票的
    All_data_train = {}

    # 算每支股票的總數
    sk_number = 0

    # 預測要預測幾維的資料
    pred_diem = True

    # 每支股票都訓練自己的AE
    only_one_stock = False

    for i in range(len(stock_pool)):

        X_train_tp1 = XX[i]
        print(X_train_tp1.shape)

        # 分成LSTM可以吃的形狀，且分成訓練和測試集
        X_train_tp_composite, X_test_tp_composite, X_train_pred_composite, X_test_pred_composite = transform_to_lstm_input_pred(timesteps, pred_steps, X_train_tp1, 0.3, dimen=0)

        # 為最後encoder做儲存
        X_train_s, X_test_s = transform_to_lstm_input(timesteps, X_train_tp1, 0)
        
        X_cla_dict[stock_pool[i]] = [X_train_tp_composite, X_train_pred_composite, X_test_tp_composite, X_test_pred_composite]
        X_save_dict[stock_pool[i]] = X_train_s

        if i == 0:
            X_train_out_composite = X_train_tp_composite
            X_test_out_composite = X_test_tp_composite
            X_train_out_pred_composite = X_train_pred_composite
            X_test_out_pred_composite = X_test_pred_composite
        else:
            X_train_out_composite = np.vstack((X_train_out_composite, X_train_tp_composite))
            X_test_out_composite = np.vstack((X_test_out_composite, X_test_tp_composite))
            X_train_out_pred_composite = np.vstack((X_train_out_pred_composite, X_train_pred_composite))
            X_test_out_pred_composite = np.vstack((X_test_out_pred_composite, X_test_pred_composite))

    if if_train:
        from tensorflow.keras import backend 
        from tensorflow.keras.layers import LeakyReLU
        from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

        model = Model()
        feature_nums = 5
        backend.clear_session()
        '''
        # define encoder
        visible = Input(shape=(X_train_out_composite.shape[1], X_train_out_composite.shape[2]))
        encoder = LSTM(feature_nums)(visible) # , return_sequences=True , activation='relu'
        encoder = LeakyReLU(alpha=0.3)(encoder)
        # encoder = LSTM(5, activation='relu')(encoder)

        # define reconstruct decoder
        decoder1 = RepeatVector(X_train_out_composite.shape[1])(encoder)
        decoder1 = LSTM(feature_nums,  return_sequences=True)(decoder1) # activation='relu',
        decoder1 = LeakyReLU(alpha=0.3)(decoder1)
        # decoder1 = LSTM(15, activation='relu', return_sequences=True)(decoder1)
        decoder1 = TimeDistributed(Dense(X_train_out_composite.shape[2]))(decoder1)

        # define predict decoder
        decoder2 = RepeatVector(X_train_out_pred_composite.shape[1])(encoder)
        decoder2 = LSTM(feature_nums, return_sequences=True)(decoder2) # activation='relu',
        decoder2 = LeakyReLU(alpha=0.3)(decoder2)
        decoder2 = TimeDistributed(Dense(X_train_out_pred_composite.shape[2]))(decoder2)

        # tie it together
        model = Model(inputs=visible, outputs=[decoder1, decoder2],name =f"composite_45tic_30d_{feature_nums}F")
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # backend.clear_session()

        print(f'Traing data dimension :')
        print(X_train_out_composite.shape)
        print(X_test_out_composite.shape)
        print(X_train_out_pred_composite.shape)
        print(X_test_out_pred_composite.shape)


        ### fit model
        use_earlyStop = True
        _min_delta = 0.01
        _patience = 150
        _monitor='val_loss'
        _epochs = 1000
        _batch_size = 128
        _shuffle = False
        _validation_split = 0.2

        if use_earlyStop == True:
            callback = EarlyStopping(monitor=_monitor, min_delta=_min_delta, patience=_patience)
            history = model.fit(X_train_out_composite, [X_train_out_composite, X_train_out_pred_composite], epochs=_epochs, batch_size = _batch_size, 
                                validation_split=_validation_split, callbacks=callback)
            
        else:
            history = model.fit(X_train_out_composite, [X_train_out_composite, X_train_out_pred_composite], epochs=_epochs, batch_size = _batch_size, 
                                validation_split=_validation_split)
        '''
        visible = Input(shape=(X_train_out_composite.shape[1], X_train_out_composite.shape[2]))
        # Encoder
        encoder = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(visible)
        encoder = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(encoder)

        # Reconstruct decoder
        decoder1 = RepeatVector(X_train_out_composite.shape[1])(encoder)
        decoder1 = LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(decoder1)
        decoder1 = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(decoder1)
        decoder1 = TimeDistributed(Dense(X_train_out_composite.shape[2]))(decoder1)

        # Predict decoder
        decoder2 = RepeatVector(X_train_out_pred_composite.shape[1])(encoder)
        decoder2 = LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(decoder2)
        decoder2 = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(decoder2)
        decoder2 = TimeDistributed(Dense(X_train_out_pred_composite.shape[2]))(decoder2)

        # Model
        model = Model(inputs=visible, outputs=[decoder1, decoder2], name=f"composite_45tic_30d_{feature_nums}F")
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        ### fit model
        use_earlyStop = True
        _min_delta = 0.01
        _patience = 150
        _monitor='val_loss'
        _epochs = 1000
        _batch_size = 128
        _shuffle = False
        _validation_split = 0.2

        # Callbacks
        callbacks = [
            EarlyStopping(monitor=_monitor, min_delta=_min_delta, patience=_patience),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, min_lr=1e-6),
            ModelCheckpoint(filepath='./dashboard/AE/best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
        ]


        # Training the model
        history = model.fit(
            X_train_out_composite,
            [X_train_out_composite, X_train_out_pred_composite],
            epochs=_epochs,
            batch_size=_batch_size,
            validation_split=_validation_split,
            callbacks=callbacks
        )

        # 計算RMSE MSE

        first_name = list(X_cla_dict.keys())[0]

        for key, items in X_cla_dict.items():
            
            if first_name == key:
                X_train_cal = items[0]
                X_train_pred = items[1]
                X_test_cal = items[2]
                X_test_pred = items[3]
            else:
                X_train_cal = np.vstack((X_train_cal, items[0]))
                X_train_pred = np.vstack((X_train_pred, items[1]))
                X_test_cal = np.vstack((X_test_cal, items[2]))
                X_test_pred = np.vstack((X_test_pred, items[3]))
                
        X_train_cal_pred = model.predict(X_train_cal)

        rmse_ae, mse_ae = rmse_mse(X_train_cal_pred[0], X_train_cal)
        rmse_pred, mse_pred = rmse_mse(X_train_cal_pred[1], X_train_pred)

        X_test_cal_pred = model.predict(X_test_cal)

        rmse_tae, mse_tae = rmse_mse(X_test_cal_pred[0], X_test_cal)
        rmse_tpred, mse_tpred = rmse_mse(X_test_cal_pred[1], X_test_pred)

        print('The RMSE in AE = ', rmse_ae)
        print('The MSE in AE = ', mse_ae)
        print('The RMSE in Pred = ', rmse_pred)
        print('The MSE in Pred = ', mse_pred)
        print('-----------------')
        print('The Train sum of RMSE in AE = ', rmse_ae + rmse_pred)
        print('The Train sum of MSE in AE = ', mse_ae + mse_pred)

        print('------------------------')
        print('The RMSE in AE = ', rmse_tae)
        print('The MSE in AE = ', mse_tae)
        print('The RMSE in Pred = ', rmse_tpred)
        print('The MSE in Pred = ', mse_tpred)
        print('-----------------')
        print('The Test sum of RMSE in AE = ', rmse_tae + rmse_tpred)
        print('The Test sum of MSE in AE = ', mse_tae + mse_tpred)


    from tensorflow.keras.models import load_model
    # Load the saved model
    saved_model = load_model(pretrain_AEModel)


    #model.load_weights('/Users/yi-hsuanlee/Desktop/WIDM/Thesis/finRL-elegant/dashboard/AE/AE_weights.h5')

    ##########Trend#############

    X_save_pred_dict_all = {}
    #有問題
    for key, items in X_save_dict.items():
        tmp_= saved_model.predict(X_save_dict[key])
        tmp_0 = tmp_[0].reshape(tmp_[0].shape[0] * tmp_[0].shape[1],tmp_[0].shape[2])
        tmp_0 = scaler.inverse_transform(tmp_0)
        # print(tmp_0)
        extract_day = 30 
        tmp_0=tmp_0.reshape(tmp_0.shape[0]//extract_day,extract_day,tmp_0.shape[1])
        tmp_1 = tmp_[1].reshape(tmp_[1].shape[0] * tmp_[1].shape[1],tmp_[1].shape[2])
        tmp_1 = scaler.inverse_transform(tmp_1)
        # print(tmp_1)
        extract_predictDay = 5
        tmp_1=tmp_1.reshape(tmp_1.shape[0]//extract_predictDay,extract_predictDay,tmp_1.shape[1])
        X_save_pred_dict_all[key] = [tmp_0,tmp_1]



    for key in All_data_dict:
        All_data_dict[key] = All_data_dict[key].iloc[29:]

    All_trend_df_dict= {}

    for key, items in All_data_dict.items():
        
        print('Stock ID ---> ', key, end=' | ')
        day=0
        try:
            final_out_df = pd.DataFrame(columns=['next_5d_trend'])
            
            for i , k in enumerate(X_save_pred_dict_all[key][1]):
                # a1 = np.array([x[0] for x in X_save_pred_dict_all[key][0][i]])
                a2 = np.array([x[0] for x in X_save_pred_dict_all[key][0][i]]) #using open price to check
                # xd=pd.Series(np.append(a1,a2))
                # xd= xd[:7]#xd[-7:]
                up_Down = 'down' if sum(1 for num in a2 if num < 0) > sum(1 for num in a2 if num > 0) else 'up'
                #up_Down = 'down' if any(a2<0) else 'up'
                temp_df = pd.DataFrame(up_Down,columns=['next_5d_trend'],index= [All_data_dict[key].index[i]])
                final_out_df =final_out_df.append(temp_df)
                day+=1
            print(f'{key} finished \n day {i}' )
            print(final_out_df.next_5d_trend.unique())
            #print(final_out_df)
            All_trend_df_dict[key] = final_out_df
        except Exception as e:
            print(f'Skip \n {e}')
            continue

    first_number = 0
    final_trend_df=pd.DataFrame(columns=['next_5d_trend','Stock id'])
    for key, item in All_trend_df_dict.items():
        
        temp_df = All_trend_df_dict[key]
        temp_df['Stock id'] = key
        final_df =temp_df

        final_trend_df = final_trend_df.append(temp_df)

    alg_type ='composite_lstm'
    tic = '45tic'
    timesteps = '30d'
    feature ='25F@5F'
    path = f'./data/latest_{alg_type}_{tic}_{timesteps}_{feature}_trend_eleganRl.pkl'
    final_trend_df.to_pickle(path)





    ##################Encoder########

    model_encoder = Model(inputs=saved_model.inputs, outputs=saved_model.layers[2].output)

    model_encoder.summary()

    # Get the all of data
    # 0908 make composite on 45tic data 25 features 7days.
    check_sum = 0
    X_save_pred_dict = {}

    for key, items in X_save_dict.items():
        
        X_save_pred_dict[key] = model_encoder.predict(X_save_dict[key])
        check_sum += X_save_pred_dict[key].shape[0]

    print('All of the data = ', check_sum)


    # 這是for lstm AE
    feature_size = 5
    timesteps = 30
    col_list = consturct_columns_name(feature_size,have_date=False)
    All_encoder_df_dict = {}
    for key in All_data_dict:
        new_df = pd.DataFrame(X_save_pred_dict[key][:, :5], index=All_data_dict[key].index, columns=col_list)
        All_encoder_df_dict[key] = new_df
        

    first_number = 0

    for key, item in All_encoder_df_dict.items():
        
        temp_df = All_encoder_df_dict[key]
        temp_df['Stock id'] = key
        
        if first_number == 0:
            final_df = temp_df
            first_number = 1
        else:
            final_df = pd.concat([final_df, temp_df], axis=0)

    alg_type ='composite_lstm'
    tic = '45tic'
    timesteps = '30d'
    feature ='25F@5F'
    path = f'./data/latest_{alg_type}_{tic}_{timesteps}_{feature}_eleganRl.pkl'
    final_df.to_pickle(path)
    print(f'Export feature success!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--if_AEtrain', type=bool, default=False, help='True for training, False for prediction')
    # Load model or not
    cmd_args = parser.parse_args()
    if_train=cmd_args.if_AEtrain
    pretrain_AEModel='./AE/best_model.h5'
    main(if_train,pretrain_AEModel)