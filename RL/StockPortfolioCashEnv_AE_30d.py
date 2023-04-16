import os,math
import pandas as pd
import numpy as np
import numpy.random as rd
import torch
import glob,pickle
from gym import spaces
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import random
import scipy.special as sc
from sklearn import preprocessing




import logging,sys
from datetime import datetime
def myLogger(name):
    nowTime =datetime.now().strftime("%Y%m%d-%H%M%S")
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(f"./RL/log/elegantRL_{nowTime}.log", mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger
logger = myLogger('elegantRL')



class StockPortfolioCashEnv_AE_30d:
    metadata = {'render.modes': ['human']}

    def extract_month_historical_interval_data(self,currentDay):
        import pandas as pd
        pastMonthlyHistoricalData = 12 * [0]
        pastMonthlyData = 12 * [0]

        if currentDay>= 360:
            for month in range(0,12):
                pastMonthD  = currentDay- (month*30)
                pastmonthData = self.totalDay[pastMonthD]
                pastMonthlyData[month] = pastmonthData
        elif currentDay < 360 and currentDay >=30 :
            for month in range(0,int(currentDay/30)):
                pastMonthD  = currentDay- (month*30)
                pastmonthData = self.totalDay[pastMonthD]
                pastMonthlyData[month] = pastmonthData   
            df_temp = pd.DataFrame({'test': pastMonthlyData})
            df2_temp = df_temp.replace(0, limit=12, method='ffill')
            pastMonthlyData= df2_temp.test.to_list()
        else:
            pastmonthData = self.totalDay[currentDay]
            pastMonthlyData = 12 * [pastmonthData]
        pastMonthlyHistoricalData = self.priceData.loc[pastMonthlyData]
        return pastMonthlyHistoricalData

    def extract_year_historical_interval_data(self, currentDay):
        pastyearlyHistoricalData = 3 * [0]
        pastyearlyData = 3 * [0]

        pastYear = currentDay -360 # past year already calculated by above
        if pastYear<360 :
            pastyearlyData = [self.totalDay[pastYear],self.totalDay[pastYear],self.totalDay[pastYear]]
        elif pastYear >=360 and pastYear<=720:
            pastyearlyData = [self.totalDay[pastYear],self.totalDay[pastYear-360],self.totalDay[pastYear-360]]
        else:
            pastyearlyData = [self.totalDay[pastYear],self.totalDay[pastYear-360],self.totalDay[pastYear-720]]
        # print(f'currentDay{currentDay}')
        # print(f'pastyearlyData{pastyearlyData}')
        pastyearlyHistoricalData = self.priceData.loc[pastyearlyData]

        return pastyearlyHistoricalData

    def reload_data(self, encoder_dataPath, data_trend_path, latest_priceBook_path , day,model_type):
        self.data_path = encoder_dataPath
        self.trend = pd.read_pickle(data_trend_path)
        self.priceData = pd.read_pickle(latest_priceBook_path)
        # self.__init__(if_eval=True,encoder_dataPath=encoder_dataPath,data_trend_path=data_trend_path,latest_priceBook_path=latest_priceBook_path,ddd=day)# All the required arguments for __init__ method)
        self.if_eval = True
        df = pd.read_pickle(self.data_path)
        training_days = df.index.unique()[len(df.index.unique())-model_type]
        df = df.loc[training_days.strftime('%Y-%m-%d'):]
        self.df = df
        self.totalDay = self.df.index.unique()
        self.day = day
        self.max_step = len(self.totalDay)-1 

    
       


    def __init__(self, cwd='./envs/FinRL', gamma=0.99,
                 max_stock=1e2, initial_capital=1e6, buy_cost_pct=1e-3, sell_cost_pct=1e-3,
                 start_date='2010-01-01', end_date='2020-01-01', env_eval_date='2022-10-31',
                 ticker_list=None, tech_indicator_list=None, initial_stocks=None, if_eval=False,turbulence_threshold=None,day = 0,if_PCA=False,if_AE_Trend=True,if_additional_data=True,model_type=30,if_past_month_year_data=False,encoder_dataPath=None,data_trend_path=None,latest_priceBook_path=None,ddd=None):

        # df,train_tech_ary,eval_tech_ary= self.load_data(cwd, if_eval, ticker_list, tech_indicator_list,
        #                                                start_date, end_date, env_eval_date, )  
        self.model_type=model_type
        self.if_eval = if_eval
        self.data_path = encoder_dataPath #'/Users/yi-hsuanlee/Desktop/WIDM/Thesis/finRL-elegant/dashboard/AE/different_weights/60day/composite_lstm_45tic_60_26F@5F_eleganRl.pkl'  #'/Users/yi-hsuanlee/Desktop/WIDM/Thesis/交接的Code和Data/Data/2.5_AE_45_tic/newTill2021/composite_lstm_45tic_30d_26F@5F_eleganRl.pkl' #'/Users/yi-hsuanlee/Desktop/WIDM/Thesis/交接的Code和Data/Data/2.5_AE_45_tic/newTill2021/special/huge_dimension_26F.pkl'#
        data_trend_path = data_trend_path #'/Users/yi-hsuanlee/Desktop/WIDM/Thesis/finRL-elegant/dashboard/AE/different_weights/30day_bk/composite_lstm_45tic_30d_26F@5F_trend_eleganRl.pkl' #'/Users/yi-hsuanlee/Desktop/WIDM/Thesis/交接的Code和Data/Data/2.5_AE_45_tic/newTill2021/newTill2021composite_lstm_45tic_30d_26F@5F_trend_eleganRl.pkl'
        # training_days = df.index.unique()[len(df.index.unique())-self.model_type]
        if not if_eval:
            df = pd.read_pickle(self.data_path)#("/Users/yi-hsuanlee/Desktop/CSIE/WIDM/Thesis/交接的Code和Data/Data/2.5_AE_45_tic/composite_lstm_45tic_30d_26F_5F_eleganRl.pkl")
            df = df.loc[:'2019-01-03']
            #df = df.loc[:training_days.strftime('%Y-%m-%d')]
        else:
            # df = df.loc[training_days.strftime('%Y-%m-%d'):]
            df = pd.read_pickle(self.data_path)#("/Users/yi-hsuanlee/Desktop/CSIE/WIDM/Thesis/交接的Code和Data/Data/2.5_AE_45_tic/composite_lstm_45tic_30d_26F_5F_eleganRl.pkl")
            df = df.loc['2019-01-04':]
        self.df = df
        self.priceData = pd.read_pickle(latest_priceBook_path) #pd.read_pickle("/Users/yi-hsuanlee/Desktop/WIDM/Thesis/交接的Code和Data/Data/2.5_AE_45_tic/newTill2021/2012-2021_priceBook.pkl")
        #self.priceData = self.priceData[self.priceData["stockno"].isin(['2330','2615'])]

        self.totalDay = self.df.index.unique()
        self.tic = self.df["Stock id"].unique()
        # if not if_eval:
        #     self.tech_ary = train_tech_ary
        # else:
        #     self.tech_ary = eval_tech_ary
        #self.tech_indicator_list = tech_indicator_list
        self.stock_dim = len(self.tic)
        self.gamma = gamma
        self.hmax = max_stock
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.transaction_cost_pct = buy_cost_pct
        self.initial_amount = initial_capital
        self.reward_scaling= 10
        # reset()
        self.cash = self.initial_amount
        if ddd is None:
            self.day = 0 if self.if_eval else random.randint(0,len(self.totalDay)-1-self.model_type)  #0 if self.if_eval else random.randint(0,len(self.totalDay)-1)
        else:
            self.day = ddd
        self.amount = None
        self.total_asset = None
        self.initial_total_asset = None
        self.gamma_reward = 0.0
        self.terminal = False 
        
        self.portfolio_value = self.initial_amount   
        
        self.portfolio_return_memory = [0]
        self.action_dim = self.stock_dim +1 # 1 means cash position
        #self.actions_memory=[[1/self.action_dim]*self.action_dim]
        self.asset_memory = [self.initial_amount]
        self.actions_memory=[np.insert(np.zeros(self.stock_dim),0,1)]
        self.date_memory=[self.totalDay[self.day]] 

        # environment information
        self.env_name = 'StockTradingPortfolioEnv_45tic_30d_-v1'
        self.max_step = len(self.totalDay)-1 if self.if_eval else len(self.totalDay)-1-self.model_type    
        self.if_discrete = False
        self.target_return = 3.5
        self.episode_return = 0.0
        self.PCA = if_PCA

        self.price = self.priceData.loc[self.totalDay[self.day]].close.values
        self.data_values = self.df.drop("Stock id",axis=1).loc[self.totalDay[self.day]].values
        self.data_values = self.data_values.reshape(self.data_values.shape[0]*self.data_values.shape[1])
        
        
        self.stocks = np.zeros(self.stock_dim)
        self.stocks_memory = [self.stocks]
        
        self.reward_memory = [self.gamma_reward]
        self.if_additional_data = if_additional_data
 
        portfolio_cash = [self.portfolio_value,self.cash]
        if self.if_additional_data:
            self.state = np.hstack([preprocessing.normalize([portfolio_cash]),preprocessing.normalize([self.price]),[self.data_values]])[0]
        else:
            self.state = np.hstack([preprocessing.normalize([portfolio_cash]),preprocessing.normalize([self.price])])[0]
        self.if_past_month_year_data = if_past_month_year_data
        if self.if_past_month_year_data and self.if_additional_data :
            pastMonthData  = self.extract_month_historical_interval_data(self.day)
            pastYearData  = self.extract_year_historical_interval_data(self.day)
            month_and_year_data=np.concatenate((pastMonthData.close.values,pastYearData.close.values))      
            self.state = np.hstack([preprocessing.normalize([portfolio_cash]),preprocessing.normalize([self.price]),preprocessing.normalize(month_and_year_data.reshape(1,-1)),[self.actions_memory[-1]],[self.data_values]])[0]
         
        self.state=self.state.astype(np.float64)
        self.state_dim= len(self.state)
        self.state_memory = [self.state]
        
        self.if_AE_Trend = if_AE_Trend
        self.trend =pd.read_pickle(data_trend_path) if self.if_AE_Trend else None
        self.trend_memory = [np.zeros(self.stock_dim)]
       




    def reset(self):
        self.reward_memory = [self.gamma_reward]
        self.asset_memory = [self.initial_amount]
        self.day = 0 if self.if_eval else random.randint(0,len(self.totalDay)-1-self.model_type)  #0 if self.if_eval else random.randint(0,len(self.totalDay)-1)
        self.start_day = self.day
        self.data_values = self.df.drop("Stock id",axis=1).loc[self.totalDay[0]].values
        self.data_values = self.data_values.reshape(self.data_values.shape[0]*self.data_values.shape[1])
        self.price = self.priceData.loc[self.totalDay[self.day]].close.values
        # load states
        #self.covs = self.data['cov_list'].values

        self.cash = self.initial_amount

        
        self.portfolio_value = self.initial_amount
        self.PCA = self.PCA
        
        self.terminal = False 
        self.portfolio_return_memory = [0]
        self.actions_memory=[np.insert(np.zeros(self.stock_dim),0,1)]#[[1/self.action_dim]*self.action_dim]
        self.date_memory=[self.totalDay[self.day]] 
        self.trend_memory = [np.zeros(self.stock_dim)]
        self.stocks = np.zeros(self.stock_dim)
        self.stocks_memory = [self.stocks]
        self.new_portfolio_return_list = [.0]

        portfolio_cash = [self.portfolio_value,self.cash]
        if self.if_additional_data:
            self.state = np.hstack([preprocessing.normalize([portfolio_cash]),preprocessing.normalize([self.price]),[self.data_values]])[0]
        else:
            self.state = np.hstack([preprocessing.normalize([portfolio_cash]),preprocessing.normalize([self.price])])[0]

        if self.if_past_month_year_data and self.if_additional_data :
            pastMonthData  = self.extract_month_historical_interval_data(self.day)
            pastYearData  = self.extract_year_historical_interval_data(self.day)
            month_and_year_data=np.concatenate((pastMonthData.close.values,pastYearData.close.values))      
            self.state = np.hstack([preprocessing.normalize([portfolio_cash]),preprocessing.normalize([self.price]),preprocessing.normalize(month_and_year_data.reshape(1,-1)),[self.actions_memory[-1]],[self.data_values]])[0]
        self.state=self.state.astype(np.float64)
        self.state_dim = len(self.state)
        self.state_memory = [self.state]
        self.day +=5
        return self.state

    def step(self, actions):
        #terminal indicate game over.
        if self.if_eval:
            self.terminal = self.day >= len(self.df.index.unique())
        else:
            self.terminal = (self.day >= len(self.df.index.unique())- self.model_type) #(self.day - self.start_day >= self.model_type) or 
        if self.terminal:

            df = pd.DataFrame(self.new_portfolio_return_list) #(self.portfolio_return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(),'r')
            plt.savefig('./RL/log/cumulative_reward.png')
            plt.close()
            
            plt.plot(self.new_portfolio_return_list,'r')#(self.portfolio_return_memory,'r')
            plt.savefig('./RL/log/rewards.png')
            plt.close()

            print("=================================")
            print("Train") if not self.if_eval else print("Eval")
            print(f"Start day {self.start_day}")
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(self.asset_memory[-1]))
            # reward = self.get_reward(self.new_portfolio_return_list,self.actions_memory) #self.get_reward(self.portfolio_return_memory,self.actions_memory)
            
            reward=(1 + np.mean(self.new_portfolio_return_list)) ** 52 - 1
            print(f"Reward {reward}")
            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            # if df_daily_return['daily_return'].std() !=0:
            #     sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
            #     df_daily_return['daily_return'].std()
            # print(f'Sharpe {sharpe}')
            print("=================================")
            return self.state, self.reward, self.terminal,{}
            
            
        else:
            if (any(np.isnan(actions))):
                print('Contains NAN stop')
            weights = self.rankTopAction(actions)  #self.softmax_normalization(actions) 
            self.price = self.priceData.loc[self.totalDay[self.day]].close.values

            if self.day ==0:
                self.pre_price  = self.price
            else:
                self.pre_price = self.priceData.loc[self.totalDay[self.day-5]].close.values
            
           
            # update portfolio value
            self.cash = self.portfolio_value*weights[0]
            self.portfolio_value-=self.cash
            #cal action/trend diff
            if self.if_AE_Trend:
                #trends = self.iter_nextfiveDay_trend(self.trend,self.day)
                trends = self.trend.loc[self.totalDay[self.day]]
                real_action , trend_action_diff_score, trend=self.trend_action_diff(weights,trends)
                self.actions_memory.append(real_action)
                self.trend_memory.append(trend)
                trendVSweightValue = len([x for x,y in zip(np.where(actions[1:]>0,1,0),trend) if x == y])/45
                trendVSweightValue = np.array(trendVSweightValue, dtype=np.float32)
                trendVSweightValue = np.array(.0,dtype=np.float32) if trendVSweightValue =={} else trendVSweightValue
            else:
                self.actions_memory.append(weights)
                trend_action_diff_score = 0
            
            

            portfolio_return = sum(((self.price / self.pre_price)-1)*self.actions_memory[-2][1:])
            #cal stocks transaction_cost
            self.stocks = ((self.portfolio_value * self.actions_memory[-1][1:])/(self.price)).round(3)
            #self.stocks = np.where(self.stocks==0,1,self.stocks)           
            self.stocks_memory.append(self.stocks)
            stocks_diff= abs(self.stocks_memory[-1] - self.stocks_memory[-2])
            # calculate transaction cost 
            #transaction_cost =int(np.sum(np.where(stocks_diff>0,stocks_diff*self.data.close.values*1000*self.transaction_cost_pct,stocks_diff)))

            transaction_cost =int(np.sum(np.where(stocks_diff>0,stocks_diff*(self.price)*self.transaction_cost_pct,stocks_diff)))

            #new_portfolio_value = self.portfolio_value*(1+portfolio_return) + self.cash -transaction_cost
            new_portfolio_value = self.portfolio_value*(1+portfolio_return) + self.cash -transaction_cost
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.totalDay[self.day])            
            self.asset_memory.append(new_portfolio_value)
            new_portfolio_return  = (self.asset_memory[-1]-self.asset_memory[-2])/self.asset_memory[-1]
            self.new_portfolio_return_list.append(new_portfolio_return)

            portfolio_cash = [self.portfolio_value,self.cash]
            

            

            if self.if_additional_data:
                self.data_values = self.df.drop("Stock id",axis=1).loc[self.totalDay[self.day]].values
                if len(self.data_values) == 1100:
                    print('stop')
                if self.PCA ==True:
                    #calc PCA
                    tic_pca=PCA(n_components=3)
                    tic_pca.fit(self.data_values)
                    self.data_values = tic_pca.transform(self.data_values)
                self.data_values = self.data_values.reshape(self.data_values.shape[0]*self.data_values.shape[1])
                self.state = np.hstack([preprocessing.normalize([portfolio_cash]),preprocessing.normalize([self.price]),[self.data_values]])[0]
            else:
                self.state = np.hstack([preprocessing.normalize([portfolio_cash]),preprocessing.normalize([self.price]),[self.data_values]])[0]
            
            
            if self.if_past_month_year_data:
                pastMonthData  = self.extract_month_historical_interval_data(self.day)
                pastYearData  = self.extract_year_historical_interval_data(self.day)
                month_and_year_data=np.concatenate((pastMonthData.close.values,pastYearData.close.values))
                self.state = np.append(a,self.state , preprocessing.normalize(month_and_year_data.reshape(1,-1)))
            
            reward = new_portfolio_return #self.get_reward(self.new_portfolio_return_list,self.actions_memory,trend_action_diff_score) #self.get_reward(self.portfolio_return_memory,self.actions_memory)
            self.reward = reward*self.reward_scaling
            self.state_memory.append(self.state)
            self.reward_memory.append(self.reward)
            
            self.day +=5
            
            
            # trend = np.array(trend, dtype=np.float32)
            # trend = np.array(.0,dtype=np.float32) if len(trend) != 45 else trend
        return self.state, self.reward, self.terminal , dict()#, trendVSweightValue


    def rankTopAction(self,actions):
        actions =torch.from_numpy(actions)
        tmp_action = torch.zeros(actions.shape,dtype=torch.float)
        ind=torch.topk(actions, k=math.ceil(len(actions)/10), axis=0).indices
        val=torch.topk(actions, k=math.ceil(len(actions)/10), axis=0).values
        tmp_action[ind]=val
        tmp_action[0] = actions[0] #assign for cash
        actions = self.maskSoftMax(tmp_action)
        return actions.numpy()

   


    def softmax(self,x: np.ndarray) -> np.ndarray:
        return np.exp(x - sc.logsumexp(x))

    def trend_action_diff(self,weights,trend):

        tmp_action = np.zeros(weights.shape)
        trend_action = np.zeros(len(weights)-1)
        #trend_ind = np.where(np.array(trend) == 1)[0]
        trend_ind = np.where(np.array(trend) == 'up')[0]
        action_ind = np.where(weights[1:]>0)
        buy_signal_ind = np.intersect1d(trend_ind, action_ind)
        
        
        if len(buy_signal_ind) !=0:
            if len(buy_signal_ind) >5:
                print('stop')
            tmp_action[1:][buy_signal_ind] = weights[1:][buy_signal_ind]
            tmp_action[0] = 1.0-tmp_action.sum()
            reweightActions = tmp_action
            score = 1
            trend_action[trend_ind] = 1.
            return reweightActions, score, trend_action
        else:
            new_ind=np.insert(trend_ind,0,0) # insert cash index
            tmp_action[new_ind] = 1/len(new_ind) #re_allocation actions
            tmp_action[0]+=(1.0-tmp_action.sum())
            reweightActions = tmp_action
            score = -1
            trend_action[trend_ind] = 1.
            return reweightActions, score, trend_action

    
    def iter_nextfiveDay_trend(self,trend,today):
        isTrendinDown =[False] * 45
        tmp_stock_list = trend['Stock id'].unique()

        for day in range(1,6):
            trend_df = trend.loc[self.totalDay[today+day]]
            if any (trend_df.next_5d_trend == 'down'):
                down_list = trend_df[trend_df.next_5d_trend=='down']['Stock id'].values
                tmp_isTrendinDown_df=pd.DataFrame({'tmp': tmp_stock_list}).isin(down_list)['tmp']
                exist_down_list = tmp_isTrendinDown_df.index[tmp_isTrendinDown_df.values == True].tolist()

                isTrendinDown = [True if x in exist_down_list else y for x,y in enumerate(isTrendinDown)]

        trend_list=[not x for x in isTrendinDown]

        return  list(map(int,trend_list))






        

        

    def maskSoftMax(self,tensorX):
        mask = ((tensorX  > 0).float() - 1) * 9999  # for -inf
        result = (tensorX + mask).softmax(dim=-1)
        return result

    def get_reward(self,portfolio_return,actions_memory,trend_action_diff_score=0):
        df_daily_return = pd.DataFrame(portfolio_return)
        df_daily_return.columns = ['daily_return']

        if self.day == 0:
            reward = .0
        else:
            reward = portfolio_return[-1]
            reward = (reward - abs(reward*1e-1)) if trend_action_diff_score == -1 else reward 
        return reward
        
    def shift_array(self,array, place):
        new_arr = np.roll(array, place, axis=0)
        new_arr[:place] = np.zeros((new_arr[:place].shape))
        return new_arr

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions 

    # def load_data(self, cwd='./envs/FinRL', if_eval=None,
    #               ticker_list=None, tech_indicator_list=None,
    #               start_date='2010-01-01', end_date='2016-01-01', env_eval_date='2021-01-01'):
    #     raw_data_path = f'{cwd}/StockTradingEnv_raw_data.df'
    #     processed_data_path = f'{cwd}/StockTradingEnv_processed_data_new0819.df'
    #     data_path_array = f'{cwd}/StockTradingEnv_arrays_float16_portfolio.npz'
    #     tech_indicator_list = [
    #         'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma'
    #     ] if tech_indicator_list is None else tech_indicator_list

    #     # ticker_list = [
    #     #     'AAPL', 'MSFT', 'JPM', 'V', 'RTX', 'PG', 'GS', 'NKE', 'DIS', 'AXP', 'HD',
    #     #     'INTC', 'WMT', 'IBM', 'MRK', 'UNH', 'KO', 'CAT', 'TRV', 'JNJ', 'CVX', 'MCD',
    #     #     'VZ', 'CSCO', 'XOM', 'BA', 'MMM', 'PFE', 'WBA', 'DD'
    #     # ] if ticker_list is None else ticker_list  # finrl.config.DOW_30_TICKER
    #     ticker_list = [
    #         'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'ALGN', 'ALXN', 'AMAT', 'AMD', 'AMGN',
    #         'AMZN', 'ASML', 'ATVI', 'BIIB', 'BKNG', 'BMRN', 'CDNS', 'CERN', 'CHKP', 'CMCSA',
    #         'COST', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'CTXS', 'DLTR', 'EA', 'EBAY', 'FAST',
    #         'FISV', 'GILD', 'HAS', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG',
    #         'JBHT', 'KLAC', 'LRCX', 'MAR', 'MCHP', 'MDLZ', 'MNST', 'MSFT', 'MU', 'MXIM',
    #         'NLOK', 'NTAP', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PEP', 'QCOM', 'REGN',
    #         'ROST', 'SBUX', 'SIRI', 'SNPS', 'SWKS', 'TTWO', 'TXN', 'VRSN', 'VRTX', 'WBA',
    #         'WDC', 'WLTW', 'XEL', 'XLNX'
    #     ] if ticker_list is None else ticker_list  # finrl.config.NAS_74_TICKER
    #     # ticker_list = [
    #     #     'AMGN', 'AAPL', 'AMAT', 'INTC', 'PCAR', 'PAYX', 'MSFT', 'ADBE', 'CSCO', 'XLNX',
    #     #     'QCOM', 'COST', 'SBUX', 'FISV', 'CTXS', 'INTU', 'AMZN', 'EBAY', 'BIIB', 'CHKP',
    #     #     'GILD', 'NLOK', 'CMCSA', 'FAST', 'ADSK', 'CTSH', 'NVDA', 'GOOGL', 'ISRG', 'VRTX',
    #     #     'HSIC', 'BIDU', 'ATVI', 'ADP', 'ROST', 'ORLY', 'CERN', 'BKNG', 'MYL', 'MU',
    #     #     'DLTR', 'ALXN', 'SIRI', 'MNST', 'AVGO', 'TXN', 'MDLZ', 'FB', 'ADI', 'WDC',
    #     #     'REGN', 'LBTYK', 'VRSK', 'NFLX', 'TSLA', 'CHTR', 'MAR', 'ILMN', 'LRCX', 'EA',
    #     #     'AAL', 'WBA', 'KHC', 'BMRN', 'JD', 'SWKS', 'INCY', 'PYPL', 'CDW', 'FOXA', 'MXIM',
    #     #     'TMUS', 'EXPE', 'TCOM', 'ULTA', 'CSX', 'NTES', 'MCHP', 'CTAS', 'KLAC', 'HAS',
    #     #     'JBHT', 'IDXX', 'WYNN', 'MELI', 'ALGN', 'CDNS', 'WDAY', 'SNPS', 'ASML', 'TTWO',
    #     #     'PEP', 'NXPI', 'XEL', 'AMD', 'NTAP', 'VRSN', 'LULU', 'WLTW', 'UAL'
    #     # ] if ticker_list is None else ticker_list  # finrl.config.NAS_100_TICKER
    #     # print(raw_df.loc['2000-01-01'])
    #     # j = 40000
    #     # check_ticker_list = set(raw_df.loc.obj.tic[j:j + 200].tolist())
    #     # print(len(check_ticker_list), check_ticker_list)

    #     '''get: train_price_ary, train_tech_ary, eval_price_ary, eval_tech_ary'''
    #     if os.path.exists(data_path_array):
    #         load_dict = np.load(data_path_array)

    #         train_price_ary = load_dict['train_price_ary'].astype(np.float32)
    #         train_tech_ary = load_dict['train_tech_ary'].astype(np.float32)
    #         eval_price_ary = load_dict['eval_price_ary'].astype(np.float32)
    #         eval_tech_ary = load_dict['eval_tech_ary'].astype(np.float32)
    #     # elif glob.glob('/Users/yi-hsuanlee/Desktop/CSIE/WIDM/Thesis/finRL-elegant/*.pkl'):
    #     #     train_df = pd.read_pickle('/Users/yi-hsuanlee/Desktop/CSIE/WIDM/Thesis/finRL-elegant/portfolio_train.pkl')
    #     #     eval_df = pd.read_pickle('/Users/yi-hsuanlee/Desktop/CSIE/WIDM/Thesis/finRL-elegant/portfolio_eval.pkl')
    #     #     train_tech_ary = self.convert_df_to_ary(train_df, tech_indicator_list)
    #     #     eval_tech_ary = self.convert_df_to_ary(eval_df, tech_indicator_list)

    #     else:
    #         processed_df= pd.read_pickle(processed_data_path)
    #         # processed_df = self.processed_raw_data(raw_data_path, processed_data_path,
    #         #                                        ticker_list, tech_indicator_list)

    #         def data_split(df, start, end,column_list):
    #             data = df[(df.date >= start) & (df.date < end)]
    #             data = data.sort_values(["date", "tic"], ignore_index=True)
    #             data.index = data.date.factorize()[0]
    #             data.columns = column_list
    #             print(data)
    #             return data

    #         def addCovMatrix(df):
    #             # add covariance matrix as states
    #             df=df.sort_values(['date','tic'],ignore_index=True)
    #             df.index = df.date.factorize()[0]

    #             cov_list = []
    #             # look back is one year
    #             lookback=252
    #             for i in range(lookback,len(df.index.unique())):
    #                 data_lookback = df.loc[i-lookback:i,:]
    #                 price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
    #                 return_lookback = price_lookback.pct_change().dropna()
    #                 covs = return_lookback.cov().values 
    #                 cov_list.append(covs)
                
    #             df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list})
    #             df = df.merge(df_cov, on='date')
    #             df = df.sort_values(['date','tic']).reset_index(drop=True)
    #             return df

    #         processed_df=addCovMatrix(processed_df)
    #         column_list=processed_df.columns.values.tolist()
    #         if 'volume'in column_list:
    #             processed_df.volume =  processed_df.volume * 1e-6  
    #         else : pass
    #                     #scale the volume
    #         if 'TWII_volume'in column_list:
    #             processed_df.TWII_volume =  processed_df.TWII_volume * 1e-6
    #             processed_df.FI =  processed_df.FI * 1e-6
    #             processed_df.MarginPurchaseTodayBalance =  processed_df.MarginPurchaseTodayBalance * 1e-6
    #             processed_df.ShortSaleTodayBalance =  processed_df.ShortSaleTodayBalance * 1e-6
                
    #         else : pass
    #         train_df = data_split(processed_df, start_date, end_date,column_list)
    #         eval_df = data_split(processed_df, end_date, env_eval_date,column_list)

    #         # train_price_ary, train_tech_ary = self.convert_df_to_ary(train_df, tech_indicator_list)
    #         # print(f'train_tech_ary_\n{train_tech_ary}')
    #         # eval_price_ary, eval_tech_ary = self.convert_df_to_ary(eval_df, tech_indicator_list)
            
    #         train_df.to_pickle("./evan_portfolio/portfolio_train.pkl")
    #         eval_df.to_pickle("./evan_portfolio/portfolio_eval.pkl")
    #         train_price_ary, train_tech_ary = self.convert_df_to_ary(train_df, tech_indicator_list)
    #         eval_price_ary, eval_tech_ary = self.convert_df_to_ary(eval_df, tech_indicator_list)
            
            

    #         np.savez_compressed(data_path_array,
    #                             train_price_ary=train_price_ary.astype(np.float16),
    #                             train_tech_ary=train_tech_ary.astype(np.float16),
    #                             eval_price_ary=eval_price_ary.astype(np.float16),
    #                             eval_tech_ary=eval_tech_ary.astype(np.float16), )

    #     if not if_eval:
    #         price_ary = np.concatenate((train_price_ary, eval_price_ary), axis=0)
    #         tech_ary = np.concatenate((train_tech_ary, eval_tech_ary), axis=0)
    #     elif if_eval:
    #         price_ary = eval_price_ary
    #         tech_ary = eval_tech_ary
    #         returnRF= eval_df
    #     else:
    #         price_ary = train_price_ary
    #         tech_ary = train_tech_ary
    #         returnRF = train_df
    #     return returnRF,train_tech_ary , eval_tech_ary

    def processed_raw_data(self, raw_data_path, processed_data_path,
                           ticker_list, tech_indicator_list):
        if os.path.exists(processed_data_path):
            processed_df = pd.read_pickle(processed_data_path)  # DataFrame of Pandas
            # print('| processed_df.columns.values:', processed_df.columns.values)
            print(f"| load data: {processed_data_path}")
        else:
            print("| FeatureEngineer: start processing data (2 minutes)")
            fe = FeatureEngineer(use_turbulence=True,
                                 user_defined_feature=False,
                                 use_technical_indicator=True,
                                 tech_indicator_list=tech_indicator_list, )
            raw_df = self.get_raw_data(raw_data_path, ticker_list)

            processed_df = fe.preprocess_data(raw_df)
            processed_df.to_pickle(processed_data_path)
            print("| FeatureEngineer: finish processing data")

        '''you can also load from csv'''
        # processed_data_path = f'{cwd}/dow_30_daily_2000_2021.csv'
        # if os.path.exists(processed_data_path):
        #     processed_df = pd.read_csv(processed_data_path)
        return processed_df

    @staticmethod
    def get_raw_data(raw_data_path, ticker_list):
        if os.path.exists(raw_data_path):
            raw_df = pd.read_pickle(raw_data_path)  # DataFrame of Pandas
            # print('| raw_df.columns.values:', raw_df.columns.values)
            print(f"| load data: {raw_data_path}")
        else:
            print("| YahooDownloader: start downloading data (1 minute)")
            raw_df = YahooDownloader(start_date="2010-01-01",
                                     end_date="2021-01-01",
                                     ticker_list=ticker_list, ).fetch_data()
            raw_df.to_pickle(raw_data_path)
            print("| YahooDownloader: finish downloading data")
        return raw_df

    @staticmethod
    def convert_df_to_ary(df, tech_indicator_list):
        tech_ary = list()
        price_ary = list()
        for i in range(len(df.index.unique())):
            # if len(df.loc[i])!=
            tech_items = [df.loc[i][tech].values.tolist() for tech in tech_indicator_list]
            tech_items_flatten = sum(tech_items, [])
            tech_ary.append(tech_items_flatten)
            price_ary.append(df.loc[i].close)  # adjusted close price (adjcp)
            #sum(tech_items, [])

        price_ary = np.array(price_ary)
        tech_ary = np.array(tech_ary)
        print(f'| price_ary.shape: {price_ary.shape}, tech_ary.shape: {tech_ary.shape}')
        return price_ary,tech_ary

    def evan_check_stock_trading_env(self,args,if_train):
        asset_memory, dd_action =self.evan_draw_cumulative_return(args, torch,if_train)
        return asset_memory, dd_action
        
    def evan_draw_cumulative_return(self, args, _torch,if_train) -> list:
        
        agent = args.agent
        net_dim = args.net_dim
        cwd = args.cwd

        # agent.init(net_dim, state_dim, action_dim)
        # agent.save_load_model(cwd=cwd, if_save=True)
        act = agent.act 
        device = agent.device

        state = self.reset()
        episode_returns = list()  # the cumulative_return / initial_account

        while if_train:
            with _torch.no_grad():
                for i in range(self.max_step):
                    s_tensor = _torch.as_tensor((state,), device=device)
                    a_tensor = act(s_tensor.float())
                    action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
                    state, reward, done, _ = self.step(action)

                    if done:
                        episode_returns = self.asset_memory
                        break

            #plotting return
            dd_return=pd.DataFrame(np.array(self.asset_memory),index = self.date_memory,columns = ['daily_returns'])        
            #saving action 
            dd_action=pd.DataFrame(np.array(self.actions_memory),index = self.date_memory, columns = np.insert(self.tic,0,0)) #remove cash : np.delete(np.array(self.actions_memory),0,axis=1)
            dd_action['Asset'] = np.array(self.asset_memory)
            return self.asset_memory, dd_action
        
        current_step = 0
        while not if_train:
            import time
            with _torch.no_grad():
                while self.day < self.max_step:
                    action, _ = agent.select_action(state)
                    # s_tensor = _torch.as_tensor((state,), device=device)
                    # a_tensor = act(s_tensor.float())
                    # action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
                    state, reward, done, _ = self.step(action)
                    current_step += 1
                    done = True if self.day>=self.max_step-5 else False

                    if done:
                        tmp_dict={}
                        #saving action 
                        dd_action=pd.DataFrame(np.array(self.actions_memory),index = self.date_memory, columns = np.insert(self.tic,0,0)) #remove cash : np.delete(np.array(self.actions_memory),0,axis=1)
                        dd_action['Asset'] = np.array(self.asset_memory)
                        portfolio = dd_action[dd_action.columns[(dd_action != 0).any()]]
                        pricedd = self.priceData
                        pricedd.rename({'Stock_ID':'stockno'},axis=1,inplace=True)
                        price_df =pricedd.loc['2019-01-04':]
                        price_df_change=price_df.pivot_table(columns='stockno',values='change',index=price_df.index)
                        price_df_close=price_df.pivot_table(columns='stockno',values='close',index=price_df.index)
                        tmp_dict['allocation_final_return'] = int(dd_action['Asset'][-1])
                        dd_action=dd_action.drop(['Asset',0], axis = 1)
                        tic  = dd_action.columns
                        inter_dateRange=dd_action.index
                        start_date = pd.Timestamp(inter_dateRange[0])  - pd.DateOffset(days=7)
                        start_date =price_df_close.iloc[0].name  if start_date < price_df_close.iloc[0].name else start_date
                        end_date = pd.Timestamp(inter_dateRange[-1])  + pd.DateOffset(days=7)
                        inter_dateRange_additional = np.hstack((np.datetime64(start_date),np.array(inter_dateRange), np.datetime64(end_date)))
                        price_df_close= price_df_close.loc[inter_dateRange_additional]
                        price_df_change=price_df_close.pct_change()
                        tmp_val = price_df_change[2:].values * dd_action.values
                        tmp_ind = dd_action.index
                        weight_returns= pd.DataFrame(tmp_val,columns = tic,index= tmp_ind)
                        weight_returns_sum= weight_returns.sum(axis=1)
                        print(f'weight_returns {weight_returns_sum}')
                        portfolio_cum_ret = (1 + weight_returns_sum).cumprod() - 1
                        print(f'portfolio_cum_rtn {portfolio_cum_ret[-1]}')
                        
                        tmp_dict['start_date'] = start_date
                        tmp_dict['end_date'] = end_date
                        tmp_dict['allocation_targets'] = portfolio
                        tmp_dict['allocation_final_cum_return'] = portfolio_cum_ret[-1]
                        with open('./dashboard/data/latest_test.pkl', 'wb') as handle:
                            pickle.dump(tmp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        # Wait for one week (in seconds)
                        time.sleep(3) #time.sleep(7 * 24 * 60 * 60)
                        # Reload the data
                        self.reload_data(args.encoder_dataPath, args.data_trend_path, args.latest_priceBook_path,self.day,self.model_type)
                        break


    def plot(self,date,data,title,xlabel,ylabel,output='/test.jpg',legend=None):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter
        import datetime as dt
        x=date
        y=data
        fig, ax=plt.subplots(1, 1, figsize=(30, 10))
        left = x[0].strftime("%Y-%m-%d") #datetime.strptime(x[0] , '%Y-%m-%d')
        

        right = x[-1].strftime("%Y-%m-%d")  #datetime.strptime(x[-1], '%Y-%m-%d')
        right = dt.date(right.year,right.month,right.day)
        x=pd.to_datetime(date)
                        
            
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30)) 
        plt.gca().xaxis.set_tick_params(rotation = 30)  

        plt.gca().set_xbound(left, right)
        plt.grid()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        df_eval = pd.read_pickle('')
        price=df_eval.pivot(index=df_eval.index,columns='stockno',values='close')
        price= price.loc[date]
        price['cash']=self.asset_memory
        price = self.normalize(price)
        if len(y.columns) >1 :
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{}%'.format(float(y)))) 
            plt.plot(x, y['cash'], '-g', label='cash')
            plt.plot(x, y['2330.tw'], '-b', label='2330.tw')
            plt.plot(x, y['2303.tw'], '-r', label='2303.tw')
            plt.plot(x, y['2603.tw'], '-y', label='2603.tw')
            ax2 = ax.twinx()
            ax2.plot(x,price['cash'] ,':g', label='cash')
            ax2.plot(x,price['2330.tw'] ,':b', label='2330.tw')
            ax2.plot(x,price['2303.tw'] ,':r', label='2303.tw')
            ax2.plot(x,price['2603.tw'] ,':y', label='2603.tw')
            plt.legend(labels=legend ,bbox_to_anchor=(0,1), loc='lower right')
        
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{}TWD'.format(float(y)))) 
            plt.plot(x,y.iloc[:,0])
        plt.savefig(output)
        plt.close()

    def normalize(self,df):
        result = df.copy()
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result

    def render(self, mode='human', close=False):
        pass

    def get_daily_return(self,df, value_col_name="account_value"):
        from copy import deepcopy
        df = deepcopy(df)
        df["daily_return"] = df[value_col_name].pct_change(1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True, drop=True)
        
        return pd.Series(df["daily_return"], index=df.index)

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output




def check_stock_trading_env():
    if_eval = True  # False

    env = StockPortfolioCashEnv_AE_30d(if_eval=if_eval)
    action_dim = env.action_dim

    state = env.reset()
    print('state_dim', len(state))

    from time import time
    timer = time()

    step = 1
    done = False
    reward = None
    while not done:
        action = rd.rand(action_dim) * 2 - 1
        next_state, reward, done, _ = env.step(action)
        # print(';', len(next_state), env.day, reward)
        step += 1

    print(f"| Random action: step {step}, UsedTime {time() - timer:.3f}")
    print(f"| Random action: terminal reward {reward:.3f}")
    print(f"| Random action: episode return {env.episode_return:.3f}")

    '''draw_cumulative_return'''
    from elegantrl.agent import AgentPPO
    from elegantrl.run import Arguments
    args = Arguments(if_on_policy=True)
    args.agent = AgentPPO()
    args.env = StockPortfolioCashEnv_AE_30d(if_eval=True)
    args.if_remove = False
    args.cwd = './AgentPPO/StockPortfolioEnv-v1_0-cash'
    args.init_before_training()

    env.draw_cumulative_return(args, torch)


"""Copy from FinRL"""


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API
    from finrl.marketdata.yahoodownloader import YahooDownloader
    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)
    Methods
    -------
    fetch_data()
        Fetches data from yahoo API
    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        import yfinance as yf  # Yahoo Finance
        """Fetches data from Yahoo API
        Parameters
        ----------
        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            temp_df = yf.download(tic, start=self.start_date, end=self.end_date)
            temp_df["tic"] = tic
            data_df = data_df.append(temp_df)
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop("adjcp", 1)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=['date', 'tic']).reset_index(drop=True)

        return data_df


class FeatureEngineer:
    """Provides methods for preprocessing the stock price data
    from finrl.preprocessing.preprocessors import FeatureEngineer
    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            user user defined features or not
    Methods
    -------
    preprocess_data()
        main method to do the feature engineering
    """

    def __init__(
            self,
            use_technical_indicator=True,
            tech_indicator_list=None,  # config.TECHNICAL_INDICATORS_LIST,
            use_turbulence=False,
            user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """

        if self.use_technical_indicator:
            # add technical indicators using stockstats
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="bfill").fillna(method="ffill")
        return df

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        from stockstats import StockDataFrame as Sdf  # for Sdf.retype

        df = data.copy()
        df = df.sort_values(by=['tic', 'date'])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator['tic'] = unique_ticker[i]
                    temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(indicator_df[['tic', 'date', indicator]], on=['tic', 'date'], how='left')
        df = df.sort_values(by=['date', 'tic'])
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    @staticmethod
    def add_user_defined_feature(data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)
        return df

    @staticmethod
    def calculate_turbulence(data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [.0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
                ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index


if __name__ == '__main__':
    check_stock_trading_env()
