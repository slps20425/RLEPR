from tutorial_run import *
from tutorial_agent import AgentPPO, AgentDDPG,AgentDiscretePPO
import argparse
from StockPortfolioCashEnv_AE_30d import StockPortfolioCashEnv_AE_30d

# Agent
args = Arguments(if_on_policy=True)
args.agent = AgentPPO() #AgentPPO() # AgentSAC(), AgentTD3(), AgentDDPG()
args.agent.if_use_gae = True

# Environment
# tickers = [x + '.tw' for x in config.TW_50_TICKER]
tickers = ['2330.tw','2303.tw','2603.tw']#,'USDTWD=X'] #last element is currency USDTWD
tech_indicator_list = ['macd','boll_ub', 'boll_lb', 'rsi_30','close_30_sma', 'close_60_sma']#,'FI','MarginPurchaseTodayBalance','ShortSaleTodayBalance','TWII_close','TWII_volume','turbulence']  # finrl.config.TECHNICAL_INDICATORS_LIST
initial_stocks = np.zeros(len(tickers), dtype=np.float32)
gamma = 0.995 #0.90
max_stock = 100
initial_capital = 5e5# 1e6
buy_cost_pct = 0.1425*1e-2
sell_cost_pct = 0.1425*1e-2
start_date = '2012-01-01' #'2010-01-01'
start_eval_date = '2018-01-01'
end_eval_date = '2021-10-31'
model_type = 0 # 250 , 500 , 750
if_PCA = False
if_AE_Trend = True
if_additional_data = True
if_past_month_year_data=False


if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument('--if_RLtrain', type=bool, default=False, help='True for training, False for evaluation')
  parser.add_argument('--encoder_dataPath', type=str,default='./data/latest_composite_lstm_45tic_30d_25F@5F_eleganRl.pkl', help='Path to the encoder data')
  parser.add_argument('--data_trend_path', type=str,default='./data/latest_composite_lstm_45tic_30d_25F@5F_trend_eleganRl.pkl', help='Path to the data trend file')
  parser.add_argument('--latest_priceBook_path', type=str,default='./data/latest_45tic_priceBook.pkl', help='Path to the latest price book file')
  cmd_args = parser.parse_args()
  encoder_dataPath = cmd_args.encoder_dataPath
  data_trend_path = cmd_args.data_trend_path
  latest_priceBook_path = cmd_args.latest_priceBook_path
  args.data_trend_path = data_trend_path
  args.encoder_dataPath = encoder_dataPath
  args.latest_priceBook_path = latest_priceBook_path
  args.env = StockPortfolioCashEnv_AE_30d('./evan_portfolio_cash_AE_30d', gamma, max_stock, initial_capital, buy_cost_pct, 
                            sell_cost_pct, start_date, start_eval_date, 
                            end_eval_date, tickers, tech_indicator_list, 
                            initial_stocks, if_eval=False, if_PCA=if_PCA, if_AE_Trend= if_AE_Trend,if_additional_data=if_additional_data,model_type=model_type,if_past_month_year_data=if_past_month_year_data,encoder_dataPath=encoder_dataPath,data_trend_path=data_trend_path,latest_priceBook_path=latest_priceBook_path)
  args.eval_env = StockPortfolioCashEnv_AE_30d('./evan_portfolio_cash_AE_30d', gamma, max_stock, initial_capital, buy_cost_pct, 
                            sell_cost_pct, start_date, start_eval_date, 
                            end_eval_date, tickers, tech_indicator_list, 
                            initial_stocks, if_eval=True,if_PCA = if_PCA, if_AE_Trend = if_AE_Trend,if_additional_data=if_additional_data,model_type=model_type,if_past_month_year_data=if_past_month_year_data,encoder_dataPath=encoder_dataPath,data_trend_path=data_trend_path,latest_priceBook_path=latest_priceBook_path)
  # Hyperparameters
  args.gamma = gamma
  args.break_step = int(5e6)
  args.net_dim = 2 ** 11
  args.max_step = args.env.max_step
  args.max_memo = args.max_step * 3
  args.batch_size = 4096 #2 ** 11
  args.repeat_times = 2 ** 2 #2 ** 4
  args.eval_gap = 2 ** 4
  args.eval_times1 = 2 ** 3
  args.eval_times2 = 2 ** 2
  args.if_allow_break = False
  args.rollout_num = 16 # the number of rollout workers (larger is not always faster)
  args.cwd= './RL/testing_temp'
  args.env.target_reward = 1.3
  args.eval_env.target_reward = 1.3
  args.action_repeats = 1
  if_train=cmd_args.if_RLtrain
  load_pretrained=False if if_train else True
  
  # train_and_evaluate(args,if_train=if_train, load_pretrained=load_pretrained, pretrained_path="/Users/yi-hsuanlee/Desktop/WIDM/Thesis/finRL-elegant/dashboard/testing_temp") # the training process will terminate once it reaches the target reward.
  train_and_evaluate(args, if_train=if_train, load_pretrained=load_pretrained, pretrained_path="./RL/best_agent")

