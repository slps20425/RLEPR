import gym  # not necessary
import time
from tutorial_agent import *
from tensorboardX import SummaryWriter
from datetime import datetime
import pickle



gym.logger.set_level(40)  # Block warning

nowTime = datetime.now().strftime("%Y%m%d")
writer = SummaryWriter(
    f'./RL/log/elegant_logs_{nowTime}')


class Arguments:
    def __init__(self, agent=None, env=None, if_on_policy=False):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.env = env  # the environment for training

        self.cwd = None  # current work directory. None means set automatically
        # remove the cwd folder? (True, False, None:ask me)
        self.if_remove = True
        self.break_step = 2 ** 24  # break training after 'total_step > break_step'
        # allow break training when reach goal (early termination)
        self.if_allow_break = True

        # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.visible_gpu = '0'
        # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.worker_num = 1
        # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)
        self.num_threads = 4

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 5e-4 # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3
        self.action_repeats= 2 #

        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 11  # the network width
            # num of transitions sampled from replay buffer.
            self.batch_size = self.net_dim * 5
            self.repeat_times = 2 ** 5  # collect target_step, then update network
            self.target_step = 2 ** 11  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            # GAE for on-policy sparse reward: Generalized Advantage Estimation.
            self.if_per_or_gae = True
        else:
            self.net_dim = 2 ** 8  # the network width
            # num of transitions sampled from replay buffer.
            self.batch_size = self.net_dim
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.target_step = 2 ** 11  # collect target_step, then update network
            self.max_memo = 2 ** 17  # capacity of replay buffer
            # PER for off-policy sparse reward: Prioritized Experience Replay.
            self.if_per_or_gae = False

        '''Arguments for evaluate'''
        self.eval_env = None  # the environment for evaluating. None means set automatically.
        self.eval_gap = 2 ** 6  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2  # number of times that get episode return in first
        self.random_seed = 0  # initialize random seed in self.init_before_training()

    def init_before_training(self, if_main):
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}_{self.env.env_name}_{self.visible_gpu}'

        if if_main:
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(
                    input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)


def train_and_evaluate(args,if_train, agent_id=0,load_pretrained=False, pretrained_path=None):
    def print_model_parameters(model):
        for name, param in model.named_parameters():
            print(name, param.data)
    '''init: Agent'''
    env = args.env
    agent = args.agent
    agent.init(args.net_dim, env.state_dim, env.action_dim,
               args.learning_rate, args.if_per_or_gae)
    if load_pretrained and pretrained_path is not None:
        args.init_before_training(if_main=False) # set false in case NOT remove my current dir
        print("Before loading the pretrained model:")
        print_model_parameters(agent.act)
        agent.save_or_load_agent(pretrained_path, if_save=False)
        print("\nAfter loading the pretrained model:")
        print_model_parameters(agent.act)
        '''init Evaluator'''
        eval_env = deepcopy(env) if args.eval_env is None else args.eval_env
        evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env,
                            args.eval_times1, args.eval_times2, args.eval_gap)
        evaluator.save_or_load_recoder(if_save=True)

    else:
        args.init_before_training(if_main=True) # set false in case NOT remove my current dir
        agent.save_or_load_agent(args.cwd, if_save=True)
        '''init Evaluator'''
        eval_env = deepcopy(env) if args.eval_env is None else args.eval_env
        evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env,
                            args.eval_times1, args.eval_times2, args.eval_gap)
        evaluator.save_or_load_recoder(if_save=True)



    '''init ReplayBuffer'''
    if agent.if_on_policy:
        buffer = list()

        def update_buffer(s_a_n_r_m):
            buffer[:] = s_a_n_r_m  # (state, action, noise, reward, mask)
            _steps = s_a_n_r_m[3].shape[0]  # buffer[3] = r_sum
            _r_exp = s_a_n_r_m[3].mean()  # buffer[3] = r_sum
            return _steps, _r_exp
    else:
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_dim,
                              action_dim=1 if env.if_discrete else env.action_dim)
        buffer.save_or_load_history(args.cwd, if_save=False)

        def update_buffer(state_other):
            _state = torch.as_tensor(state_other[0], dtype=torch.float32)
            _other = torch.as_tensor(state_other[1], dtype=torch.float32)
            buffer.extend_buffer(_state, _other)

            _steps = _other.size()[0]
            _r_exp = _other[:, 0].mean().item()  # other = (reward, mask, ...)
            return _steps, _r_exp

    '''start training'''
    cwd = args.cwd
    gamma = args.gamma
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    reward_scale = args.reward_scale
    repeat_times = args.repeat_times
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    

    agent.state = env.reset()

    if_train = if_train
    # Add these lines before the training loop
    patience = 100
    best_goal_reward = -np.inf
    evaluations_without_improvement = 0
    ratio_clip_increment = 0.015
    lambda_entropy_increment = 0.01
    lambda_gae_adv_increment = 0.005
    gamma_increment = 0.01

    ratio_clip_max = 0.5
    lambda_entropy_max = 0.15
    lambda_gae_adv_min = 0.80
    gamma_max = 0.99
    while if_train:
        with torch.no_grad():
            array_tuple = agent.explore_env(
                env, target_step, reward_scale, args.gamma, args.action_repeats)
            steps, r_exp = update_buffer(array_tuple)
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

        with torch.no_grad():
            if_reach_goal , r_max , r_avg = evaluator.evaluate_and_save(cwd,
                agent, steps, r_exp, logging_tuple)

            # Early stopping check
            if r_avg < r_max:
                evaluations_without_improvement += 1
                
            if evaluations_without_improvement >= 50:
                # Increment the parameters and clamp them to their respective upper limits
                agent.ratio_clip = min(agent.ratio_clip + ratio_clip_increment, ratio_clip_max)
                agent.lambda_entropy = min(agent.lambda_entropy + lambda_entropy_increment, lambda_entropy_max)
                agent.lambda_gae_adv = max(agent.lambda_gae_adv - lambda_gae_adv_increment, lambda_gae_adv_min)
                args.gamma = min(agent.lambda_gae_adv + gamma_increment, gamma_max)
                evaluations_without_improvement=0

                
            if evaluations_without_improvement >= patience:
                print("Early stopping triggered.")
                print(f'agent.ratio_clip= {agent.ratio_clip}\n')
                print(f'agent.lambda_entropy= {agent.lambda_entropy}\n')
                print(f'agent.lambda_gae_adv= {agent.lambda_gae_adv}\n')
                if_train=False
                break
            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))
        print(
            f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')
        # agent.save_or_load_agent(cwd, if_save=True)
        exp = False
        import pandas as pd
        import pickle
        if exp:
            save_dict = {}       
            try:
                for _ in range(2):               
                    print(f'{_}times')
                    tmp_dict= {}
                    asset_memory, dd_action=eval_env.evan_check_stock_trading_env(args,if_train)
                    print(f'this is Final Asset {dd_action} \n')
                    portfolio = dd_action[dd_action.columns[(dd_action != 0).any()]]
                    price = pd.read_pickle('./data/latest_45tic_priceBook.pkl')
                    price.rename({'Stock_ID':'stockno'},axis=1,inplace=True)
                    price_df =price.loc['2019-01-01':]
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
                    time.sleep(1)
                    #buffer.save_or_load_history(cwd, if_save=True)
                    save_dict[_] = tmp_dict
                with open('./data/latest_test_live_update.pkl', 'wb') as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except KeyError:
                print(f"error happening")
        else:
            pass

       
    if not if_train:
        evaluator.evaluate_and_save_evan(agent,args)
        #eval_env.evan_check_stock_trading_env(args,if_train)
    
    print(f'end')

class Evaluator:
    def __init__(self, cwd, agent_id, device, env, eval_times1, eval_times2, eval_gap, ):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f'{cwd}/recorder.npy'
        self.r_max = -np.inf
        self.total_step = 0

        self.env = env
        self.cwd = cwd
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.target_return = env.target_return

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = 0
        print(f"{'#' * 80}\n"
              f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
              f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>8}{'objC':>7}{'etc.':>7}")

    def evaluate_and_save_evan(self, agent,args) -> bool:
        get_episode_return_for_live_update(self.env, agent.act, self.device, args) 
        print(rewards_steps_list)

    def evaluate_and_save(self,cwd, agent, steps, r_exp, log_tuple) -> bool:
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:
            return False  # if_reach_goal

        self.eval_time = time.time()
        rewards_steps_list = [get_episode_return_and_step(self.env, agent.act, self.device) for _ in
                              range(self.eval_times2)]
        r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(
            rewards_steps_list)
        print(f'r_avg {r_avg}')

        if r_avg > self.r_max:  # save checkpoint with highest episode return
            self.r_max = r_avg  # update max reward (episode return)
            agent.save_or_load_agent(cwd, if_save=True)
            # act_save_path = f'{self.cwd}/actor.pth'
            # save policy network in *.pth
            # torch.save(act.state_dict(), act_save_path)
            # save policy and print
            print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |")

        '''
        adding tensorboard writer by evan 08132021
        '''
        global writer
        obj_c = log_tuple[0]
        obj_a = log_tuple[1]
#        logprob = log_tuple[2]
        writer.add_scalars('PPO', {
            'obj_a': obj_a,
            'obj_c': obj_c,
            #'logprob_mean': logprob,
            'episode_avg_return': r_avg,
        }, self.total_step)
        # writer.add_graph(agent.act)
        self.recorder.append(
            (self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder

        # recorder_ary.append((self.total_step, r_avg, r_std, obj_a, obj_c))
        recorder = np.array(self.recorder)
        stepss = recorder[:, 0]  # x-axis is training steps
        rr_avg = recorder[:, 1]
        rr_std = recorder[:, 2]
        obj_cc = recorder[:, 4]
        obj_aa = recorder[:, 5]
        self.plotting(self.cwd, rewards_steps_list,
                      stepss, rr_avg, rr_std, obj_cc, obj_aa)
        # check if_reach_goal
        if_reach_goal = bool(self.r_max > self.target_return)
        if if_reach_goal and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':<3}{'Step':>8}{'TargetR':>8} |"
                  f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                  f"{'UsedTime':>8}  ########\n"
                  f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                  f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                  f"{self.used_time:>8}  ########")

        print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
              f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
              f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}")
        return if_reach_goal, self.r_max, r_avg


    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
        # average of episode return and episode step
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)
        # standard dev. of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)
        return r_avg, r_std, s_avg, s_std

    @staticmethod
    def plotting(cwd, rewards_steps_list, steps, r_avg, r_std, obj_c, obj_a):
        import matplotlib.pyplot as plt

        print(f'drawing plot \n reward_list : {rewards_steps_list} \n steps: {steps} \n r_avg :{r_avg} r_std: {r_std} \n  obj_c: {obj_c} \n obj_a: {obj_a}\n')

        '''draw return graph'''
        plt.plot(rewards_steps_list)
        plt.grid()
        plt.title('cumulative return222222')
        plt.xlabel('day')
        plt.xlabel('multiple of initial_account')
        plt.savefig(f'{cwd}/cumulative_return.jpg')
    
        '''draw learning curve'''
        '''plot subplots'''
        import matplotlib as mpl
        mpl.use('Agg')
        """Generating matplotlib graphs without a running X server [duplicate]
        write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
        https://stackoverflow.com/a/4935945/9293137
        """
        import matplotlib.pyplot as plt
        plt.subplots_adjust(top=0.99, bottom=0.01, hspace=1.5, wspace=0.4)
        fig, axs = plt.subplots(2)

        axs0 = axs[0]
        axs0.cla()
        color0 = 'lightcoral'
        axs0.set_xlabel('Total Steps')
        axs0.set_ylabel('Episode Return')
        axs0.plot(steps, r_avg, label='Episode Return', color=color0)
        axs0.fill_between(steps, r_avg - r_std, r_avg +
                          r_std, facecolor=color0, alpha=0.3)

        ax11 = axs[1]
        ax11.cla()
        color11 = 'royalblue'
        axs0.set_xlabel('Total Steps')
        ax11.set_ylabel('objA', color=color11)
        ax11.plot(steps, obj_a, label='objA', color=color11)
        ax11.tick_params(axis='y', labelcolor=color11)

        ax12 = axs[1].twinx()
        color12 = 'darkcyan'
        ax12.set_ylabel('objC', color=color12)
        ax12.fill_between(steps, obj_c, facecolor=color12, alpha=0.2, )
        ax12.tick_params(axis='y', labelcolor=color12)

        '''plot save'''
        plt.tight_layout()
        plt.title('learning curve', y=2.3)
        plt.savefig(f"{cwd}/plot_learning_curve.jpg")
        # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
        plt.close('all')
        # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()

    def save_or_load_recoder(self, if_save):
        if if_save:
            np.save(self.recorder_path, self.recorder)
        elif os.path.exists(self.recorder_path):
            recorder = np.load(self.recorder_path)
            self.recorder = [tuple(i) for i in recorder]  # convert numpy to list
            self.total_step = self.recorder[-1][0]


def get_episode_return_and_step(env, act, device) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 0
    max_step = env.max_step+1
    if_discrete = env.if_discrete

    state = env.reset()
    for episode_step in range(max_step):
        # state = env.reset()
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor =act(s_tensor.float())  #act(s_tensor) 0820 for portfolio change
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        # not need detach(), because with torch.no_grad() outside
        action = a_tensor.detach().cpu().numpy()[0]
        state, reward, done, _ = env.step(action)
        if done:
            annualized_reward=(1 + np.mean(env.new_portfolio_return_list)) ** 52 - 1
            profit=(env.asset_memory[-1]-env.asset_memory[0])/env.asset_memory[0]
            print(f'annualized_reward: {annualized_reward}')
            print(f'last day asset: {env.asset_memory[-1]}')
            state = env.reset()
            break

    return annualized_reward, episode_step, 

def get_episode_return_for_live_update(env, act, device,args) -> (float, int):
    episode_step = 0
    max_step = env.max_step+1
    if_discrete = env.if_discrete

    state = env.reset()
    for episode_step in range(max_step):
        # state = env.reset()
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor =act(s_tensor.float())  #act(s_tensor) 0820 for portfolio change
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        # not need detach(), because with torch.no_grad() outside
        action = a_tensor.detach().cpu().numpy()[0]
        state, reward, done, _ = env.step(action)
        # done = True if env.day>=env.max_step-5 else False
        if done:
            annualized_reward=(1 + np.mean(env.new_portfolio_return_list)) ** 52 - 1
            profit=(env.asset_memory[-1]-env.asset_memory[0])/env.asset_memory[0]
            print(f'annualized_reward: {annualized_reward}')
            print(f'last day asset: {env.asset_memory[-1]}')
            tmp_dict={}
            #saving action 
            dd_action=pd.DataFrame(np.array(env.actions_memory),index = env.date_memory, columns = np.insert(env.tic,0,0)) #remove cash : np.delete(np.array(self.actions_memory),0,axis=1)
            dd_action['Asset'] = np.array(env.asset_memory)
            portfolio = dd_action[dd_action.columns[(dd_action != 0).any()]]
            pricedd = env.priceData
            pricedd.rename({'Stock_ID':'stockno'},axis=1,inplace=True)
            price_df =pricedd.loc['2019-01-04':]
            price_df_change=price_df.pivot_table(columns='stockno',values='change',index=price_df.index)
            price_df_close=price_df.pivot_table(columns='stockno',values='close',index=price_df.index)
            tmp_dict['allocation_final_return'] = int(dd_action['Asset'][-1])
            dd_action=dd_action.drop(['Asset',0], axis = 1)
            tic  = dd_action.columns
            inter_dateRange=dd_action.index
            start_date = pd.Timestamp(inter_dateRange[0]) # - pd.DateOffset(days=7)
            start_date =price_df_close.iloc[0].name  if start_date < price_df_close.iloc[0].name else start_date
            end_date = pd.Timestamp(inter_dateRange[-1])  #+ pd.DateOffset(days=7)
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
            with open('./data/latest_test_live_update.pkl', 'wb') as handle:
                pickle.dump(tmp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Wait for one week (in seconds)
            time.sleep(7 * 24 * 60 * 60)
            # Reload the data
            env.reload_data(args.encoder_dataPath, args.data_trend_path, args.latest_priceBook_path,env.day,env.model_type)
            break


    return annualized_reward, episode_step, 

class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True):
        self.env = gym.make(env) if isinstance(env, str) else env
        super().__init__(self.env)

        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_return) = get_gym_env_info(self.env, if_print)

    def reset(self) -> np.ndarray:
        state = self.env.reset()
        return state.astype(np.float32)

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
        state, reward, done, info_dict = self.env.step(
            action * self.action_max)
        return state.astype(np.float32), reward, done, info_dict


def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
    assert isinstance(env, gym.Env)

    env_name = getattr(env, 'env_name', None)
    env_name = env.unwrapped.spec.id if env_name is None else None

    state_shape = env.observation_space.shape
    # sometimes state_dim is a list
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape

    target_return = getattr(env.spec, 'reward_threshold', 2 ** 16)

    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    # make sure it is continuous action space
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
        assert not any(env.action_space.high + env.action_space.low)
    else:
        raise RuntimeError(
            '| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    print(f"\n| env_name:  {env_name}, action if_discrete: {if_discrete}"
          f"\n| state_dim: {state_dim:4}, action_dim: {action_dim}, action_max: {action_max}"
          f"\n| max_step:  {max_step:4}, target_return: {target_return}") if if_print else None
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_return
