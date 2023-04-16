import os, math
import numpy.random as rd
from copy import deepcopy
from tutorial_net import *
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter



class AgentBase:
    def __init__(self):
        self.state = None
        self.device = None
        self.action_dim = None
        self.if_on_policy = None
        self.explore_noise = None
        self.trajectory_list = None
        self.if_use_gae = None
        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, _if_per_or_gae=False, gpu_id=0):
        # explict call self.init() for multiprocessing
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.action_dim = action_dim

        self.cri = self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        # self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate, weight_decay=1e-5)
        # self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate, weight_decay=1e-5) if self.ClassAct else self.cri
        # 0401 exp to using RMSpros to improve training gradient converge 
        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate, weight_decay=1e-5)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate, weight_decay=1e-5) if self.ClassAct else self.cri

        del self.ClassCri, self.ClassAct

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(states)[0]
        action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.detach().cpu().numpy()

    def explore_env(self, env, target_step, reward_scale, gamma) -> tuple:
        state = self.state

        trajectory_list = list()
        for _ in range(target_step):
            action = self.select_action(state)
            next_s, reward, done, _ = env.step(action)
            trajectory_list.append((state, (reward, done, *action)))

            state = env.reset() if done else next_s
        self.state = state

        '''convert list to array'''
        trajectory_list = list(map(list, zip(*trajectory_list)))  # 2D-list transpose
        ary_state = np.stack(trajectory_list[0])
        ary_other = np.stack(trajectory_list[1])
        ary_other[:, 0] = ary_other[:, 0] * reward_scale  # ary_reward
        ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ary_mask = (1.0 - ary_done) * gamma
        return ary_state, ary_other

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]
        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None 


class AgentDQN(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy
        self.if_use_cri_target = True
        self.ClassCri = QNet

    def select_action(self, state) -> int:  # for discrete action space
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_int = rd.randint(self.action_dim)  # choosing action randomly
        else:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            action = self.act(states)[0]
            a_int = action.argmax(dim=0).detach().cpu().numpy()
        return a_int

    def explore_env(self, env, target_step, reward_scale, gamma) -> tuple:
        state = self.state

        trajectory_list = list()
        for _ in range(target_step):
            action = self.select_action(state)  # assert isinstance(action, int)
            next_s, reward, done, _ = env.step(action)
            trajectory_list.append((state, (reward, done, action)))

            state = env.reset() if done else next_s
        self.state = state

        '''convert list to array'''
        trajectory_list = list(map(list, zip(*trajectory_list)))  # 2D-list transpose
        ary_state = np.stack(trajectory_list[0])
        ary_other = np.stack(trajectory_list[1])
        ary_other[:, 0] = ary_other[:, 0] * reward_scale  # ary_reward
        ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ary_mask = (1.0 - ary_done) * gamma
        return ary_state, ary_other

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()
        obj_critic = q_value = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, q_value = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)
        return obj_critic.item(), q_value.mean().item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value


class AgentDoubleDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.softMax = torch.nn.Softmax(dim=1)
        self.ClassCri = QNetTwin

    def select_action(self, state) -> int:  # for discrete action space
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions = self.act(states)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_prob = self.softMax(actions)[0].detach().cpu().numpy()
            a_int = rd.choice(self.action_dim, p=a_prob)  # choose action according to Q value
        else:
            action = actions[0]
            a_int = action.argmax(dim=0).detach().cpu().numpy()
        return a_int

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1


class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.1  # explore noise of action
        self.if_use_cri_target = self.if_use_act_target = True
        self.ClassCri = Critic
        self.ClassAct = Actor

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()
            self.optim_update(self.act_optim, obj_actor)
            self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_actor.item(), obj_critic.item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state


class AgentTD3(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.1  # standard deviation of exploration noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency
        self.if_use_cri_target = self.if_use_act_target = True
        self.ClassCri = CriticTwin
        self.ClassAct = Actor

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, action_pg).mean()  # use cri_target instead of cri for stable training
            self.optim_update(self.act_optim, obj_actor)
            if update_c % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state


class AgentSAC(AgentBase):
    def __init__(self):
        super().__init__()
        self.if_use_cri_target = True
        self.ClassCri = CriticTwin
        self.ClassAct = ActorSAC

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act.get_action(states)[0]
        return action.detach().cpu().numpy()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()
        log_alpha = self.act.log_alpha
        obj_critic = obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size, log_alpha.exp())
            self.optim_update(self.cri_optim, obj_critic)

            action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_actor = (-torch.min(*self.cri_target.get_q1_q2(state, action_pg)).mean()
                         + logprob.mean() * log_alpha.exp().detach()
                         + self.act.get_obj_alpha(logprob))
            self.optim_update(self.act_optim, obj_actor)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item(), log_alpha.item()

    def get_obj_critic(self, buffer, batch_size, alpha) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a, next_logprob = self.act.get_action_logprob(next_s)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
            q_label = reward + mask * (next_q + next_logprob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)  # twin critics
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, state

class AgentPPO(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassCri = CriticAdv
        self.ClassAct = ActorPPO
        self.criterion = torch.nn.SmoothL1Loss()
        self.if_on_policy = True
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.1  # could be 0.01~0.05 , 0822 set as 0.00 to increase exploration 
        self.lambda_gae_adv = 0.95  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        self.writerGradient = SummaryWriter('./dashboard/log/gradient')  

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, gpu_id=0, env_num=1):
        super().init(net_dim, state_dim, action_dim, learning_rate, if_use_gae, gpu_id)
        self.trajectory_list = [list() for _ in range(env_num)]
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
         # Initialize the LR schedulers
        self.lr_scheduler_act = LambdaLR(self.act_optim, lr_lambda=lambda epoch: 0.99 ** epoch)
        self.lr_scheduler_cri = LambdaLR(self.cri_optim, lr_lambda=lambda epoch: 0.99 ** epoch)


    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(states)
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy() 


    def explore_env(self, env, target_step, reward_scale, gamma,action_repeats):
        state = self.state
        trajectory_temp = list()
        last_done = 0
        for i in range(target_step):
            action, noise = self.select_action(state)
            reward_total= 0 
            for x in range(action_repeats):
                next_state, reward, done, _ = env.step(np.tanh(action))
                reward_total+=reward
            trajectory_temp.append((state, reward_total, done, action, noise))
            if done:
                state = env.reset()
                last_done = i
            else:
                state = next_state
        self.state = state

        '''splice list'''
        trajectory_list = self.trajectory_list[0] + trajectory_temp[:last_done + 1]
        self.trajectory_list[0] = trajectory_temp[last_done:]

        '''convert list to array'''
        trajectory_list = list(map(list, zip(*trajectory_list)))  # 2D-list transpose
        ary_state = np.array(trajectory_list[0],dtype=np.float32)
        ary_reward = np.array(trajectory_list[1], dtype=np.float32) * reward_scale
        ary_mask = (1 - np.array(trajectory_list[2], dtype=np.float32)) * gamma
        ary_action = np.array(trajectory_list[3], dtype=np.float32)
        ary_noise = np.array(trajectory_list[4], dtype=np.float32)
        return ary_state, ary_action, ary_noise, ary_reward, ary_mask

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        clip_value = 2  # Adjust this value as needed
        cri_clip_value = 2  # Adjust this value as needed
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_action, buf_noise = [torch.as_tensor(ary, device=self.device) for ary in buffer[:3]]
            # (ary_state, ary_action, ary_noise, ary_reward, ary_mask) = buffer

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 12  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            ary_r_sum, ary_advantage = self.get_reward_sum(buf_len,
                                                           ary_reward=buffer[3],
                                                           ary_mask=buffer[4],
                                                           ary_value=buf_value.cpu().numpy())  # detach()
            buf_r_sum, buf_advantage = [torch.as_tensor(ary, device=self.device)
                                        for ary in (ary_r_sum, ary_advantage)]
            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5) 
            del buf_noise, buffer[:], ary_r_sum, ary_advantage

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = logprob = None
        training_step = 0
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            # Apply gradient clipping to the actor network, new improvement added by Evan 03252023
            torch.nn.utils.clip_grad_norm_(self.act.parameters(), clip_value)
            self.optim_update(self.act_optim, obj_actor)
            # Record gradients for the actor network
            for name, param in self.act.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    if not math.isnan(grad_norm):  # Ignore NaN values
                        self.writerGradient.add_scalar(f"Actor_gradient_norm/{name}", grad_norm, training_step)

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

            # Apply gradient clipping to the critic network, new improvement added by Evan 03252023
            torch.nn.utils.clip_grad_norm_(self.cri.parameters(), cri_clip_value)
            self.optim_update(self.cri_optim, obj_critic)
            # Record gradients for the critic network
            for name, param in self.cri.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    if not math.isnan(grad_norm):  # Ignore NaN values
                        self.writerGradient.add_scalar(f"Critic_gradient_norm/{name}", grad_norm, training_step)

            
            # Update the LR schedulers
            self.lr_scheduler_act.step()
            self.lr_scheduler_cri.step()
            assert 0 <= soft_update_tau <= 1, "soft_update_tau must be between 0 and 1"
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None
            training_step += 1

        return obj_critic.item(), obj_actor.item(), logprob.mean().item()  # logging_tuple

    @staticmethod
    def get_reward_sum_raw(buf_len, ary_reward, ary_mask, ary_value) -> (torch.Tensor, torch.Tensor):
        ary_r_sum = np.empty(buf_len, dtype=np.float32)  # reward sum
        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            ary_r_sum[i] = ary_reward[i] + ary_mask[i] * pre_r_sum
            pre_r_sum = ary_r_sum[i]
        ary_advantage = ary_r_sum - (ary_mask * ary_value[:, 0])
        return ary_r_sum, ary_advantage

    def get_reward_sum_gae(self, buf_len, ary_reward, ary_mask, ary_value) -> (torch.Tensor, torch.Tensor):
        ary_r_sum = np.empty(buf_len, dtype=np.float32)  # old policy value
        ary_advantage = np.empty(buf_len, dtype=np.float32)  # advantage value
        pre_r_sum = 0
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            ary_r_sum[i] = ary_reward[i] + ary_mask[i] * pre_r_sum
            pre_r_sum = ary_r_sum[i]
            ary_advantage[i] = ary_reward[i] + ary_mask[i] * (pre_advantage - ary_value[i])  # fix a bug here
            pre_advantage = ary_value[i] + ary_advantage[i] * self.lambda_gae_adv
        return ary_r_sum, ary_advantage


class AgentPPONEW(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassCri = CriticAdvSub #CriticAdv  >>>>>> changed by evan 0924
        self.ClassAct = ActorPPO
        self.criterion = torch.nn.SmoothL1Loss()
        self.if_on_policy = True
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.00  # could be 0.01~0.05 , 0822 set as 0.00 to increase exploration 
        self.lambda_gae_adv = 0.99  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None #self.get_reward_sum_gae #if if_use_gae else self.get_reward_sum_raw

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=True, gpu_id=0, env_num=1):
        super().init(net_dim, state_dim, action_dim, learning_rate, if_use_gae, gpu_id)
        self.trajectory_list = [list() for _ in range(env_num)]
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
       
    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(states)
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy() 


    def explore_env(self, env, target_step, reward_scale, gamma):
        
        state = self.state
        trajectory_temp = list()
        last_done = 0
        for i in range(target_step):
            action, noise = self.select_action(state)
            next_state, reward, done, _ = env.step(action)  # _ is trend  np.tanh(action)
            trajectory_temp.append((state, reward, done, action, noise, _)) # _ is trend
            if done:
                state = env.reset()
                last_done = i
            else:
                state = next_state
        self.state = state

        '''splice list'''
        trajectory_list = self.trajectory_list[0] + trajectory_temp[:last_done + 1]
        self.trajectory_list[0] = trajectory_temp[last_done:]

        '''convert list to array'''
        trajectory_list = list(map(list, zip(*trajectory_list)))  # 2D-list transpose
        ary_state = np.array(trajectory_list[0],dtype=np.float32)
        ary_reward = np.array(trajectory_list[1], dtype=np.float32) * reward_scale
        ary_mask = (1 - np.array(trajectory_list[2], dtype=np.float32)) * gamma
        ary_action = np.array(trajectory_list[3], dtype=np.float32)
        ary_noise = np.array(trajectory_list[4], dtype=np.float32)
        #ary_trend_val_ary=[np.zeros(45) if len(x) !=45 else x for x in trajectory_list[5]] #pd.DataFrame(trajectory_list[5]).where(pd.DataFrame(trajectory_list[5]).astype(bool),.0).to_numpy(dtype=np.float32)
        ary_trend_val_ary=[np.array(.0,dtype=np.float32) if x == {} else x for x in trajectory_list[5]]
        ary_trend_val = np.array(ary_trend_val_ary, dtype=np.float32)
        return ary_state, ary_action, ary_noise, ary_reward, ary_mask, ary_trend_val#.reshape(ary_noise.shape[0])

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_action, buf_noise = [torch.as_tensor(ary, device=self.device) for ary in buffer[:3]]
            # (ary_state, ary_action, ary_noise, ary_reward, ary_mask) = buffer

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_action_noCash = buf_action[:,1:] ## >>>>>> changed by evan 0924
            buf_value = [self.cri_target(buf_action_noCash[i:i + bs]) for i in range(0, buf_len, bs)] ## >>>>>> changed by evan 0924
           # buf_value = [sum([self.cri_target(x.reshape(1)) for x in buf_action_noCash[i:i + bs]]) for i in range(0, buf_len, bs)] ## >>>>>> changed by evan 1004
            #buf_value = [sum([self.cri_target(x.reshape(1)) for x in i]) for i in buf_action_noCash[c:c+bs] for c in range(0, buf_len, bs)]
            # buf_value = []
            # for c in range(0, buf_len, bs):
            #     for i in buf_action_noCash[c:c+bs]:
            #         buf_value.append(sum([self.cri_target(x.reshape(1)) for x in i]))


            # buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            ary_r_sum, ary_advantage = self.get_reward_sum(buf_len,
                                                           ary_reward=buffer[5], # change by evan 0903, original is buffer[3]
                                                           ary_mask=buffer[4],
                                                           ary_value = buf_value.cpu().numpy().reshape(buf_value.shape[0]))
                                                           #ary_value=buf_value.cpu().numpy())  # detach()
            buf_r_sum, buf_advantage = [torch.as_tensor(ary, device=self.device)
                                        for ary in (ary_r_sum, ary_advantage)]
            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5) # normmalize advantage
            del buf_noise, buffer[:], ary_r_sum, ary_advantage

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = logprob = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            action_noCash = action[:,1:]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(action_noCash).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            # value = []
            # for action_day in action_noCash:
            #     value.append(sum([self.cri(i.reshape(1)) for i in action_day]))
            # value = torch.cat(value, dim=0)
            ## >>>>>> changed by evan 0924, original should be "state" self.cri(state).squeeze(1)
            obj_critic = self.criterion(value, r_sum) #/ (r_sum.std() + 1e-6) changebyEvan 2022
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        a_logstd = getattr(self.act, "a_logstd", torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_logstd.mean().item()  # logging_tuple

    @staticmethod
    def get_reward_sum_raw(buf_len, ary_reward, ary_mask, ary_value) -> (torch.Tensor, torch.Tensor):
        ary_r_sum = np.empty(buf_len, dtype=np.float32)  # reward sum
        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            ary_r_sum[i] = ary_reward[i] + ary_mask[i] * pre_r_sum
            pre_r_sum = ary_r_sum[i]
        ary_advantage =ary_r_sum - (ary_mask * ary_value) # ary_r_sum - (ary_mask * ary_value[:, 0])
        return ary_r_sum, ary_advantage
   
    def get_reward_sum_gae(self, buf_len, ary_reward, ary_mask, ary_value) -> (torch.Tensor, torch.Tensor):
        ary_r_sum = np.empty(buf_len, dtype=np.float32)  # old policy value
        ary_advantage = np.empty(buf_len, dtype=np.float32)  # advantage value
        pre_r_sum = 0
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            ary_r_sum[i] = ary_reward[i] + ary_mask[i] * pre_r_sum
            pre_r_sum = ary_r_sum[i]
            ary_advantage[i] = ary_reward[i] + ary_mask[i] * (pre_advantage - ary_value[i])  # fix a bug here
            pre_advantage = ary_value[i] + ary_advantage[i] * self.lambda_gae_adv
        return ary_r_sum, ary_advantage


class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.ClassAct = ActorDiscretePPO

    def explore_env(self, env, target_step, reward_scale, gamma):
        state = self.state

        trajectory_temp = list()
        last_done = 0
        for i in range(target_step):
            # action, noise = self.select_action(state)
            # next_state, reward, done, _ = env.step(np.tanh(action))
            action, a_prob = self.select_action(state)  # different from `action, noise`
            a_int = int(action)  # different
            next_state, reward, done, _ = env.step(a_int)  # different from `np.tanh(action)`
            trajectory_temp.append((state, reward, done, a_int, a_prob))
            if done:
                state = env.reset()
                last_done = i
            else:
                state = next_state
        self.state = state

        '''splice list'''
        srdan_list = self.trajectory_list[0] + trajectory_temp[:last_done + 1]
        self.trajectory_list[0] = trajectory_temp[last_done:]

        '''convert list to array'''
        srdan_list = list(map(list, zip(*srdan_list)))  # 2D-list transpose
        ary_state = np.array(srdan_list[0])
        ary_reward = np.array(srdan_list[1], dtype=np.float32) * reward_scale
        ary_mask = (1.0 - np.array(srdan_list[2], dtype=np.float32)) * gamma
        ary_action = np.array(srdan_list[3], dtype=np.uint8)  # different from `np.float32`
        ary_noise = np.array(srdan_list[4], dtype=np.float32)
        return ary_state, ary_action, ary_noise, ary_reward, ary_mask


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        other_dim = 1 + 1 + self.action_dim
        self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)

        if isinstance(state_dim, int):  # state is pixel
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(state_dim, tuple):
            self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('state_dim')

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def save_or_load_history(self, cwd, if_save, buffer_id=0):
        save_path = f"{cwd}/replay_{buffer_id}.npz"

        if if_save:
            self.update_now_len()
            state_dim = self.buf_state.shape[1]
            other_dim = self.buf_other.shape[1]
            buf_state = np.empty((self.max_len, state_dim), dtype=np.float16)  # sometimes np.uint8
            buf_other = np.empty((self.max_len, other_dim), dtype=np.float16)

            temp_len = self.max_len - self.now_len
            buf_state[0:temp_len] = self.buf_state[self.now_len:self.max_len].detach().cpu().numpy()
            buf_other[0:temp_len] = self.buf_other[self.now_len:self.max_len].detach().cpu().numpy()

            buf_state[temp_len:] = self.buf_state[:self.now_len].detach().cpu().numpy()
            buf_other[temp_len:] = self.buf_other[:self.now_len].detach().cpu().numpy()

            np.savez_compressed(save_path, buf_state=buf_state, buf_other=buf_other)
            print(f"| ReplayBuffer save in: {save_path}")
        elif os.path.isfile(save_path):
            buf_dict = np.load(save_path)
            buf_state = buf_dict['buf_state']
            buf_other = buf_dict['buf_other']

            buf_state = torch.as_tensor(buf_state, dtype=torch.float32, device=self.device)
            buf_other = torch.as_tensor(buf_other, dtype=torch.float32, device=self.device)
            self.extend_buffer(buf_state, buf_other)
            self.update_now_len()
            print(f"| ReplayBuffer load: {save_path}")