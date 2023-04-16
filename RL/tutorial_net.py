import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class QNet(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state)  # q value


class QNetTwin(nn.Module):  # Double DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))  # q2 value

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


class Actor(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))

        # nn_dense = DenseNet(mid_dim // 2)
        # inp_dim = nn_dense.inp_dim  
        # out_dim = nn_dense.out_dim
        # self.net = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
        #                                  nn_dense,
        #                                  nn.Linear(out_dim, action_dim), )
                    
            
            
    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class ActorSAC(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_a_avg = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the average of action
        self.net_a_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the log_std of action
        self.num_logprob = -np.log(action_dim)
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.log_alpha = nn.Parameter(torch.zeros((1, action_dim)) - np.log(action_dim), requires_grad=True)

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)
        a_std = self.net_a_std(t_tmp).clamp(-16, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get_action_logprob(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)
        a_std_log = self.net_a_std(t_tmp).clamp(-16, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        action = a_avg + a_std * noise
        a_tan = action.tanh()

        logprob = -(a_std_log + self.log_sqrt_2pi + ((a_avg - action) / a_std).pow(2) * 0.5)
        logprob = logprob - (-a_tan.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        return a_tan, logprob.sum(1, keepdim=True)

    def get_obj_alpha(self, logprob):
        return -(self.log_alpha * (logprob - self.num_logprob).detach()).mean()


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
      
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.LeakyReLU(),
                            nn.Linear(mid_dim, mid_dim), nn.LeakyReLU(),
                            nn.Linear(mid_dim, mid_dim), nn.LeakyReLU(),
                            nn.Linear(mid_dim, action_dim), )

        # nn_dense = DenseNet(mid_dim // 2)
        # inp_dim = nn_dense.inp_dim  
        # out_dim = nn_dense.out_dim
        # self.net = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
        #                                  nn_dense,
        #                                  nn.Linear(out_dim, action_dim), )
                    
        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_logstd = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()


    def get_action(self, state):
        #a_avg = self.netLSTM(state)
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_logprob_entropy(self, state, action):
        #a_avg = self.netLSTM(state)
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        delta = noise.pow(2) * 0.5
        return -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # old_logprob

        ''' LSTM approach
        self.state_size = state_dim
        self.action_size = action_dim
        self.linear1 = nn.Linear(self.state_size, mid_dim)
        self.linear2 = nn.Linear(mid_dim, mid_dim*2)
        self.LSTM_layer_3 = nn.LSTM(mid_dim*2,mid_dim,1, batch_first=True)
        self.linear4 = nn.Linear(mid_dim,mid_dim)
        self.linear5 = nn.Linear(mid_dim,action_dim)
        self.mu = nn.Linear(32,self.action_size)  #256 linear2
        self.sigma = nn.Linear(32,self.action_size)
        '''

class ActorDiscretePPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state):
        return self.net(state)  # action_prob without softmax

    def get_action(self, state):
        a_prob = self.soft_max(self.net(state))
        # action = Categorical(a_prob).sample()
        samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
        action = samples_2d.reshape(state.size(0))
        return action, a_prob

    def get_logprob_entropy(self, state, a_int):
        a_prob = self.soft_max(self.net(state))
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, a_int, a_prob):
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int)


class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.LeakyReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.LeakyReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.LeakyReLU(),
                                 nn.Linear(mid_dim, 1))


        # nn_dense = DenseNet(mid_dim // 2)
        # inp_dim = nn_dense.inp_dim  
        # out_dim = nn_dense.out_dim
        # self.net = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
        #                                  nn_dense,
        #                                  nn.Linear(out_dim, action_dim), )

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # q value
        #return self.netK(torch.cat((state, action), dim=1))


class CriticAdv(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()
        # self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
        #                          nn.Linear(mid_dim, mid_dim), nn.ReLU(),
        #                          nn.Linear(mid_dim, mid_dim), nn.ReLU(),
        #                          nn.Linear(mid_dim, 1))

        self.net = nn.Sequential(
                    nn.Linear(state_dim, mid_dim),
                    nn.LeakyReLU(),
                    nn.Linear(mid_dim, mid_dim),
                    nn.LeakyReLU(),
                    nn.Linear(mid_dim, 1)
                )

        self.net.apply(self.weights_init)  # Apply the weight initialization

    def forward(self, state):
        return self.net(state)  # advantage value
        #return self.netLSTM(state)

    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            # Use Kaiming initialization
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            # Use Xavier initialization
            # nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)



class CriticAdvSub(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()

        ''' LSTM approach
        self.state_size = state_dim
        self.action_size = _action_dim
        self.linear1 = nn.Linear(self.state_size, mid_dim)
        self.linear2 = nn.Linear(mid_dim, mid_dim*2)
        self.LSTM_layer_3 = nn.LSTM(mid_dim*2,mid_dim,1, batch_first=True)
        self.linear4 = nn.Linear(mid_dim,mid_dim)
        self.linear5 = nn.Linear(mid_dim,1)
        '''

        self.net = nn.Sequential(nn.Linear(45, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))
                                 #change to input 45 and output 45 to evaluate each stock asset value, changed by Evan 10/04/2022


    def forward(self, state):
        # return torch.sum(self.net(state), dim=1)
        return self.net(state)  # advantage value

class CriticTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU())  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q2 value

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values



class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3  # x2.shape==(-1, lay_dim*4)