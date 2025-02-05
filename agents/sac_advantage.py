from networks.network import Actor, Critic
from utils.utils import ReplayBuffer, convert_to_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


use_advantage = True

class SAC(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(SAC,self).__init__()
        self.args = args
        self.actor = Actor(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, \
                           self.args.activation_function, self.args.last_activation, self.args.trainable_std)

        self.q_1 = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function,self.args.last_activation)
        self.q_2 = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function,self.args.last_activation)
        
        if use_advantage:
            self.vf = Critic(self.args.layer_num, state_dim, 1, self.args.hidden_dim, self.args.activation_function,self.args.last_activation)
            self.target_vf = Critic(self.args.layer_num, state_dim, 1, self.args.hidden_dim, self.args.activation_function,self.args.last_activation)
            self.soft_update(self.vf, self.target_vf, 1.)
        else:
            self.target_q_1 = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function, self.args.last_activation)
            self.target_q_2 = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function, self.args.last_activation)
        
            self.soft_update(self.q_1, self.target_q_1, 1.)
            self.soft_update(self.q_2, self.target_q_2, 1.)
        
        self.alpha = nn.Parameter(torch.tensor(self.args.alpha_init))
        
        self.data = ReplayBuffer(action_prob_exist = False, max_size = int(self.args.memory_size), state_dim = state_dim, num_action = action_dim)
        self.target_entropy = - torch.tensor(action_dim)

        self.q_1_optimizer = optim.Adam(self.q_1.parameters(), lr=self.args.q_lr)
        self.q_2_optimizer = optim.Adam(self.q_2.parameters(), lr=self.args.q_lr)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=self.args.alpha_lr)
        
        if use_advantage:
            self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.args.q_lr)
            
        
        self.device = device
        self.writer = writer
        
    def compile_models(self, compile_functions= True):
        if use_advantage:
            self.actor = torch.compile(self.actor, mode= "reduce-overhead")
            self.q_1 = torch.compile(self.q_1, mode= "reduce-overhead")
            self.q_2 = torch.compile(self.q_2, mode= "reduce-overhead")
            self.vf = torch.compile(self.vf, mode= "reduce-overhead")
            self.target_vf = torch.compile(self.target_vf, mode= "reduce-overhead")
        else:
            self.actor = torch.compile(self.actor, mode= "reduce-overhead")
            self.q_1 = torch.compile(self.q_1, mode= "reduce-overhead")
            self.q_2 = torch.compile(self.q_2, mode= "reduce-overhead")
            self.target_q_1 = torch.compile(self.target_q_1, mode= "reduce-overhead")
            self.target_q_2 = torch.compile(self.target_q_2, mode= "reduce-overhead")           
        
    def put_data(self,transition):
        self.data.put_data(transition)
        
    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)
    
    def get_action(self,state):
        mu,std = self.actor(state)
        dist = Normal(mu, std)
        u = dist.rsample()
        u_log_prob = dist.log_prob(u)
        a = torch.tanh(u)
        a_log_prob = u_log_prob - torch.log(1 - torch.square(a) +1e-3)
        return a, a_log_prob.sum(-1, keepdim=True)
    
    def q_update(self, Q, q_optimizer, states, actions, rewards, next_states, dones):
        ###target
        if use_advantage:
            with torch.no_grad():
                v_targets = self.target_vf(next_states)
                q_targets = rewards + self.args.gamma * v_targets * (1- dones)
        else:
            with torch.no_grad():
                next_actions, next_action_log_prob = self.get_action(next_states)
                q_1 = self.target_q_1(next_states, next_actions)
                q_2 = self.target_q_2(next_states, next_actions)
                q = torch.min(q_1,q_2)
                v_targets = (q - self.alpha * next_action_log_prob)
                q_targets = rewards + self.args.gamma * v_targets * (1 - dones) 
        
        q = Q(states, actions)
        loss = F.smooth_l1_loss(q, q_targets)
        q_optimizer.zero_grad()
        loss.backward()
        q_optimizer.step()
        q = q.cpu().mean().item()
        return loss, q
 
    def vf_update(self, states, actions, rewards, next_states, dones):

        # v function loss
        new_actions, new_action_log_prob = self.get_action(states)
        v_pred = self.vf(states)
        q_pred = torch.min(self.q_1(states, new_actions), self.q_2(states, new_actions))
        
        v_target = q_pred - self.alpha * new_action_log_prob
        loss = F.mse_loss(v_pred, v_target.detach())
        
        self.vf_optimizer.zero_grad()
        loss.backward()
        self.vf_optimizer.step()
        return loss, v_pred 
 
    def actor_update(self, states):
        now_actions, now_action_log_prob = self.get_action(states)
        q_1 = self.q_1(states, now_actions)
        q_2 = self.q_2(states, now_actions)
        q = torch.min(q_1, q_2)
        
        loss = (self.alpha.detach() * now_action_log_prob - q).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss,now_action_log_prob
    
    def alpha_update(self, now_action_log_prob):
        loss = (- self.alpha * (now_action_log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()    
        loss.backward()
        self.alpha_optimizer.step()
        return loss
    
    def train_net(self, batch_size, n_epi):
        data = self.data.sample(shuffle = True, batch_size = batch_size)
        states, actions, rewards, next_states, dones = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])

        ###q update
        q_1_loss, q_1 = self.q_update(self.q_1, self.q_1_optimizer, states, actions, rewards, next_states, dones)
        q_2_loss, q_2 = self.q_update(self.q_2, self.q_2_optimizer, states, actions, rewards, next_states, dones)

        ### actor update
        actor_loss,prob = self.actor_update(states)
        
        ###alpha update
        alpha_loss = self.alpha_update(prob)
        
        
        if use_advantage:
            vf_loss, vf = self.vf_update(states, actions, rewards, next_states, dones)
            self.soft_update(self.vf, self.target_vf, self.args.soft_update_rate)
        else:
            self.soft_update(self.q_1, self.target_q_1, self.args.soft_update_rate)
            self.soft_update(self.q_2, self.target_q_2, self.args.soft_update_rate)
        
        
        
        if self.writer != None:
            self.writer.add_scalar("loss/q_1", q_1_loss, n_epi)
            self.writer.add_scalar("loss/q_2", q_2_loss, n_epi)
            self.writer.add_scalar("loss/actor", actor_loss, n_epi)
            self.writer.add_scalar("loss/alpha", alpha_loss, n_epi)
            if use_advantage:
                self.writer.add_scalar("loss/vf", vf.mean().item(), n_epi)
                
            
            self.writer.add_scalar("trace/alpha", self.alpha.item(), n_epi)
            self.writer.add_scalar("trace/q_1_mean", q_1, n_epi)
            self.writer.add_scalar("trace/q_2_mean", q_2, n_epi)
            if use_advantage:
                self.writer.add_scalar("trace/vf_mean", vf.cpu().mean().item(), n_epi)
            


