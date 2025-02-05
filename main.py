from configparser import ConfigParser
from argparse import ArgumentParser

import torch
import gym
import numpy as np
import os

from agents.ppo import PPO
from agents.sac_advantage import SAC
from agents.ddpg import DDPG

from utils.utils import make_transition, Dict, RunningMeanStd


parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default = 'Humanoid-v4', help = "'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
parser.add_argument("--algo", type=str, default = 'sac', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=True, help="(default: False)")
parser.add_argument('--epochs', type=int, default=50000, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=bool, default=True, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
parser.add_argument("--reward_scaling", type=float, default = 0.1, help = 'reward scaling(default : 0.1)')
parser.add_argument('--rendering_periodicity', type=int, default=200, help="(default: 100)")
args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser,args.algo)


run_name = 'test'
i_gpu = 1


os.makedirs('./model_weights/{}/'.format(run_name), exist_ok=True)
os.makedirs('./videos/{}/'.format(run_name), exist_ok=True)



device = torch.device( 'cuda:{}'.format(i_gpu) if torch.cuda.is_available() else 'cpu')
if device.type != 'cpu':
    torch.cuda.set_device(i_gpu)
torch.set_float32_matmul_precision('high')
print('Using device:', device)


    
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('/home/python-user/board/RL_base/tensorboard/'+run_name)
    def tensorboard_writer(metrics_dict, training_step):
        for key in metrics_dict.keys():
            writer.add_scalar(key, metrics_dict[key], training_step)
else:
    writer = None
  
    
if args.render:   
    render_mode = 'rgb_array'#'human'
else:
    render_mode = None
    

class ReacherRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        #x_position = info['x_position']
        #y_position = info['y_position']
        x_velocity = info['x_velocity']
        y_velocity = info['y_velocity']            
        distance_from_origin = info['distance_from_origin']
        
        custom_velocity_reward = 2 * np.sqrt(x_velocity**2+y_velocity**2)
        if distance_from_origin > 5:
            custom_position_reward = - 0.1
        else:
            custom_position_reward = 0
        custom_reward = 1.0 * custom_velocity_reward + custom_position_reward
        total_reward = reward + custom_reward
        info['custom_velocity_reward'] = custom_velocity_reward
        info['velocity'] = np.sqrt(x_velocity**2+y_velocity**2)
        
        info['custom_position_reward'] = custom_position_reward
        return obs, total_reward, terminated, truncated, info 
    
def gen_env(render_mode=None):
    env = gym.make(args.env_name, max_episode_steps=1000, exclude_current_positions_from_observation= False, render_mode= render_mode)
    env = ReacherRewardWrapper(env)
    return env


env = gen_env(render_mode=None)










action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
state_rms = RunningMeanStd(state_dim)


if args.algo == 'ppo' :
    agent = PPO(writer, device, state_dim, action_dim, agent_args)
elif args.algo == 'sac' :
    agent = SAC(writer, device, state_dim, action_dim, agent_args)
elif args.algo == 'ddpg' :
    from utils.noise import OUNoise
    noise = OUNoise(action_dim,0)
    agent = DDPG(writer, device, state_dim, action_dim, agent_args, noise)

    
if (torch.cuda.is_available()) and (args.use_cuda):
    agent = agent.cuda()

if args.load != 'no':
    agent.load_state_dict(torch.load("./model_weights/"+args.load))
    
score_lst = []
state_lst = []

if agent_args.on_policy == True:
    score = 0.0
    (state_, info) = (env.reset())
    state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    for n_epi in range(args.epochs):
        for t in range(agent_args.traj_length):
            if args.render:    
                env.render()
            state_lst.append(state_)
            mu,sigma = agent.get_action(torch.from_numpy(state).float().to(device))
            dist = torch.distributions.Normal(mu,sigma[0])
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1,keepdim = True)
            next_state_, reward, done, trunc, info = env.step(action.cpu().numpy())
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            transition = make_transition(state,\
                                         action.cpu().numpy(),\
                                         np.array([reward*args.reward_scaling]),\
                                         next_state,\
                                         np.array([done]),\
                                         log_prob.detach().cpu().numpy()\
                                        )
            agent.put_data(transition) 
            score += reward
            if done or trunc:
                (state_, info) = (env.reset())
                state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                if args.tensorboard:
                    writer.add_scalar("score/score", score, n_epi)
                score = 0
            else:
                state = next_state
                state_ = next_state_

        agent.train_net(n_epi)
        state_rms.update(np.vstack(state_lst))
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score))
            score_lst = []
        if n_epi%args.save_interval==0 and n_epi!=0:
            torch.save(agent.state_dict(),'./model_weights/agent_'+str(n_epi))
            
else : # off policy 
    for n_epi in range(1,args.epochs):
        score = 0.0
        
        
        if  args.render:
            if ((n_epi +1) % args.rendering_periodicity == 0) and not env.render_mode:
                print('=================================================')               
                env = gen_env(render_mode=render_mode)
                
                video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env,'./videos/'+run_name+'/video_{0:09d}.mp4'.format(n_epi),enabled=True)
                state, reset_dict = env.reset()

            elif (n_epi) % args.rendering_periodicity == 0:
                print("Saved video.")
                video_recorder.close()
                video_recorder.enabled = False
                
                env.close()
                env = gen_env(render_mode=None)
                state, reset_dict = env.reset()
            else: 
                state, reset_dict = env.reset()
        else:
            state, reset_dict = env.reset() 
        
        
        
        
        
        state, info = env.reset()
        done = False
        n_steps = 0
        while not done:
            n_steps +=1
            if args.render and env.render_mode:
                env.unwrapped.render()
                video_recorder.capture_frame()
                env.render()   
                
            action, _ = agent.get_action(torch.from_numpy(state).float().to(device))
            action = action.cpu().detach().numpy()
            action = action.squeeze()
            next_state, reward, done, trunc, info = env.step(action)
            transition = make_transition(state,\
                                         action,\
                                         np.array([reward*args.reward_scaling]),\
                                         next_state,\
                                         np.array([done])\
                                        )
            agent.put_data(transition) 

            state = next_state

            score += reward
            if agent.data.data_idx > agent_args.learn_start_size: 
                #print('train on episode {}'.format(n_epi))
                agent.train_net(agent_args.batch_size, n_epi)
                
        score_lst.append(score)
        if args.tensorboard:
            writer.add_scalar("score/score", score, n_epi)
            writer.add_scalar("episode/velocity", info['velocity'], n_epi)
            writer.add_scalar("episode/velocity_reward", info['custom_velocity_reward'], n_epi)
            writer.add_scalar("episode/n_steps", n_steps, n_epi)
            
            
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, with {} steps".format(n_epi, score, n_steps))
            score_lst = []
        if n_epi%args.save_interval==0 and n_epi!=0:
            torch.save(agent.state_dict(),'./model_weights/'+run_name+'/agent_'+str(n_epi))
