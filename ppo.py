import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn as nn
from argparses import device, args
from memory import Transition
import cv2
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import   BatchSampler, SubsetRandomSampler


class PolicyNet(nn.Module):
    def __init__(self,state_dim=54, hidden_dim=256, action_dim=4):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        self.flat = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.Tanh())
        self.drop = nn.Dropout(0.5)
        self.fc3 = nn.Sequential(nn.Linear(512, 13))
    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.flat(output)
        output = self.drop(output)
        output = self.fc1(output)
        return nn.functional.softmax(self.fc3(output), dim=1)

class ValueNet(nn.Module):
    def __init__(self,state_dim=54, hidden_dim=256):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512), nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, input):
        return self.net(input)


class PPO:
    def __init__(self, state_dim=3200, hidden_dim=256, action_dim=4, actor_lr=3e-4, critic_lr=1e-3,
                 lmbda=0.95, gamma=0.99, epochs=10, eps=0.2, device=device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.batch_size=args.ppo_batch_size
        self.mini_batch_size=64
        
        self.action_sizes = [4]
        self.action_space=sum(self.action_sizes)
        print("action_space:",self.action_space)

        if args.ppo_actor_model_path and os.path.exists(args.ppo_actor_model_path):
            actor_dict = torch.load(args.ppo_actor_model_path)
            self.actor.load_state_dict(actor_dict["net"])
            self.actor_optimizer.load_state_dict(actor_dict["optimizer"])
            print(f"Model loaded from {args.ppo_actor_model_path}")
       #ppo_critic_model_path 
        if args.ppo_critic_model_path and os.path.exists(args.ppo_critic_model_path):
            critic_dict = torch.load(args.ppo_critic_model_path)
            self.critic.load_state_dict(critic_dict["net"])
            self.critic_optimizer.load_state_dict(critic_dict["optimizer"])
            print(f"Model loaded from {args.ppo_critic_model_path}")
    
    def select_action(self, state):
        probs = self.actor(state)
        probs = torch.nn.functional.softmax(probs, dim=1)  # 使用softmax函数将值归一化
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    

    def compute_advantage(self,gamma, lmbda, td_delta):
        """
        计算优势函数
        :param gamma: 折扣因子
        :param lmbda: GAE参数
        :param td_delta: TD误差
        :return: 优势函数
        """
        advantage = 0
        advantage_list = []
        print("length of td_delta: ", len(td_delta))
        # 确保td_delta不是空的
        if len(td_delta) == 0:
            print("td_delta is empty")
            advantage_list.append(0)
            return torch.tensor([advantage], dtype=torch.float32)
        
        # # 将优势函数列表反转
        # advantage_list.reverse()
        for delta in td_delta:
            # 计算优势函数
            advantage = delta + gamma * lmbda * advantage
            # 将优势函数添加到优势函数列表中
            advantage_list.append(advantage)
        advantage_tensor = torch.tensor(advantage_list, dtype=torch.float)
        return advantage_tensor
    
    def update(self, replay_memory):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*replay_memory)
        states = torch.cat(state_batch, dim=0).to(device)
        actions = torch.tensor(action_batch).view(-1, 1).to(device)
        rewards = torch.tensor(reward_batch).view(-1, 1).to(device)
        dones = torch.tensor(terminal_batch).view(-1, 1).int().to(device)
        next_states = torch.cat(next_state_batch, dim=0).to(device)

        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)
            advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(device)
            old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                log_probs = torch.log(self.actor(states[index]).gather(1, actions[index]))
                ratio = torch.exp(log_probs - old_log_probs[index])
                surr1 = ratio * advantage[index]
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage[index]  # 截断
                actor_loss = torch.mean(-torch.min(surr1, surr2))
                critic_loss = torch.mean(
                    nn.functional.mse_loss(self.critic(states[index]), td_target[index].detach()))
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        del replay_memory


        # torch.cuda.empty_cache()

        #  # 如果epsilon大于epsilon_min，则将epsilon乘以epsilon_decay
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay


    def save_model(self,a_path=args.ppo_actor_model_path,c_path=args.ppo_critic_model_path):
         # 保存actor模型
        actor_dict = {"net": self.actor.state_dict(), "optimizer": self.actor_optimizer.state_dict()}
        # 保存critic模型
        critic_dict = {"net": self.critic.state_dict(), "optimizer": self.critic_optimizer.state_dict()}
        # 保存actor模型
        torch.save(actor_dict, "{}".format(a_path))
        # 保存critic模型
        torch.save(critic_dict, "{}".format(c_path))
        print(f"Model saved to {a_path} and {c_path}")























