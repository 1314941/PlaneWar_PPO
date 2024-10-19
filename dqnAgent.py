import os

import cv2
import numpy as np
import torch
from torch import optim, nn

from argparses import device, args
from memory import Transition
from net_actor import NetDQN
from globalInfo import singleton
import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if end_time - start_time > 1:
            print(f"函数 {func.__name__} 花费了 {end_time - start_time} 秒来执行")
        return result
    return wrapper




class DQNAgent:
    @timer_decorator
    def __init__(self):
        torch.backends.cudnn.enabled = False

        self.action_sizes = [4,50,2]
        self.action_space=sum(self.action_sizes)
        print("action_space:",self.action_space)
        
        self.device = device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.learning_rate = args.learning_rate

        self.steps_done = 0
        self.target_update = args.target_update

        self.policy_net = NetDQN().to(self.device)
        self.target_net = NetDQN().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        if args.dqn_model_path and os.path.exists(args.dqn_model_path):
            self.policy_net.load_state_dict(torch.load(args.dqn_model_path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded from {args.dqn_model_path}")

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def select_action(self, state):
        # 如果随机数小于等于epsilon，则随机选择动作
        if np.random.rand() <= self.epsilon:
            print("random action")
            # with open("log.txt", "a") as f:
            #     f.write("random action")
            return [np.random.randint(size) for size in self.action_sizes]
        # 对输入的图像进行预处理，并增加一个维度
        tmp_state_640_640 = self.preprocess_image(state).unsqueeze(0)
        # 将策略网络设置为评估模式
        self.policy_net.eval()
        # 不计算梯度
        with torch.no_grad():
            # 获取当前状态下每个动作的Q值
            q_values = self.policy_net(tmp_state_640_640)
        # 返回Q值最大的动作
        return [np.argmax(q.detach().cpu().numpy()) for q in q_values]

    def preprocess_image(self, image, target_size=(54, 54)):
        # 调整图像大小
        resized_image = cv2.resize(image, target_size)
        # 转换为张量并调整维度顺序 [height, width, channels] -> [channels, height, width]
        tensor_image = torch.from_numpy(resized_image).float().permute(2, 0, 1)
    # 将tensor_image转换为指定设备上的张量
        return tensor_image.to(device)

    def replay(self,transitions):
        # print("replaying")
        
        # transitions = globalInfo.random_batch_size_memory_dqn()
        # 将transitions中的元素按照顺序打包成Transition对象
        print("length of transitions:",len(transitions))
        batch = Transition(*zip(*transitions))

        # 将 batch 转换为张量，并移动到设备上
        batch_state = torch.stack([self.preprocess_image(state) for state in batch.state]).to(device)
        batch_action = torch.LongTensor(batch.action).to(self.device)
        batch_reward = torch.FloatTensor(batch.reward).to(self.device)
        batch_next_state = torch.stack([self.preprocess_image(state) for state in batch.next_state]).to(device)
        batch_done = torch.FloatTensor(batch.done).to(self.device)

        # 计算当前状态的 Q 值
        state_action_values = self.policy_net(batch_state)

        # 计算每个动作类别的 Q 值
        direction_action_q,distance_action_q,attack_q = state_action_values

        # 选择执行的动作的 Q 值
      # 根据batch_action中的索引，从move_action_q、angle_q、info_action_q、attack_action_q、action_type_q、arg1_q、arg2_q、arg3_q中取出对应的Q值，并相加
        state_action_q_values = direction_action_q.gather(1, batch_action[:, 0].unsqueeze(1)) + \
                                distance_action_q.gather(1, batch_action[:, 1].unsqueeze(1)) + \
                                attack_q.gather(1, batch_action[:, 2].unsqueeze(1))


        # 计算下一个状态的 Q 值
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        non_final_mask = (batch_done == 0)
        non_final_next_states = batch_next_state[non_final_mask]
        if non_final_next_states.size(0) > 0:
            next_state_action_values = self.target_net(non_final_next_states)
            next_dir_action_q,next_distance_action_q,next_attack_q = next_state_action_values
  # 计算下一个状态的价值
            next_state_values[non_final_mask] = torch.max(next_dir_action_q, 1)[0]+\
                                                torch.max(next_distance_action_q, 1)[0]+\
                                                torch.max(next_attack_q, 1)[0]


        # 计算期望的 Q 值
        # 计算预期状态-动作值
        # batch_reward：当前状态下的奖励
        # self.gamma：折扣因子
        # next_state_values：下一个状态下的值
        # batch_done：是否完成
        expected_state_action_values = batch_reward + self.gamma * next_state_values * (1 - batch_done)

        # 计算损失
        loss = self.criterion(state_action_q_values, expected_state_action_values.unsqueeze(1))

        print("loss", loss)

        # 优化模型
     # 将优化器的梯度置零
        self.optimizer.zero_grad()
     # 反向传播计算梯度
        loss.backward()
     # 更新优化器的参数
        self.optimizer.step()

     # 如果epsilon大于epsilon_min，则将epsilon乘以epsilon_decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

     # 如果steps_done是target_update的倍数，则将policy_net的参数加载到target_net中
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

     # steps_done加1
        self.steps_done += 1

