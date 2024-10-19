import torch
import torch.nn as nn
import torch.nn.functional as F


# Actor 网络
class NetDQN(nn.Module):
    def __init__(self):
        super(NetDQN, self).__init__()
        #想改，不知道咋改
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)

        conv_output_size = self._get_conv_output_size(54, 54)
        self.fc = nn.Linear(conv_output_size, 256)

        self.fc1 = nn.Linear(256, 256)
        self.fc_direction = nn.Linear(256, 4)  # move_action_list Q-values
        self.fc_distance=nn.Linear(256,50)
        self.fc_attack=nn.Linear(256, 2)
        self._initialize_weights()

    def _get_conv_output_size(self, height, width):
        # 创建一个假的输入，大小为1x3xheightxwidth
        dummy_input = torch.zeros(1, 3, height, width)
        # 不计算梯度
        with torch.no_grad():
            # 将输入通过第一个卷积层
            x = F.relu(self.conv1(dummy_input))
            # 将输出通过第二个卷积层
            x = F.relu(self.conv2(x))
        # 将输出展平为一维向量，并返回其大小
        return x.view(x.size(0), -1).size(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = F.relu(self.fc1(x))

        direction_action_q = self.fc_direction(x)
        distance_action_q = self.fc_distance(x)
        attack_action_q = self.fc_attack(x)

        return direction_action_q,distance_action_q,attack_action_q
