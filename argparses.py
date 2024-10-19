# config.py
import argparse

import torch


# 移动坐标和滑动半径
direction_actions_detail = {
    1: {'action_name': '右移'},
    2: {'action_name': '左移'},
    3: {'action_name': '上移'},
    4: {'action_name': '下移'},
    0: {'action_name': '不动'}
}


# 无操作, 攻击，攻击小兵，攻击塔，回城，恢复，装备技能, 1技能，2技能，3技能,
attack_actions_detail = {
    1: {'action_name': '王炸！'}
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iphone_id', type=str, default='127.0.0.1:7555', help="iphone_id")
    parser.add_argument('--real_iphone', type=bool, default=False, help="real_iphone")
    parser.add_argument('--window_title', type=str, default='PlaneWar', help="window_title")
    parser.add_argument('--device_id', type=str, default='cuda:0', help="device_id")
    parser.add_argument('--memory_size', type=int, default=512, help="Replay memory size")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--epsilon', type=float, default=0.1, help="Initial exploration rate")
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help="Exploration rate decay")
    parser.add_argument('--epsilon_min', type=float, default=0.01, help="Minimum exploration rate")
    parser.add_argument('--model_name', type=str, default="PlaneWar", help="Path to the model to load")
    parser.add_argument('--ppo_actor_model_path', type=str, default="src/ppo/PlaneWar_actor.pt", help="Path to the model to load")
    parser.add_argument('--ppo_critic_model_path', type=str, default="src/ppo/PlaneWar_critic.pt", help="Path to the model to load")
    parser.add_argument('--log_dir', type=str, default="logs", help="Directory to store logs")
    parser.add_argument('--dqn_model_path', type=str, default="src/dqn/PlaneWar.pt", help="Path to the model to load")
    parser.add_argument('--num_episodes', type=int, default=10, help="Number of episodes to collect data")
    parser.add_argument('--target_update', type=int, default=10, help="Number of episodes to collect data")
    parser.add_argument('--eps_clip', type=float, default=0.2, help="PPO的epsilon裁剪")
    parser.add_argument('--ppo_epochs', type=int, default=4, help="PPO的更新周期")
    parser.add_argument('--ppo_batch_size', type=int, default=512, help="batch_size for ppo")
    return parser.parse_args()


# 解析参数并存储在全局变量中
args = get_args()

device = torch.device(args.device_id if torch.cuda.is_available() else 'cpu')

# 全局变量   内存泄漏！
# globalInfo = GlobalInfo(batch_size=args.batch_size, buffer_capacity=args.memory_size)
