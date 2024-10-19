import itertools
import threading
import time

import torch

from globalInfo import GlobalInfo
from GameWrapper_ppo import timer_decorator


class Environment():
    @timer_decorator
    def __init__(self, android_controller, rewordUtil,gameWrapper,globalInfo):
        self.android_controller = android_controller
        self.rewordUtil = rewordUtil
        self.globalInfo = globalInfo
        self.gameWrapper = gameWrapper
        self.lock = threading.Lock()

        # 输出，方向 移动次数(一次固定距离)  0 不动 1-4 上下左右移动一次 类推
        # move_action_list = list(range(21))
        direction_list=list(range(4))
        distance_list=list(range(50))
        attack_action_list=list(range(2))

        # 计算每个列表的长度
        lengths = [
            len(direction_list),
            len(distance_list),
            len(attack_action_list)
        ]
        # 操作空间
        self.action_space_n = sum(lengths)

        print("动作空间:",self.action_space_n)
    
    @timer_decorator
    def step(self, action,s_pipe,a_pipe):
        direction = action
        if direction == 0:
            distance = 0
        else:
            #1 2 3 4 上 右 左 下 好别扭呀
            distance=5*int(direction/4+1)
            direction=direction%4+1
           
        
        # distance = action[1]
        # print("move_action:", str([direction, distance]))
        a_pipe.send({"direction": direction, "distance": distance})
        s_pipe.send("move")

        
        attack_action = 1
        if attack_action == 1:
            s_pipe.send("attack")


        # #等待游戏画面更新
        # time.sleep(0.01)

        reward, done, info = self.rewordUtil.get_reword((direction),s_pipe,a_pipe)
        s_pipe.send("refresh")

        next_state = self.android_controller.screenshot_window()
        while next_state is None or next_state.size == 0:
            time.sleep(0.01)
            next_state = self.android_controller.screenshot_window()
            continue

        return next_state, reward, done, info

