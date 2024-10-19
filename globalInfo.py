import datetime
import json
import os
import threading
import psutil
from filelock import FileLock
import tracemalloc
from memory import ReplayMemory
from pympler import summary, muppy
from argparses import args

tracemalloc.start()

# 定义一个装饰器，用于实现单例模式
def singleton(cls):
    # 创建一个字典，用于存储实例
    instances = {}

    # 定义一个函数，用于获取实例
    def get_instance(*args, **kwargs):
        # 如果字典中没有该类的实例，则创建一个实例
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        # 返回该类的实例
        return instances[cls]

    # 返回获取实例的函数
    return get_instance


n_globalInfo=0



class GlobalInfo:
    # 初始化全局信息类
    def __init__(self, batch_size=args.batch_size, buffer_capacity=args.memory_size):
        global n_globalInfo
        n_globalInfo+=1

        self.best_score = 0
        self.score=0
        self.last_count=0

        # 设置批量大小
        self.batch_size = batch_size
        # 初始化信息字典
        self._info = {}
        # 初始化DQN记忆库
        self.dqn_memory = ReplayMemory(buffer_capacity)
        # 初始化锁
        self.lock = threading.Lock()

    def set_value(self, key, value):
        self._info[key] = value

    def get_value(self, key):
        return self._info.get(key, None)

    # -------------------------------对局状态-------------------------------------
    def set_game_start(self):
        self.set_value('start_game', True)

    def is_start_game(self):
        start_game = self.get_value('start_game')
        if start_game is None:
            return False
        else:
            return start_game

    def set_game_end(self):
        self.set_value('start_game', False)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        #写入txt文件
        with open('global_info_memory_usage.txt', 'w+') as f:
            f.write('-----------Memory usage-----------\n')
            f.write('[top 10]\n')
            for i in range(10):
                f.write(str(top_stats[i]))
                f.write('\n')
            f.write('[end]')
        self.monitor_memory()

    # -------------------------------dqn经验池-------------------------------------
    def store_transition_dqn(self, *args):
        self.dqn_memory.push(*args)

    def is_memory_bigger_batch_size_dqn(self):
        if len(self.dqn_memory) < self.batch_size:
            return False
        else:
            return True

    def random_batch_size_memory_dqn(self):
        transitions = self.dqn_memory.sample(self.batch_size)
        return transitions

   
    #监控内存
    def monitor_memory(self):
        # all_objects = muppy.get_objects()
        # sum1 = summary.summarize(all_objects)
        # summary.print_(sum1)

        # 获取当前进程内存占用。
        pid = os.getpid()
        p = psutil.Process(pid)
        info = p.memory_full_info()
        memory = info.uss / 1024. / 1024. / 1024.
                
        with open('memory.txt', 'w') as f:
            f.write(str(datetime.datetime.now()) + '\n')
            # f.write(str(sum1) + '\n')  太长了
            f.write(str(memory) + ' GB\n')
            



