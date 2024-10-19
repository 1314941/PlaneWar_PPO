import threading
import time
import os
import cv2
import numpy as np
from android_tool import AndroidTool
from argparses import args
from dqnAgent import DQNAgent
from getReword import GetRewordUtil,read_score,write_score
from globalInfo import GlobalInfo,n_globalInfo
from GameWrapper import GameWrapper

from wzry_env import Environment
from onnxRunner import OnnxRunner
import multiprocessing
import memory_profiler
from functools import partial
import tracemalloc
import json

tracemalloc.start()

# Exception has occurred: ModuleNotFoundError
# No module named 'resource'
#   File "C:\chat\wzry_ai-main\train.py", line 2, in <module>
#     import resource
# ModuleNotFoundError: No module named 'resource' 莫名其妙的bug 如装

# def set_memory_limit(max_mem_in_bytes):
#     soft, hard = resource.getrlimit(resource.RLIMIT_AS)


#     resource.setrlimit(resource.RLIMIT_AS, (max_mem_in_bytes, hard))

# set_memory_limit(6*1024*1024*1024)  #10GB

# 全局变量  改了，还是爆内存
# globalInfo = GlobalInfo()


# gameWrapper=GameWrapper()
# rewordUtil = GetRewordUtil(gameWrapper=gameWrapper)

# tool = AndroidTool(gameWrapper=gameWrapper)
# tool.show_scrcpy()
# tool.show_action_log()
# env = Environment(tool, rewordUtil, gameWrapper)

# agent = DQNAgent()



class TrainWorker:
    def _init__(self):
        pass

  
    def start(self,s_pipe,a_pipe):
        globalInfo = GlobalInfo()
        agent=DQNAgent()
        train_process = threading.Thread(target=partial(self.train_agent,globalInfo,agent))
        train_process.start()

        self.data_collector(s_pipe,a_pipe,globalInfo,agent)

    def train_agent(self,globalInfo,agent):
        # 初始化计数器
        count = 1
        last_count=0
        # 无限循环
        while True:
            # 如果内存中的数据量小于batch_size，则等待1秒
            if not globalInfo.is_memory_bigger_batch_size_dqn():
                time.sleep(1)
                continue
            # 打印训练信息
            # print("training")
            # 回放记忆
            self.replay(globalInfo, agent)
            # 如果计数器达到num_episodes，则保存模型
            if count % args.num_episodes*10 == 0:
                self.agent.save_model('src/dqn/PlaneWar.pt')
                if float(self.globalInfo.score)>float(self.globalInfo.best_score):  
                    #去除上次最佳分数
                    try:
                        if last_count!=0:
                            os.remove('src/dqn/count_{}_{}.pt'.format(args.model_name, self.globalInfo.best_score))
                    except:
                        pass
                    agent.save_model('src/dqn/count_{}_{}.pt'.format(args.model_name, count))
                    last_count=count
                pass
            # 计数器加1
            count = count + 1
            # 如果计数器大于等于100000，则重置计数器
            if count >= 100000:
                count = 1

    def replay(self,globalInfo, agent):
        transitions = globalInfo.random_batch_size_memory_dqn()
        # print("length of transitions:", len(transitions))
        try:
            agent.replay(transitions)
           
            # print(f"replay finished")
        except Exception as exc:
            print(f"replay generated an exception: {exc}")


   
    def data_collector(self,s_pipe,a_pipe,globalInfo,agent):
        gameWrapper=GameWrapper()
        self.rewordUtil = GetRewordUtil(gameWrapper=gameWrapper)
        self.tool = AndroidTool(s_pipe=s_pipe,a_pipe=a_pipe)
        # self.globalInfo = GlobalInfo()
        self.globalInfo=globalInfo
        self.agent=agent
        self.env = Environment(self.tool, self.rewordUtil, gameWrapper,self.globalInfo)
        # self.agent = DQNAgent()

        self.globalInfo.best_score=read_score()

        while True:
            # 初始化对局状态 对局未开始
            self.globalInfo.set_game_end()

            checkGameStart='not started'
            # 判断对局是否开始
            if self.rewordUtil.game_over ==False:
                checkGameStart = 'started'

            step=0
            if checkGameStart == 'started':
                print("-------------------------------对局开始-----------------------------------")
                self.globalInfo.set_game_start()

                # 对局开始了，进行训练
                while self.globalInfo.is_start_game():
                     # 获取当前的图像
                    state = self.tool.screenshot_window()
                    # 保证图像能正常获取
                    if state is None:
                        print("图像获取失败")
                        time.sleep(0.01)
                        continue

                    # 获取预测动作
                    action = self.agent.select_action(state)

                    next_state, reward, done, info = self.env.step(action,s_pipe,a_pipe)

                    step+=1
                    if reward!=100:
                        print("info:",info, "reward:",reward)
                        pass

                    self.globalInfo.score=self.rewordUtil.score

                    # 追加经验
                    self.globalInfo.store_transition_dqn(state, action, reward, next_state, done)
                    del state
                    state = next_state

                    # 对局结束
                    if done == 1:
                        # del self.agent
                        # self.agent=DQNAgent()

                        if float(self.rewordUtil.score)>float(self.rewordUtil.best_score):
                            #去除上次最佳分数
                            try:
                                os.remove('src/dqn/score_{}_{}.pt'.format(args.model_name, self.rewordUtil.best_score))
                            except:
                                pass
                            write_score(self.rewordUtil.score)
                            self.rewordUtil.best_score=self.rewordUtil.score
                            self.globalInfo.best_score=self.rewordUtil.best_score
                            # self.agent.save_model('src/dqn/PlaneWar.pt')
                            self.agent.save_model('src/dqn/score_{}_{}.pt'.format(args.model_name, self.rewordUtil.best_score))

                        print("-------------------------------对局结束-----------------------------------")
                        self.globalInfo.set_game_end()
                        snapshot = tracemalloc.take_snapshot()
                        top_stats = snapshot.statistics('lineno')
                        #写入txt文件
                        with open('memory_usage.txt', 'w+') as f:
                            f.write('-----------Memory usage-----------\n')
                            f.write('[top 10]\n')
                            for i in range(10):
                                f.write(str(top_stats[i]))
                                f.write('\n')
                            f.write('[end]\n')
                            f.write('---num---\n')
                            f.write('globalInfo num:{}\n'.format(n_globalInfo))
                            f.write('----------\n')
                            break
            else:
                print("对局未开始")
                print("尝试重启游戏")
                s_pipe.send('game restart')

                if s_pipe.poll(1):
                    data=s_pipe.recv()
                    if data=='game restart success':
                        print("游戏重启成功")
                        self.rewordUtil.game_over=False
                time.sleep(0.1)
           


if __name__ == '__main__':
    s_parent_conn,s_child_conn = multiprocessing.Pipe()
    a_parent_conn,a_child_conn = multiprocessing.Pipe()

    gameWrapper=GameWrapper()
    trainWorker=TrainWorker()

  

    #multiprocessing模块在尝试序列化main函数时遇到了问题。multiprocessing模块需要将函数对象传递给子进程
    #使用functools.partial函数来创建一个新的函数对象，该函数对象可以传递给multiprocessing模块。
    main_process = multiprocessing.Process(target=partial(trainWorker.start,s_child_conn,a_child_conn))
    main_process.start()
    
    game_process =multiprocessing.Process(target=partial(gameWrapper.start_game,s_parent_conn,a_parent_conn))
    game_process.start()

    main_process.join()    
    game_process.join()