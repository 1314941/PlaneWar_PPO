import threading
import time
import os
import cv2
import numpy as np
from android_tool import AndroidTool
from argparses import args
from ppo import PPO
import torch
from getReword import GetRewordUtil,read_score,write_score
from globalInfo import GlobalInfo,n_globalInfo
from GameWrapper_ppo import GameWrapper
from torch.utils.tensorboard import SummaryWriter
from wzry_env_ppo import Environment
from onnxRunner import OnnxRunner
import multiprocessing
import memory_profiler
from functools import partial
import tracemalloc
import json

tracemalloc.start()

class TrainWorker:
    def _init__(self):
        pass
      
  
    def start(self,s_pipe,a_pipe):
        globalInfo = GlobalInfo()
        # agent=DQNAgent()
        agent=PPO()

        self.data_collector(s_pipe,a_pipe,globalInfo,agent)

   
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
        total_reward=0
        evaluate_num=0
        iter=0
        writer = SummaryWriter(args.log_dir)

        self.all_memory = []
        self.replay_memory = []
        self.bad_replay_memory = []  #实验，反复学习失败的经验

        while True:
            # 初始化对局状态 对局未开始
            self.globalInfo.set_game_end()

            checkGameStart='not started'
            # 判断对局是否开始
            if self.rewordUtil.game_over ==False:
                checkGameStart = 'started'

            
            if checkGameStart == 'started':
                print("-------------------------------对局开始-----------------------------------")
                self.globalInfo.set_game_start()
                 # 获取当前的图像
                state = self.tool.screenshot_window()
                  # 保证图像能正常获取
                if state is None:
                    print("图像获取失败")
                    time.sleep(0.01)
                    continue

                episode_reward = 0
              
  
                # 对局开始了，进行训练
                while self.globalInfo.is_start_game():
                    action= self.agent.select_action(state)
                    # print("move:",action[0],"distance:",action[1],"attack:",action[2])

                    next_state, reward, done, info = self.env.step(action,s_pipe,a_pipe)
                    episode_reward += reward
                   
                    if reward < 0 and done!=1:
                        print("bad_action")
                        self.bad_replay_memory.append([state, action, reward, next_state, done])
                    else:
                        print("action")
                        self.replay_memory.append([state, action, reward, next_state, done])

                    print("info:",info, "reward:",reward)
                
                    self.globalInfo.score=self.rewordUtil.score
                    state = next_state
                  
                    # 对局结束
                    if done == 1:
                        self.agent.save_model()
                        if float(self.rewordUtil.score)>float(self.rewordUtil.best_score):
                            #去除上次最佳分数
                            # try:
                            #     os.remove('src/ppo/score_{}_{}.pt'.format(args.model_name, self.rewordUtil.best_score))
                            # except:
                            #     pass
                            write_score(self.rewordUtil.score)
                            self.rewordUtil.best_score=self.rewordUtil.score
                            self.globalInfo.best_score=self.rewordUtil.best_score
                            a_path='src/ppo/{}_{}_a.pt'.format(args.model_name, self.rewordUtil.best_score)
                            c_path='src/ppo/{}_{}_c.pt'.format(args.model_name, self.rewordUtil.best_score)
                            self.agent.save_model(a_path,c_path)
                           

                       
                        # 每迭代10次，进行一次评估
                        if (iter+1) % 10 == 0:
                            # 记录评估次数
                            evaluate_num += 1
                            # 打印评估次数和奖励
                            print("evaluate_num:{} \t episode_return:{} \t".format(evaluate_num, episode_reward))
                            # 将评估奖励写入日志
                            writer.add_scalar('step_rewards', episode_reward, global_step= iter)

                            with open('log/log.txt', 'a') as f:
                                f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                                f.write('\n')
                                f.write(" evaluate_num:{} \t average_reward:{} \t".format(evaluate_num, total_reward/10))
                                f.write('\n')
                                #最高分
                                f.write(" best_score:{} \t".format(self.globalInfo.best_score))
                                f.write('\n')
                            a_path='src/ppo/count_{}_{}_a.pt'.format(args.model_name, evaluate_num)
                            c_path='src/ppo/count_{}_{}_c.pt'.format(args.model_name, evaluate_num)
                            self.agent.save_model(a_path,c_path)
                            total_reward=0

                        max_bad_memory_size=int(args.ppo_batch_size)
                        #限制bad_replay_memory的长度
                        if len(self.bad_replay_memory)>max_bad_memory_size:
                            print("bad_replay_memory too long, cut it")
                            self.bad_replay_memory=self.bad_replay_memory[-max_bad_memory_size:]
                        update=len(self.replay_memory)+len(self.bad_replay_memory)-args.ppo_batch_size
                        print("update:",update)
                        if update>0:
                            print("update model")
                            self.all_memory = self.replay_memory + self.bad_replay_memory
                            print("all_memory:",len(self.all_memory))
                            self.agent.update(self.all_memory)
                            del self.replay_memory[:]
                            del self.all_memory[:]
                            self.replay_memory=[]
                            self.all_memory=[]

                            del self.bad_replay_memory[:]
                            self.bad_replay_memory=[]

                            iter+=1
                            total_reward+=episode_reward

                        print("-------------------------------对局结束-----------------------------------")
                        self.globalInfo.set_game_end()
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
    #重置最高分
    write_score(0)
    #检测cuda是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("cuda is available")

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