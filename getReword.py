import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from ppocronnx import TextSystem

from argparses import device
from globalInfo import GlobalInfo
from onnxRunner import OnnxRunner
from GameWrapper import GameWrapper
import os
import json

def read_score():
    if not os.path.exists('score.json'):
        return 0
    with open('score.json', 'r') as f:
        data = json.load(f)
    return float(data['best_score'])

def write_score(score):
    with open('score.json', 'w') as f:
        json.dump({'best_score': float(score)}, f)



class GetRewordUtil:
    def __init__(self,gameWrapper=None):
        self.device = device
        self.gameWrapper = gameWrapper
        self.game_over=False
        self.best_score=read_score()
        self.score=0

        self.height_percent=0
        self.hp_percent=1

        # 全局状态
        self.globalInfo = GlobalInfo()

    def predict(self):
        is_attack, rewordCount = self.calculate_attack_reword()
        return is_attack, rewordCount

    def calculate_attack_reword(self):
        isAttack = False
        #受到伤害，造成伤害，杀敌数
        self.s_pipe.send("get_game_data")

        #reward=self.one()

        reward=self.two()

        return isAttack,reward
    
    def two(self):
        #等待数据
        if self.a_pipe.poll(1):
            data = self.a_pipe.recv()
            # e_injury 血量下降百分比
            e_injury,h_kill,get_buff,collided = data['e_injury'],data['killed_enemies'],data['get_buff'],data['collided']
            e_injury = float(e_injury)
            h_kill = float(h_kill)
            get_buff = float(get_buff)
            collided = float(collided)
            
            self.hp_percent,self.height_percent = float(data['hp_percent']),float(data['height_percent'])
            self.score=float(data['score'])
            self.game_over=data['game_over']
            self.s_pipe.send("refresh")
        else:
            e_injury,h_kill,get_buff,collided = 0,0,0,0

        live_reward=0
        buff_reward=0
        if not self.game_over:
            live_reward=self.score/100 if self.score<200 else 2  # 100分->1
            # print("hp_percent:",hp_percent)
            live_reward=live_reward + self.hp_percent+self.height_percent*2.0 # # 血量越低，奖励越低
            buff_reward=get_buff
        
        #活着 杀敌 获得buff  
        if e_injury<=0:
            reward=live_reward*3.6+buff_reward*1.0
            reward=reward*0.1
        else:
            reward=(-100.0*e_injury)/(self.score+1) #碰撞 受到伤害

        return reward
     

    def one(self):
        #等待数据
        if self.a_pipe.poll(1):
            data = self.a_pipe.recv()
            # e_injury 血量下降百分比
            e_injury,h_kill,get_buff,collided = data['e_injury'],data['killed_enemies'],data['get_buff'],data['collided']
            e_injury = float(e_injury)
            h_kill = float(h_kill)
            get_buff = float(get_buff)
            collided = float(collided)
            
            self.hp_percent,self.height_percent = float(data['hp_percent']),float(data['height_percent'])
            self.score=float(data['score'])
            self.game_over=data['game_over']
            self.s_pipe.send("refresh")
        else:
            e_injury,h_kill,get_buff,collided = 0,0,0,0,0,0

        live_reward=0
        kill_reward=0
        buff_reward=0
        if not self.game_over:
            # live_reward=self.score/100 if self.score<200 else 2  # 100分->1
            live_reward=1
            # print("hp_percent:",hp_percent)
            live_reward=live_reward * self.hp_percent # 血量越低，奖励越低
            # print("live_reward:",live_reward,"height_reward:",height_percent*3)
            live_reward=live_reward + self.height_percent*4
            #击杀奖励
            kill_reward=h_kill*self.hp_percent+h_kill*self.height_percent*0.2
            buff_reward=get_buff*self.hp_percent+get_buff*self.height_percent*0.2
        #活着 杀敌 获得buff  碰撞 受到伤害
        reward=live_reward*1.0+kill_reward*1.5+buff_reward*4.5-5.0*e_injury
        return reward
     
    def calculate_reword(self, status_name, attack_reword, action):
        rewordResult = 0

        if status_name == "playing":
            rewordResult = attack_reword
        elif status_name == "successes":
            rewordResult = 10000
        elif status_name == "death":
            rewordResult = float(-10000/self.score)

        return rewordResult

    def check_finish(self):
        class_name = None
        done = 0
        if float(self.score)>=20000:
            done = 1
            class_name = "successes"
        # if self.check_death(image):
        if self.game_over:
            done = 1
            class_name = "death"
        return done, class_name

 

    def get_reword(self,  action,s_pipe,a_pipe):
        self.s_pipe=s_pipe
        self.a_pipe=a_pipe

        done = 0
        class_name = None
        md_class_name = "playing"
        is_attack, attack_rewordCount = self.predict()
        done, class_name = self.check_finish()
        
        # 如果没结束，判断局内状态
        if done == 0:
            if md_class_name is not None:
                class_name = md_class_name

        # 计算回报
        rewordCount = self.calculate_reword(class_name, attack_rewordCount, action)

        return rewordCount, done, class_name
    
    # def check_death(self, image):
    #     text_sys = TextSystem()
    #     res = text_sys.detect_and_ocr(image)
    #     done = 0
    #     class_name = None
    #     for boxed_result in res:
    #         # print("{}, {:.3f}".format(boxed_result.ocr_text, boxed_result.score))
    #         if boxed_result.ocr_text == "重新开始":
    #             done = 1
    #             class_name = 'death'
    #             break
    #     if class_name == 'death':
    #         return True
    #     return  False

