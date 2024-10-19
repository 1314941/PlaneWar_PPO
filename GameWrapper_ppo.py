#通过终端启动程序
# python wrapper.py

#--coding:utf-8--
import os
import sys
from PlaneGame.plane_main import PlaneGame,FRAME_PER_SEC
import json
import time
import multiprocessing
import tracemalloc
import cv2 as CV2
import cv2
import numpy as np
from PyQt5.QtGui import QImage
import threading
from functools import partial
from pygame.surfarray import array3d, pixels_alpha
import torch
from argparses import args,device
# 创建一个锁对象
lock = multiprocessing.Lock()


# def start_game():
#     #切换到指定目录的子目录
#     os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PlaneGame'))
#     #启动程序
#     os.system('python plane_main.py')

DATA_FILE='game_data.json'
ACTION_FILE='action_data.json'

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if end_time - start_time > 1:
            print(f"函数 {func.__name__} 花费了 {end_time - start_time} 秒来执行")
        return result
    return wrapper



def pre_processing(image, width, height):
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image[None, :, :].astype(np.float32)




class GameWrapper:
    def __init__(self):
        #Lock objects should only be shared between processes through inheritance
        # self.lock = multiprocessing.Lock()  # 创建一个锁对象

        self.is_reading = False
        self.is_saving = False
        self.update=False
        self.stop=True
        self.need_screen=False
        self.arr=None
        self.level=3  #关卡
        pass
  
    #human为False时，不显示，只进行训练(简称静默模式)
    def start_game(self,s_pipe,a_pipe,human=True):
        try:
            self.game = PlaneGame(human)
            self.set_level(self.level)

            if not human: 
                game_process = threading.Thread(target=self.game.pygame_update_ai)
                game_process.start()

            screen_process = threading.Thread(target=self.screenshot_proc)
            screen_process.start()
            
            print("游戏开始...")
            while True:
                if s_pipe.poll():
                    if self.stop: #有信号，说明ai启动成功
                        self.stop=False
                    data=s_pipe.recv()
                    if data=="refresh":  
                        self.game.refresh_data()
                    #json格式数据
                    elif data=="move":
                        action=a_pipe.recv()
                        self.move(action['direction'],action['distance'])
                    elif data=="attack":
                        self.attack()
                    elif data=="get_game_data":
                        a_pipe.send(self.get_game_data())
                    elif data=="game restart":
                        if self.game.game_over ==True:
                            #清空游戏
                            time.sleep(1)
                            self.game.reset()
                            self.set_level(self.level)
                            s_pipe.send("game restart success")
                        self.stop=True
                    elif data=="screenshot":
                        self.stop=True
                        self.need_screen=True
                        while self.need_screen:
                            time.sleep(0.1)
                        s_pipe.send("screenshot success")
                        a_pipe.send(self.arr)
                      
                # 先将死前状态传送过去，再刷新数据
              
                if human==True:
                    #时间大户
                    self.render()  
                else:
                    self.game.stop=self.stop
        except Exception as e:
            print(f"starting game generate an error: {e}")


    #关卡
    def set_level(self,level=1):
        if level==1:
           self.game.set_score(0)
        elif level==2:
           self.game.set_score(50)
           self.game.hero.buff1_num=2
        elif level==3:
           self.game.set_score(200)
           self.game.hero.buff1_num=5
           self.game.team_show()

    @timer_decorator
    def render(self):
        if not self.stop:
            #启动时没有添加计时器，update时添加一个计时器  重置时静止计时器  刷新时开始计时器
            self.game.pygame_update()
  


    
    
    @timer_decorator
    def screenshot(self):
        try:
            self.image_size = 84
            self.screen_height=700
            self.screen_width=480
            self.base_y = self.screen_height * 0.79

            image = array3d(self.game.display.get_surface())
            image = pre_processing(image[:self.screen_width, :int(self.base_y)], self.image_size, self.image_size)

            image = torch.tensor(image).to(device)
            state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
            return state
        except Exception as e:
            print("screenshot error:",e)
            return self.screenshot()
       


    def screenshot_proc(self):
        while True:
            if self.need_screen:
                self.arr=self.screenshot()
                self.need_screen=False
            time.sleep(0.1)


    def is_game_over(self):
        data=self.read_data()
        if data is None:
            return True
        return data['game_over']

    #分数
    def get_score(self):
       data=self.read_data()
       if data is None:
           return 0
       score=data['score']
       return float(score)
    
 
    def attack(self):
        self.game.bomb_throw()


    def get_game_data(self):
        from_enemy_injury, from_hero_injury, killed_enemies,get_buff,game_over,get_mate,collided,hp_percent,height_percent=self.game.get_data()
        score=self.game.score.getvalue()
        if float(score)>200:
            self.level=3
        elif float(score)>50:
            self.level=2

        data={}
        data['e_injury']=str(from_enemy_injury)
        data['h_injury']=str(from_hero_injury)
        data['killed_enemies']=str(killed_enemies)
        data['get_buff']=str(get_buff)
        data['game_over']=game_over
        data['score']=str(score)
        data['get_mate']=str(get_mate)
        data['collided']=str(collided)
        data['hp_percent']=str(hp_percent)
        data['height_percent']=str(height_percent)


        return data

    def read_data(self,data_file=DATA_FILE):
        lock.acquire()
        try:
            with open(data_file, 'r',encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            #print(f"read json {data_file} error :",e,"文件为空或格式不正确")
            data = None
        finally:
            lock.release()
        return data
    
    

    @timer_decorator
    def move(self,direction,distance=5):
        try:
            # distance=5
            direction=int(direction)
            if direction == 1:
                self.game.heros_move(distance)
            elif direction == 2:
                self.game.heros_move(distance*-1)
            elif direction == 3:
                self.game.heros_move(0,distance)
            elif direction == 4:
                self.game.heros_move(0,distance*-1)
        except Exception as e:
            print("moving error:",e)
        
 