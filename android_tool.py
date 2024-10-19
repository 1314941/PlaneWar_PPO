import random
import time
from ctypes import windll
import cv2 as CV2
import cv2
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
import json
import tracemalloc
import gc
import memory_profiler
from argparses import direction_actions_detail,attack_actions_detail, args
from globalInfo import singleton
from GameWrapper import timer_decorator

tracemalloc.start()

@singleton
class AndroidTool:
    def __init__(self,s_pipe=None,a_pipe=None):   
        self.actual_height, self.actual_width = 480,700
        self.s_pipe=s_pipe
        self.a_pipe=a_pipe
        self.count=0


    # @memory_profiler.profile
    @timer_decorator
    def screenshot_window(self):
        """
        截取指定窗口的内容并返回图像数据。

        参数:
        window_name (str): 窗口标题的部分或全部字符串。

        返回:
        np.ndarray: 截图的图像数据，如果窗口未找到则返回 None。
        """
        try:
            # # 将 QImage 转换为 numpy 数组
            if self.s_pipe:
                self.s_pipe.send('screenshot')
                # time.sleep(0.1)
            else:
                return None

            while True:
                # print('waiting for screenshot success')
                if self.s_pipe.poll(0.1):
                    sg = self.s_pipe.recv()
                    # print(sg)
                    if sg == 'screenshot success':
                        arr=self.a_pipe.recv()
                        break
                    
            if self.count%100==0:
                #打印内存使用情况
                # snapshot = tracemalloc.take_snapshot()
                # top_stats = snapshot.statistics('lineno')
                # #写入txt文件
                # with open('android_memory_usage.txt', 'w+') as f:
                #     f.write('-----------Memory usage-----------\n')
                #     f.write('[top 10]\n')
                #     for i in range(10):
                #         f.write(str(top_stats[i]))
                #         f.write('\n')
                #     f.write('[end]')
                #     f.write('\n-----------gc-----------\n')
                #     f.write(str(len(gc.garbage))+'\n\n')
               
                # #显式清理
                # tracemalloc.clear_traces()
                gc.collect()
            self.count+=1

            #无效
            
            return arr
        except Exception as e:
            print(f"An error occurred while taking a screenshot: {e}")
            return None


def generate_random_number(n):
    return random.randint(0, n)


