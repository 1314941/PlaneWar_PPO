# PlaneWar_PPO
飞机大战ai 
python版本 3.10.14
使用ppo算法的py文件带有ppo后缀，同名无后缀的使用的是dqn算法。其余皆为通用的。
txt文件多为修bug时图方便加上去的。score.json存放最高分，可以在
```
if __name__ == '__main__':
```
选择是否开局重置最高分
PlaneGame文件夹内存放游戏文件
ppo算法效果较好，并且由于晚开发一点，bug比dqn的少。
>自由选择开局分数，避免重复通关
# 借鉴
>[飞机大战源码 github](https://github.com/Junieson/PlaneGame)
>[王者荣耀ai训练](https://github.com/myBoris/wzry_ai)(试了一下，发现电脑带不动，就改了一点，训练其他游戏)
>[flappyBird](https://github.com/luozhiyun993/FlappyBird-PPO-pytorch)
