import sys
import pygame
import time
import tracemalloc

pygame.init()
from pygame import Rect, init, time, display
from PlaneGame.plane_sprites import *
# **************************************************************
# FileName: plane_main.py***************************************
# Author:  Junieson *********************************************
# Version:  2019.8.12 ******************************************
# ****************************************************************
class PlaneGame(object):
    """飞机大战主游戏"""

    def __init__(self,human=True,score=0):
        print("游戏初始化")
        # 创建对象池
        self.bullet_pool = ObjectPool(lambda: Bullet(), 1000) 
        self.enemy_pool = ObjectPool(lambda: Enemy(), 40)  

        if not human:
            screen_width, screen_height = 480, 700
            self.screen = pygame.Surface((screen_width, screen_height))
        else:
        #    1. 创建游戏的窗口
            self.screen = pygame.display.set_mode(SCREEN_RECT.size)
            # 设置窗口标题
            pygame.display.set_caption("PlaneWar")
            # 创建结束界面
        self.canvas_over = CanvasOver(self.screen)
        # 2. 创建游戏的时钟
        self.clock = pygame.time.Clock()
        # 3. 调用私有方法，精灵和精灵组的创建
        self.__create_sprites()

        # 分数对象
        self.score = GameScore()
        self.score.reset(score)
        self.refresh_data()

        # 程序控制指针
        self.index = 0
        # 音乐bgm
        self.bg_music = pygame.mixer.Sound("./music/game_music.ogg")
        self.bg_music.set_volume(0.3)
        self.bg_music.play(-1)
        self.display=pygame.display
        # 游戏结束了吗
        self.game_over = False

        self.playing_game=False
        self.stop=True

        self.human=human


    def sleep(self, sec=0.1):
        pygame.time.delay(int(sec * 1000)) #以毫秒为单位的整数参数，因此需要将秒数转换为毫秒。

    def __create_sprites(self):

        # 创建背景精灵和精灵组
        bg1 = Background()
        bg2 = Background(True)
        self.back_group = pygame.sprite.Group(bg1, bg2)

        # 创建敌机的精灵组

        self.enemy_group = pygame.sprite.Group()
        self.boss_group = pygame.sprite.Group()

          # 创建英雄的精灵和精灵组
        self.hero = Hero()
        self.hero_group = pygame.sprite.Group(self.hero)

        # 创建敌军子弹组
        self.enemy_bullet_group = pygame.sprite.Group()

        # 血条列表
        self.bars = []
        self.bars.append(self.hero.bar)

        # 创建buff组
        self.buff1_group = pygame.sprite.Group()

        # 创建假象boom组
        self.enemy_boom = pygame.sprite.Group()

        # bomb列表
        self.bombs = []

    def destroy(self):
        self.bg_music.stop()
        self.hero.reset()
        #先回收子弹  不然boss销毁时会导致子弹消失  造成bug
        for bullet in self.enemy_bullet_group:
            self.bullet_pool.release_object(bullet)
            bullet.hide()

        for enemy in self.enemy_group:
            if enemy.i_is_boss:
                enemy.kill()  #被boss打死，回收boss 不然会有随机飞弹
                continue
            self.enemy_pool.release_object(enemy)
            enemy.hide()

        

        self.enemy_group.empty()
        self.boss_group.empty()
        self.enemy_bullet_group.empty()
        self.back_group.empty()
        self.hero_group.empty()
        self.enemy_boom.empty()
        self.buff1_group.empty()
        self.bombs.clear()
        # self.bars.clear()

    def stop_timer(self):
        pygame.time.set_timer(CREATE_ENEMY_EVENT, 0)
        pygame.time.set_timer(HERO_FIRE_EVENT, 0)
        pygame.time.set_timer(BUFF1_SHOW_UP, 0)
        pygame.time.set_timer(BUFF2_SHOW_UP, 0)
        pygame.time.set_timer(ENEMY_FIRE_EVENT, 0)
        self.playing_game = False

    def restart_timer(self):
        #本想实现计时器恢复继续之前的进度，但不好搞，暂时先这样
        pygame.time.set_timer(CREATE_ENEMY_EVENT, random.randint(1000, 2000))  # 1-2秒
        pygame.time.set_timer(HERO_FIRE_EVENT, 400)
        pygame.time.set_timer(BUFF1_SHOW_UP, random.randint(10000, 20000))
        pygame.time.set_timer(BUFF2_SHOW_UP, random.randint(20000, 40000))
        pygame.time.set_timer(ENEMY_FIRE_EVENT, 2000)
        self.playing_game = True



    def refresh_data(self):
        self.hero.from_enemy_injury = 0
        self.from_hero_injury = 0
        self.killed_enemies = 0
        self.get_mate=0
        self.collided=0
        self.get_buff=0

    def reset(self):
        self.destroy()

        self.score.reset()
        self.refresh_data()

        # 3. 调用私有方法，精灵和精灵组的创建
        self.__create_sprites()

        # 分数对象
        self.score = GameScore()
        self.refresh_data()

        # 程序控制指针
        self.index = 0
        # 音乐bgm
        self.bg_music = pygame.mixer.Sound("./music/game_music.ogg")
        self.bg_music.set_volume(0.3)
        self.bg_music.play(-1)
        
        self.stop_timer()
        self.playing_game=False
        self.stop=True

        self.game_over = False

    def pygame_update(self):
        if not self.playing_game:
            self.restart_timer()
        
        # 1. 设置刷新帧率
        self.clock.tick(FRAME_PER_SEC)
        # 2. 事件监听
        self.__event_handler()
        # 3. 碰撞检测
        self.__check_collide()
        # 4. 更新/绘制精灵组
        self.__update_sprites()

        # pygame.display.update()
        if self.human:
            pygame.display.flip()

    def pygame_update_ai(self):
        while True:
            if not self.stop:
                if not self.playing_game:
                    self.restart_timer()
                
                # 1. 设置刷新帧率
                self.clock.tick(FRAME_PER_SEC)
                # 2. 事件监听
                self.__event_handler()
                # 3. 碰撞检测
                self.__check_collide()
                # 4. 更新/绘制精灵组
                self.__update_sprites()

                if self.human:
                    pygame.display.flip()
            else:
                self.sleep(0.1)
           


    def start_game(self):
        print("游戏开始...")

        while True:
            # 1. 设置刷新帧率
            self.clock.tick(FRAME_PER_SEC)
            # 2. 事件监听
            self.__event_handler()
            # 3. 碰撞检测
            self.__check_collide()
            # 4. 更新/绘制精灵组
            self.__update_sprites()

            # 是否要结束游戏

            if self.game_over:
                #self.canvas_over.show()
                #重新开
                self.__start__()

            # 5. 更新显示
            pygame.display.update()

    def screenshot(self):
        try:
            pygame.image.save(self.screen, "screenshot.png")
        except Exception as e:
            print("截图失败：" + str(e))
            # time.sleep(0.1)
            # self.screenshot()


    def get_data(self):
        from_enemy_injury=self.hero.from_enemy_injury
        # self.hero.from_enemy_injury=0

        from_hero_injury=self.from_hero_injury
        # self.from_hero_injury=0

        killed_enemies=self.killed_enemies
        # self.killed_enemies=0

        get_buff=self.get_buff
        # self.get_buff=0

        get_mate=self.get_mate
        # self.get_mate=0

        collided=self.collided
        # self.collided=0

        hp_percent=float(self.hero.bar.length)/self.hero.origin_hp
        #飞机 屏幕上方 percent<0   屏幕下方 percent>0  中间无奖励，上方惩罚 
        height_percent=float(self.hero.rect.bottom-SCREEN_RECT.centery)/SCREEN_RECT.height
        # high_percent=float(self.hero.rect.bottom-SCREEN_RECT.bottom)/SCREEN_RECT.height

        return from_enemy_injury, from_hero_injury, killed_enemies,get_buff,self.game_over,get_mate,collided,hp_percent,height_percent

    def __event_handler(self):  # 事件检测

        if self.score.getvalue() > 200+500*self.index:
            self.boss = Boss()
            self.enemy_group.add(self.boss)
            self.bars.append(self.boss.bar)
            self.index += 1

        for event in pygame.event.get():
            # 判断是否退出游戏
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == CREATE_ENEMY_EVENT:
                if len(self.enemy_group) < 20:
                    # 创建敌机精灵将敌机精灵添加到敌机精灵组
                    if self.score.getvalue() < 20:
                        enemy = self.enemy_pool.get_object()
                    else:
                        if random.randint(0, 100) % 4:
                            enemy = self.enemy_pool.get_object()
                        else:
                            enemy = self.enemy_pool.get_object()
                            enemy.upgrade()

                    self.enemy_group.add(enemy)
                    self.bars.append(enemy.bar)

            elif event.type == HERO_FIRE_EVENT:
                for hero in self.hero_group:
                    hero.fire(self.bullet_pool)
            elif event.type == BUFF1_SHOW_UP:
                buff1 = Buff1()
                self.buff1_group.add(buff1)
            elif event.type == BUFF2_SHOW_UP:
                if self.hero.bar.color == color_red:#按需分配
                    buff = Buff3()
                else:
                    buff= Buff2()
                self.buff1_group.add(buff)
            elif event.type == ENEMY_FIRE_EVENT:
                for enemy in self.enemy_group:
                    if enemy.number >= 2:
                        enemy.fire(self.bullet_pool)
                        for bullet in enemy.bullets:
                            self.enemy_bullet_group.add(bullet)

        # 使用键盘提供的方法获取键盘按键 - 按键元组
        keys_pressed = pygame.key.get_pressed()
        # 判断元组中对应的按键索引值 1
        if keys_pressed[pygame.K_RIGHT]:
            self.heros_move(5)
        elif keys_pressed[pygame.K_LEFT]:
            self.heros_move(-5)
        elif keys_pressed[pygame.K_UP]:
            self.heros_move(0, -5)
        elif keys_pressed[pygame.K_DOWN]:
            self.heros_move(0, 5)


    def heros_move(self, x=0, y=0):
        self.hero.speedx = x
        self.hero.speedy = y
        # print("set hero speed:",x,y)


    def bomb_throw(self):
        music_use_bomb = pygame.mixer.Sound("./music/use_bomb.wav")
        if self.hero.bomb > 0:
            music_use_bomb.play()
            self.hero.bomb -= 1
            self.bombs.pop()
            for enemy in self.enemy_group:
                if enemy.number < 3:
                    enemy.bar.length = 0
                    enemy.isboom = True
                    self.from_hero_injury += enemy.bar.value
                else:
                    enemy.injury = 20
                    enemy.isboom = True
                    self.from_hero_injury += enemy.injury

    def __check_collide(self):

        # 1. 子弹摧毁敌机
        for enemy in self.enemy_group:
            for hero in self.hero_group:
                for bullet in hero.bullets:
                    if pygame.sprite.collide_mask(bullet, enemy):  # 这种碰撞检测可以精确到像素去掉alpha遮罩的那种哦
                        self.bullet_pool.release_object(bullet)
                        hero.bullets.remove(bullet)
                        bullet.hide()
                        # bullet.kill()

                        enemy.injury = bullet.hity
                        enemy.isboom = True
                        self.from_hero_injury += bullet.hity


                        
        # 2. 敌机撞毁英雄
        for enemy in self.enemy_group:
            #撞到本体才有伤害
            if pygame.sprite.collide_mask(self.hero, enemy):
                if enemy.bar.length <= 0:  #正在爆炸
                    continue
                self.collided += 1
                if enemy.number < 3:
                    enemy.bar.length = 0  # 敌机直接死
                    self.hero.injury = self.hero.bar.value / 4  # 英雄掉四分之一的血
                    if self.hero.buff1_num > 0:
                        self.hero.buff1_num -= 1
                        self.hero.music_degrade.play()
                        self.get_buff-=1
                    self.enemy_boom.add(enemy)
                    enemy.isboom = True
                    self.from_hero_injury += enemy.bar.value
                    self.killed_enemies += self.caulate_kill(enemy.number)
                else:
                    self.hero.from_enemy_injury=self.hero.bar.length/self.hero.bar.length_init
                    self.hero.bar.length = 0
                self.hero.isboom = True

        # 子弹摧毁英雄
        for bullet in self.enemy_bullet_group:
            if pygame.sprite.collide_mask(self.hero, bullet):
                self.bullet_pool.release_object(bullet)
                self.enemy_bullet_group.remove(bullet)
                bullet.hide()

                self.hero.injury = 1
                if self.hero.buff1_num > 0:
                    self.hero.music_degrade.play()
                    if self.hero.buff1_num== 5:
                        self.get_mate=-1
                        #副机死，子弹销
                        for bullet1 in self.mate1.bullets:
                            self.bullet_pool.release_object(bullet1)
                            self.mate1.bullets.remove(bullet1)
                            bullet1.hide()
                        for bullet2 in self.mate2.bullets:
                            self.bullet_pool.release_object(bullet2)
                            self.mate2.bullets.remove(bullet2)
                            bullet2.hide()
                        self.mate1.kill()
                        self.mate2.kill()
                    self.hero.buff1_num -= 1
                    self.get_buff-=1

                self.hero.isboom = True

        # if not self.hero.alive():
        if self.hero.is_hide: # 英雄死了
            if self.hero.buff1_num == 5:
                self.mate1.rect.right = -10
                self.mate2.rect.right = -10
            self.game_over = True

        # 3.buff吸收
        for buff in self.buff1_group:
            if pygame.sprite.collide_mask(self.hero, buff):
                self.get_buff+=buff.speedy*2  #buff越厉害，奖励越多

                buff.music_get.play()
                if buff.speedy == 1:  # 用速度来区分
                    if self.hero.buff1_num < 6:
                        self.hero.buff1_num += 1
                        self.hero.music_upgrade.play()
                        if self.hero.buff1_num == 5:
                            self.team_show()
                elif buff.speedy==2:
                    self.hero.bomb += 1
                    image = pygame.image.load("./images/bomb.png")
                    self.bombs.append(image)
                elif buff.speedy==3:
                    if self.hero.bar.length < self.hero.bar.weight*self.hero.bar.value:
                        self.hero.bar.length += self.hero.bar.weight*self.hero.bar.value
                buff.kill()

    def team_show(self):
        #爆种状态
        self.get_mate+=1
        self.mate1 = Heromate(-1)
        self.mate2 = Heromate(1)
        self.mate1.image = pygame.image.load("./images/life.png")
        self.mate1.rect = self.mate1.image.get_rect()
        self.mate2.image = pygame.image.load("./images/life.png")
        self.mate2.rect = self.mate2.image.get_rect()
        self.hero_group.add(self.mate1)
        self.hero_group.add(self.mate2)

    def caulate_kill(self,level):
        if level == 1:
            return 0.1
        elif level == 2:
            return 0.4
        elif level == 3:
            return 1

    # 各种更新
    def __update_sprites(self):

        self.back_group.update()
        self.back_group.draw(self.screen)

      
        for enemy in self.enemy_group:
            enemy.update()
            if enemy.i_is_boss:  #boss 特权
                    continue
            
            if enemy.is_hide:
                lt=len(self.enemy_group)
                self.enemy_group.remove(enemy)
                self.enemy_boom.add(enemy)
                #飞出去的不要计数
                self.killed_enemies += self.caulate_kill(enemy.number)
               
                self.enemy_pool.release_object(enemy)
                # nlt=len(self.enemy_group)
                # if lt==nlt+1:
                #     print("enemy removed successfully")
                # else:
                #     print("enemy removed failed")

        self.enemy_group.draw(self.screen)

        self.enemy_boom.update()
        self.enemy_boom.draw(self.screen)

        self.heros_update()
        self.hero_group.draw(self.screen)

        for hero in self.hero_group:
            #注意，这里不能用self.hero ！ 不然敌机可能射出辅机的子弹，自杀
            hero.bullets.update(self.bullet_pool,hero.bullets)
            hero.bullets.draw(self.screen)

        self.buff1_group.update()
        self.buff1_group.draw(self.screen)

        self.bars_update()
        self.bombs_update()

        self.enemy_bullet_group.update(self.bullet_pool,self.enemy_bullet_group)
        self.enemy_bullet_group.draw(self.screen)

        self.score_show()

    def heros_update(self):
        for hero in self.hero_group:
            if hero.number == 1:
                hero.rect.bottom = self.hero.rect.bottom
                hero.rect.left = self.hero.rect.right
            if hero.number == -1:
                hero.rect.bottom = self.hero.rect.bottom
                hero.rect.right = self.hero.rect.left
            hero.update()

    def bars_update(self):
        for bar in self.bars:
            if bar.length > 0:
                bar.update(self.screen)
            else:
                self.bars.remove(bar)

    def bombs_update(self):
        i = 1
        for bomb in self.bombs:
            self.screen.blit(bomb, (0, 700 - (bomb.get_rect().height) * i))
            i += 1

    def score_show(self):
        score_font = pygame.font.Font("./STCAIYUN.ttf", 33)
        image = score_font.render("SCORE:" + str(int(self.score.getvalue())), True, color_gray)
        rect = image.get_rect()
        rect.bottom, rect.right = 700, 480
        self.screen.blit(image, rect)

    def set_score(self,score=0):
        self.score.reset(score=score)


    @staticmethod
    def __start__():
        # 创建游戏对象
        game = PlaneGame()

        # 启动游戏
        game.start_game()


if __name__ == '__main__':
    PlaneGame.__start__()
