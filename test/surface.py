import pygame
import time

# 初始化Pygame
pygame.init()

# 设置Surface大小
screen_width, screen_height = 640, 480
screen = pygame.Surface((screen_width, screen_height))

# 创建Sprite类
class MySprite(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((50, 50))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect(center=(screen_width // 2, screen_height // 2))
        self.speed = 5

    def update(self):
        self.rect.x += self.speed
        if self.rect.left > screen_width or self.rect.right < 0:
            self.speed = -self.speed

# 创建Sprite实例并添加到精灵组
sprite_group = pygame.sprite.Group()
sprite = MySprite()
sprite_group.add(sprite)

# 截屏间隔（秒）
screenshot_interval = 1
last_screenshot_time = time.time()

# 主循环
running = True
while running:
    current_time = time.time()

    # 更新sprites
    sprite_group.update()

    # 渲染并截屏
    screen.fill((255, 255, 255))
    sprite_group.draw(screen)

    # 如果达到截屏间隔，则保存截屏
    if current_time - last_screenshot_time >= screenshot_interval:
        pygame.image.save(screen, f'screenshot_{int(current_time)}.png')
        last_screenshot_time = current_time

    # 休眠一段时间
    time.sleep(0.1)

# 退出Pygame
pygame.quit()
