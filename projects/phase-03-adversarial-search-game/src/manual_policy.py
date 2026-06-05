import pygame
from src.agent import Agent


class Manual(Agent):
    def __init__(self, color=(255, 255, 0)):
        super().__init__(name="Han",
                         avatar_path="src/icons/Han.png",
                         color=color)
        self.x = 0

    def get_action(self, state):
        if self.x == 0:
            self.x = 1
            return "U"
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        exit()

                    if event.key == pygame.K_w: return "U"
                    if event.key == pygame.K_s: return "D"
                    if event.key == pygame.K_a: return "L"
                    if event.key == pygame.K_d: return "R"

            pygame.time.wait(10)
