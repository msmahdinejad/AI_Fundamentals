from src.env import PaintBattle
from src.manual_policy import Manual
from src.bin.enemies import DustPig, DarthMaul, MrSibil
import pygame
from Minimax1 import MinimaxAgent

def main():
    env = PaintBattle("map1")
    p1 = MinimaxAgent(max_depth=9)
    p2 = DustPig()
    env.play(
        p2,
        p1
    )


if __name__ == "__main__":
    pygame.init()
    main()
