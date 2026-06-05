"""
Paint Battle - Competitive Multi-Agent Environment
Fundamentals of AI Course

University of Isfahan - Fall 2025

Course Instructor: Professor Dr. Hossein Karshenas
Teaching Assistants:
    - Pooya Esfandany
    - Danial Shafiei
    - Mahdi Mahdieh
    - Younes Rad

This environment implements a competitive two-player paint battle game on a grid.
Two agents compete to capture territory by painting tiles while avoiding elimination.
The project focuses on implementing adversarial search algorithms like Minimax,
Alpha-Beta Pruning, and heuristic evaluation functions.

Key Components:
- Agent: Base class for all player implementations (human or AI)
- PaintBattle: Main game environment with rendering and game logic
- PBState: Game state representation for search algorithms
- Manual: Human-controlled agent using keyboard input

Environment Features:
- Grid-based movement with obstacle avoidance
- Territory capture mechanics (similar to Go)
- Player elimination via area capture or collision
- Turn-based gameplay with Pygame visualization
- Customizable maps with different layouts
- Support for various AI agent implementations

Usage:
    from src.env import PaintBattle

    env = PaintBattle("map1")
    env.play(p1, p2)
"""

import pygame
import numpy as np
import os
import random
from collections import deque

CELL_SIZE = 64
HUD_H = 56
MAX_STEPS = 300
DELAY_MS = 120

NEUTRAL = 0
TEAM1 = 1
TEAM2 = 2

DIRECTIONS = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1),
}


class PBState:
    __slots__ = (
        "grid",
        "paint",
        "p1",
        "p2",
        "alive1",
        "alive2",
        "step"
    )

    def __init__(self, grid, paint, p1, p2, step=0, alive1=True, alive2=True):
        self.grid = grid
        self.paint = paint.copy()
        self.p1 = p1
        self.p2 = p2
        self.step = step
        self.alive1 = alive1
        self.alive2 = alive2
        if self.p1 is None:
            self.alive1 = False
        if self.p2 is None:
            self.alive2 = False


    def inside(self, r, c):
        R, C = self.grid.shape
        return 0 <= r < R and 0 <= c < C

    def copy(self):
        return PBState(
            self.grid,
            self.paint.copy(),
            self.p1,
            self.p2,
            self.step,
            self.alive1,
            self.alive2
        )

    def get_scores(self):
        s1 = np.sum(self.paint == TEAM1)
        s2 = np.sum(self.paint == TEAM2)
        return s1, s2

    def is_terminal(self):
        if not self.alive1 or not self.alive2:
            return True

        if self.step >= MAX_STEPS:
            return True

        total = np.sum(self.grid != 'R')
        painted = np.sum((self.paint == TEAM1) | (self.paint == TEAM2))

        return painted >= total
    def capture_logic(self, paint_grid, p1, p2, team):
        def detect_local_capture(grid,paint,team,player):
            grid_with_margin = np.pad(grid, pad_width=1, mode='constant', constant_values=-1)
            paint_grid_with_margin = np.pad(paint, pad_width=1, mode='constant', constant_values=-1)
            new_state = (player[0]+1, player[1]+1)
            grid_crop = grid_with_margin[new_state[0]-1:new_state[0]+2, new_state[1]-1:new_state[1]+2]
            paint_crop = paint_grid_with_margin[new_state[0]-1:new_state[0]+2, new_state[1]-1:new_state[1]+2]
            mask_team = (paint_crop == team).astype(np.uint8)
            mask_grid = (grid_crop == 'R').astype(np.uint8)
            result = mask_team + mask_grid
            color_count = result.sum()
            if color_count < 2:
                return False
            top = result[0]
            right = np.array([result[1][2]])
            left = np.array([result[1][0]])
            bottom = result[2, ::-1]
            pattern = np.concatenate((
                left,
                top,
                right,
                bottom,
                left,
            ))
            pattern_count = np.sum((pattern[:-1] == 1) & (pattern[1:] == 0))
            if pattern_count < 2:
                return False
            return True
        R, C = self.grid.shape
        walkable = (self.grid != 'R')
        free = np.zeros((R, C), dtype=bool)
        capture_detect= detect_local_capture(self.grid, paint_grid, team, p1 if team == TEAM1 else p2)
        if not capture_detect:
            return False,False
        q = deque()

        for r in range(R):
            for c in range(C):
                if r == 0 or r == R-1 or c == 0 or c == C-1:
                    if walkable[r, c] and paint_grid[r, c] != team:
                        free[r, c] = True
                        q.append((r, c))

        while q:
            r, c = q.popleft()
            for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C:
                    if walkable[nr, nc] and paint_grid[nr, nc] != team and not free[nr, nc]:
                        free[nr, nc] = True
                        q.append((nr, nc))

        enclosed = walkable & (~free) & (paint_grid != team)
        removed_p1 = False
        removed_p2 = False
        if team == TEAM1 and p2 is not None:
            if enclosed[p2[0], p2[1]]:
                removed_p2 = True
        if team == TEAM2 and p1 is not None:
            if enclosed[p1[0], p1[1]]:
                removed_p1 = True
        paint_grid[enclosed] = team

        return removed_p1, removed_p2
    def get_successors(self, team):
        successors = []

        actor = self.p1 if team == TEAM1 else self.p2
        if actor is None:
            return successors

        R, C = self.grid.shape

        for action, (dr, dc) in DIRECTIONS.items():
            r, c = actor
            nr, nc = r + dr, c + dc

            new_pos = (r, c)
            if abs(nr - r) + abs(nc - c) == 1 \
               and 0 <= nr < R and 0 <= nc < C \
               and self.grid[nr, nc] != 'R':
                new_pos = (nr, nc)

            new_paint = self.paint.copy()
            new_p1, new_p2 = self.p1, self.p2
            new_alive1, new_alive2 = self.alive1, self.alive2

            if team == TEAM1:
                new_p1 = new_pos
            else:
                new_p2 = new_pos

            if team == TEAM1 and new_alive1:
                new_paint[new_pos] = TEAM1
            if team == TEAM2 and new_alive2:
                new_paint[new_pos] = TEAM2

            removed_p1 = False
            removed_p2 = False

            if new_p1 is not None and new_p2 is not None and new_p1 == new_p2:
                s1 = np.sum(new_paint == TEAM1)
                s2 = np.sum(new_paint == TEAM2)
                if s1 > s2:
                    removed_p2 = True
                elif s2 > s1:
                    removed_p1 = True
                else:
                    removed_p1 = True
                    removed_p2 = True

            cap_p1, cap_p2 = self.capture_logic(new_paint, new_p1, new_p2, team)
            removed_p1 |= cap_p1
            removed_p2 |= cap_p2

            new_alive1 = new_alive1 and not removed_p1
            new_alive2 = new_alive2 and not removed_p2

            if not new_alive1:
                new_p1 = None
            if not new_alive2:
                new_p2 = None

            successors.append((
                action,
                new_paint,
                new_p1,
                new_p2,
                self.step + 1,
                new_alive1,
                new_alive2
            ))

        return successors




class PaintBattle:

    def __init__(self, map_name):
        self.map_name = map_name
        self.grid = None
        self.avatars_loaded = False
        self.paint = None
        self.start_p1 = None
        self.start_p2 = None
        self.current_state = None
        self.screen = None
        self.font = None
        self.agent1 = None
        self.agent2 = None
        self._load_map(map_name)
        self._load_assets()
        self.current_state = PBState(self.grid.copy(), self.paint.copy(), self.start_p1, self.start_p2, 1)
        self.current_state.paint[self.start_p1] = TEAM1
        self.current_state.paint[self.start_p2] = TEAM2
    def get_initial_state(self):
        return self.current_state
    def is_terminal(self, state: PBState):
        return state.is_terminal()
    def get_scores(self, state: PBState):
        return state.get_scores()
    def env_step(self, current_state: PBState, action, team):
        p1 = current_state.p1
        p2 = current_state.p2
        actor = p1 if team == TEAM1 else p2
        if actor is None:
            return current_state

        dr, dc = DIRECTIONS.get(action, (0, 0))
        target = (actor[0] + dr, actor[1] + dc)
        new_pos = self._get_valid_move_pos(actor, target)

        new_paint = current_state.paint.copy()

        if team == TEAM1 and current_state.alive1 and new_pos is not None:
            new_paint[new_pos] = TEAM1
        if team == TEAM2 and current_state.alive2 and new_pos is not None:
            new_paint[new_pos] = TEAM2

        if team == TEAM1:
            p1_next = new_pos
            p2_next = p2
        else:
            p2_next = new_pos
            p1_next = p1

        removed_p1 = False
        removed_p2 = False
        if p1_next is not None and p2_next is not None and p1_next == p2_next:
            s1 = np.sum(new_paint == TEAM1)
            s2 = np.sum(new_paint == TEAM2)
            if s1 > s2:
                removed_p2 = True
            elif s2 > s1:
                removed_p1 = True
            else:
                removed_p1 = True
                removed_p2 = True

        rem_p1_cap, rem_p2_cap = current_state.capture_logic(new_paint, p1_next, p2_next, team)

        removed_p1 = removed_p1 or rem_p1_cap
        removed_p2 = removed_p2 or rem_p2_cap

        alive1 = current_state.alive1 and not removed_p1
        alive2 = current_state.alive2 and not removed_p2

        final_p1 = p1_next if alive1 else None
        final_p2 = p2_next if alive2 else None

        return PBState(
            current_state.grid,
            new_paint,
            final_p1,
            final_p2,
            current_state.step + 1,
            alive1,
            alive2
        )

    def _prepare_agent_avatar(self, agent):
        if agent is None or agent.avatar_path is None:
            return

        try:
            img = pygame.image.load(agent.avatar_path).convert_alpha()
        except Exception as e:
            print(f"Could not load avatar for {agent.name}: {e}")
            return

        size = CELL_SIZE - 12
        img = pygame.transform.smoothscale(img, (size, size))
        agent.avatar = img

    def _load_map(self, map_name):
        path = f"src/maps/{map_name}.txt"
        if not os.path.exists(path):
            path = f"{map_name}.txt"
            if not os.path.exists(path):
                raise FileNotFoundError(f"map {map_name}.txt Not Found.")

        with open(path, 'r') as f:
            content = f.read().strip()

        content = content.replace('B', 'G')

        lines = [line for line in content.split('\n') if line.strip()]
        self.grid = np.array([list(line) for line in lines])

        R, C = self.grid.shape
        self.paint = np.zeros((R, C), dtype=np.uint8)

        p1 = p2 = None
        for r in range(R):
            for c in range(C):
                ch = self.grid[r, c]
                if ch == '1':
                    p1 = (r, c)
                    self.grid[r, c] = 'G'
                elif ch == '2':
                    p2 = (r, c)
                    self.grid[r, c] = 'G'

        if p1 is None or p2 is None:
            empties = [(r, c) for r in range(R) for c in range(C) if self.grid[r, c] != 'R']
            random.shuffle(empties)
            if p1 is None: p1 = empties[0]
            if p2 is None: p2 = empties[-1]

        self.start_p1, self.start_p2 = p1, p2

    def _load_assets(self):
        if not pygame.get_init():
            pygame.init()
        self.font = pygame.font.Font(None, 26)

    def _init_screen(self):
        if self.screen: return
        R, C = self.grid.shape
        self.screen = pygame.display.set_mode((C * CELL_SIZE, R * CELL_SIZE + HUD_H))
        pygame.display.set_caption("Paint Battle – Minimax Env")
        if not self.avatars_loaded:
            self._prepare_agent_avatar(self.agent1)
            self._prepare_agent_avatar(self.agent2)
            self.avatars_loaded = True

    def _draw(self, p1, p2, step, scores):
        self._init_screen()
        R, C = self.grid.shape
        self.screen.fill((18, 18, 18))
        for r in range(R):
            for c in range(C):
                x, y = c * CELL_SIZE, r * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

                if self.grid[r, c] == 'R':
                    pygame.draw.rect(self.screen, (80, 80, 80), rect)
                else:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)

                team_on_cell = self.current_state.paint[r, c]

                if team_on_cell == TEAM1:
                    agent = getattr(self, "agent1", None)
                    if agent is not None and agent.color is not None:
                        color = agent.color
                    else:
                        color = (60, 140, 255)
                    pygame.draw.rect(self.screen, color, rect.inflate(-8, -8))
                elif team_on_cell == TEAM2:
                    agent = getattr(self, "agent2", None)
                    if agent is not None and agent.color is not None:
                        color = agent.color
                    else:
                        color = (255, 80, 80)
                    pygame.draw.rect(self.screen, color, rect.inflate(-8, -8))

                pygame.draw.rect(self.screen, (25, 25, 25), rect, 1)

        def draw_agent(pos, agent, alive, default_color):
            if pos is None or not alive:
                return

            r, c = pos
            if agent is not None and getattr(agent, "avatar", None) is not None:
                img = agent.avatar
                w, h = img.get_width(), img.get_height()
                x = c * CELL_SIZE + (CELL_SIZE - w) // 2
                y = r * CELL_SIZE + (CELL_SIZE - h) // 2
                self.screen.blit(img, (x, y))
            else:
                cx = c * CELL_SIZE + CELL_SIZE // 2
                cy = r * CELL_SIZE + CELL_SIZE // 2
                color = default_color
                if agent is not None and getattr(agent, "color", None) is not None:
                    color = agent.color
                pygame.draw.circle(self.screen, color, (cx, cy), CELL_SIZE // 3)
                pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), CELL_SIZE // 3, 2)

        draw_agent(p1, getattr(self, "agent1", None), self.current_state.alive1, (0, 90, 255))
        draw_agent(p2, getattr(self, "agent2", None), self.current_state.alive2, (255, 30, 30))

        hud_y = R * CELL_SIZE
        pygame.draw.rect(self.screen, (30, 30, 30), (0, hud_y, C * CELL_SIZE, HUD_H))
        pygame.draw.line(self.screen, (70, 70, 70), (0, hud_y), (C * CELL_SIZE, hud_y), 2)

        name1 = self.agent1.name if self.agent1 else "Player1"
        name2 = self.agent2.name if self.agent2 else "Player2"

        txt = self.font.render(
            f"Turn {step} | {name1}: {scores[TEAM1]}   {name2}: {scores[TEAM2]}",
            True,
            (230, 230, 230)
        )
        self.screen.blit(txt, (10, hud_y + 14))
        pygame.display.flip()

    def _get_valid_move_pos(self, pos, target):
        if pos is None or target is None: return None
        r, c = pos
        nr, nc = target

        if (nr, nc) == pos: return pos
        if abs(nr - r) + abs(nc - c) != 1: return pos
        if not (0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1]): return pos
        if self.grid[nr, nc] == 'R': return pos

        return (nr, nc)


    def env_step_train(self, state, action, team):
        next_state = self.env_step(state, action, team)
        pos = next_state.p1 if team == TEAM1 else next_state.p2
        removed_p1 = (state.alive1 and not next_state.alive1)
        removed_p2 = (state.alive2 and not next_state.alive2)
        return next_state, removed_p1, removed_p2, pos

    def _end(self, winner, scores):
        R, C = self.grid.shape
        surf = pygame.Surface((C * CELL_SIZE, R * CELL_SIZE))
        surf.set_alpha(220)
        surf.fill((0, 0, 0))
        self.screen.blit(surf, (0, 0))

        try:
            big = pygame.font.Font(None, 50)
            t1 = big.render(f"Winner: {winner}", True, (255, 255, 255))
            t2 = self.font.render(f"{self.agent1.name}: {scores[TEAM1]}   {self.agent2.name}: {scores[TEAM2]}", True, (220, 220, 220))

            rect = t1.get_rect(center=(C * CELL_SIZE // 2, R * CELL_SIZE // 2 - 20))
            self.screen.blit(t1, rect)
            self.screen.blit(t2, (rect.x + 20, rect.y + 60))
            pygame.display.flip()
        except pygame.error as e:
            print(f"Could not render end screen: {e}")

    def play(self, agent1, agent2, delay=DELAY_MS):
        self.agent1 = agent1
        self.agent2 = agent2
        agent1.set_team(TEAM1)
        agent2.set_team(TEAM2)

        clock = pygame.time.Clock()
        turn = TEAM1
        while not self.current_state.is_terminal():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            if turn == TEAM1:
                action = agent1.get_action(self.current_state)
            else:
                action = agent2.get_action(self.current_state)

            self.current_state = self.env_step(self.current_state, action, turn)

            s1, s2 = self.current_state.get_scores()
            scores = {TEAM1: s1, TEAM2: s2}
            self._draw(self.current_state.p1, self.current_state.p2, self.current_state.step, scores)

            turn = TEAM2 if turn == TEAM1 else TEAM1

            pygame.time.wait(delay)
            clock.tick(60)

        s1, s2 = self.current_state.get_scores()
        scores = {TEAM1: s1, TEAM2: s2}
        self._draw(self.current_state.p1, self.current_state.p2, self.current_state.step, {TEAM1: s1, TEAM2: s2})
        if not self.current_state.alive1 and self.current_state.alive2:
            winner = self.agent2.name
        elif not self.current_state.alive2 and self.current_state.alive1:
            winner = self.agent1.name
        elif not self.current_state.alive1 and not self.current_state.alive2:
            winner = "Draw"
        else:
            if s1 > s2:
                winner = self.agent1.name
            elif s2 > s1:
                winner = self.agent2.name
            else:
                winner = "Draw"

        self._end(winner, scores)
        pygame.time.wait(3000)
        print(f"Game Over. Winner: {winner}")
