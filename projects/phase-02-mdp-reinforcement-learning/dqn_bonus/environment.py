import numpy as np
import pygame
import random
import copy

# =============================================================================
# CONSTANTS 
# =============================================================================

COLORS = {
    'T': (135, 206, 235),
    'S': (135, 206, 235),
    'E': (135, 206, 235),
    'F': (135, 206, 235),
    'B': (135, 206, 235),
    'TNT': (135, 206, 235),
}
ACTIONS = [0, 1, 2, 3]

GOAL_REWARD = 700
ENEMY_REWARD = -500
DEFAULT_REWARD = -1
TNT_REWARD = -2000
ACTION_TAKEN_REWARD = -1000

GOOD_TRASH_REWARD = 400
THRESHOLDS = [5, 10]
TRASH_VALUES = [5, 5, 10]


ENEMIES = 2
BUILDINGS = 2
TNTs = 0
MAX_ACTIONS = 150


# =============================================================================
# PygameInit
# =============================================================================
class PygameInit:
    @classmethod
    def initialization(cls):
        grid_size = 5
        tile_size = 80

        pygame.init()

        screen = pygame.display.set_mode((grid_size * tile_size, grid_size * tile_size))
        pygame.display.set_caption("WALL-E (RL Version)")
        clock = pygame.time.Clock()

        return screen, clock


# =============================================================================
# WallE_RL
# =============================================================================
class WallE_RL:
    def __init__(self):
        self.__grid_size = 5
        self.__tile_size = 80
        self.__num_enemies = ENEMIES
        self.__num_buildings = BUILDINGS
        self.__num_tnts = TNTs
        self.__trash_values_list = TRASH_VALUES
        self.__num_trash = len(self.__trash_values_list)

        self.reward = 0
        self.done = False
        self.agent_power = 5
        self.__trash_data = {}

        self.__base_grid = self.__generate_grid()
        self.__grid = copy.deepcopy(self.__base_grid)
        self.__probability_dict = self.__generate_probability_dict()

        self.__agent_pos = (0, 0)
        self.__max_actions = MAX_ACTIONS
        self.__actions_taken = 0

        self.__load_images()

    def __load_images(self):
        try:
            img = pygame.image.load("Env/icons/snow.png")
            self.__ground_image = pygame.transform.smoothscale(img, (self.__tile_size, int(self.__tile_size * 2.05)))
        except Exception:
            self.__ground_image = pygame.Surface((self.__tile_size, self.__tile_size))
            self.__ground_image.fill((135, 206, 235))

        try:
            img = pygame.image.load("Env/icons/wall-e.png")
            self.__agent_image = pygame.transform.smoothscale(img, (self.__tile_size * 1.1, self.__tile_size * 1.1))
        except Exception:
            self.__agent_image = pygame.Surface((self.__tile_size, self.__tile_size))
            self.__agent_image.fill((255, 0, 0))

        try:
            self.__trash_image = pygame.image.load('Env/icons/trash.png')
            self.__trash_image = pygame.transform.smoothscale(self.__trash_image,
                                                              (self.__tile_size * 1.1, self.__tile_size * 1.1))
        except Exception:
            self.__trash_image = pygame.Surface((self.__tile_size, self.__tile_size), pygame.SRCALPHA)
            self.__trash_image.fill((0, 255, 0))

        self.__value_images = {}
        overlay_scale = 0.4
        overlay_size = (int(self.__tile_size * overlay_scale), int(self.__tile_size * overlay_scale))
        for v in [5, 10, 20, 30, 40]:
            try:
                img = pygame.image.load(f'Env/icons/{v}.png')
                img = pygame.transform.smoothscale(img, overlay_size)
                offset = ((self.__tile_size - overlay_size[0]) // 2, (self.__tile_size - overlay_size[1]) // 2)
                self.__value_images[v] = (img, offset)
            except Exception:
                self.__value_images[v] = None

        try:
            self.__flower_image = pygame.image.load('Env/icons/flower.png')
            self.__flower_image = pygame.transform.smoothscale(self.__flower_image,
                                                               (self.__tile_size, self.__tile_size))
        except Exception:
            self.__flower_image = pygame.Surface((self.__tile_size, self.__tile_size))
            self.__flower_image.fill((135, 206, 235))

        try:
            self.__enemy_image = pygame.image.load('Env/icons/enemy.png')
            self.__enemy_image = pygame.transform.smoothscale(self.__enemy_image, (self.__tile_size, self.__tile_size))
        except Exception:
            self.__enemy_image = pygame.Surface((self.__tile_size, self.__tile_size))
            self.__enemy_image.fill((135, 206, 235))

        try:
            self.__building_image = pygame.image.load('Env/icons/building.png')
            self.__building_image = pygame.transform.smoothscale(self.__building_image,
                                                                 (self.__tile_size, int(self.__tile_size * 1.6)))
        except Exception:
            self.__building_image = pygame.Surface((self.__tile_size, self.__tile_size))
            self.__building_image.fill((135, 206, 235))

        try:
            img = pygame.image.load('Env/icons/TNT.png')
            self.__tnt_image = pygame.transform.smoothscale(img, (self.__tile_size, self.__tile_size))
        except Exception:
            self.__tnt_image = pygame.Surface((self.__tile_size, self.__tile_size))
            self.__tnt_image.fill((255, 100, 0))

    def __generate_grid(self):
        grid = [[('T', 0) for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]

        while True:
            grid = [[('T', 0) for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]
            filled_spaces = [(0, 0), (self.__grid_size - 1, self.__grid_size - 1)]
            self.__trash_data = {}
            trash_to_place = self.__trash_values_list.copy()
            random.shuffle(trash_to_place)

            for val in trash_to_place:
                while True:
                    r, c = random.randint(0, self.__grid_size - 1), random.randint(0, self.__grid_size - 1)
                    if (r, c) not in filled_spaces:
                        grid[r][c] = ('S', val)
                        filled_spaces.append((r, c))
                        self.__trash_data[(r, c)] = val
                        break

            for _ in range(self.__num_enemies):
                while True:
                    r, c = random.randint(0, self.__grid_size - 1), random.randint(0, self.__grid_size - 1)
                    if (r, c) not in filled_spaces:
                        grid[r][c] = ('E', 0)
                        filled_spaces.append((r, c))
                        break

            for _ in range(self.__num_buildings):
                while True:
                    r, c = random.randint(0, self.__grid_size - 1), random.randint(0, self.__grid_size - 1)
                    if (r, c) not in filled_spaces:
                        grid[r][c] = ('B', 0)
                        filled_spaces.append((r, c))
                        break

            for _ in range(self.__num_tnts):
                while True:
                    r, c = random.randint(0, self.__grid_size - 1), random.randint(0, self.__grid_size - 1)
                    if (r, c) not in filled_spaces:
                        grid[r][c] = ('TNT', 0)
                        filled_spaces.append((r, c))
                        break

            grid[self.__grid_size - 1][self.__grid_size - 1] = ('F', 0)

            if WallE_RL.__is_path_exists(grid=grid, start=(0, 0), goal=(4,4)):
                break

        self.__trash_data = dict(sorted(self.__trash_data.items()))
        return grid

    def reset(self):
        self.__grid = copy.deepcopy(self.__base_grid)
        self.__agent_pos = (0, 0)
        self.agent_power = 5
        self.done = False
        self.__actions_taken = 0
        return self.__agent_pos, self.agent_power

    def step(self, action):
        actions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        neighbors = {
            0: [2, 3],
            1: [2, 3],
            2: [0, 1],
            3: [0, 1]
        }

        intended_probability = self.__probability_dict[self.__agent_pos][action]['intended']
        neighbors_probability = self.__probability_dict[self.__agent_pos][action]['neighbor']

        prob_dist = [0, 0, 0, 0]
        prob_dist[action] = intended_probability
        prob_dist[neighbors[action][0]] = neighbors_probability
        prob_dist[neighbors[action][1]] = neighbors_probability

        chosen_action = np.random.choice([0, 1, 2, 3], p=prob_dist)

        dx, dy = actions[chosen_action]
        new_row = self.__agent_pos[0] + dx
        new_col = self.__agent_pos[1] + dy

        if (0 <= new_row < self.__grid_size and 0 <= new_col < self.__grid_size and
                self.__grid[new_row][new_col][0] != 'B'):
            self.__agent_pos = (new_row, new_col)

        self.__actions_taken += 1
        current_tile_type, current_tile_value = self.__grid[self.__agent_pos[0]][self.__agent_pos[1]]
        reward = DEFAULT_REWARD

        if current_tile_type == 'E':
            reward = ENEMY_REWARD
            self.__grid[self.__agent_pos[0]][self.__agent_pos[1]] = ('T', 0)

        elif current_tile_type == 'S':
            trash_value = current_tile_value

            current_threshold = THRESHOLDS[0]
            for t in THRESHOLDS:
                if self.agent_power >= t:
                    current_threshold = t

            if trash_value > current_threshold:
                reward = ((current_threshold / trash_value) * GOOD_TRASH_REWARD) - GOOD_TRASH_REWARD
                self.__grid[self.__agent_pos[0]][self.__agent_pos[1]] = ('T', 0)
            else:
                self.agent_power += trash_value
                reward = GOOD_TRASH_REWARD + trash_value
                self.__grid[self.__agent_pos[0]][self.__agent_pos[1]] = ('T', 0)

        elif current_tile_type == 'F':
            reward = GOAL_REWARD
            self.done = True

        elif current_tile_type == 'TNT':
            reward = TNT_REWARD
            self.done = True

        elif current_tile_type == 'T':
            reward = DEFAULT_REWARD

        if self.__actions_taken >= MAX_ACTIONS:
            reward = ACTION_TAKEN_REWARD
            self.done = True

        next_state = self.__agent_pos
        is_terminated = self.done
        self.reward = reward
        trash_state = self.__get_trash_state()

        return next_state, self.reward, trash_state, is_terminated, self.agent_power

    def render(self, screen):
        screen.fill((0, 0, 0))

        rows, cols = self.__grid_size, self.__grid_size
        agent_r, agent_c = self.__agent_pos

        y_offset_ground = self.__tile_size * 0.6
        y_offset_building = self.__tile_size * 0.5
        y_offset_flower = self.__tile_size * 0.1
        y_offset_trash = self.__tile_size * 0.2
        y_offset_enemy = self.__tile_size * 0.2
        y_offset_agent = self.__tile_size * 0.2
        y_offset_tnt = self.__tile_size * 0.2

        scale_map = {
            5: 0.8,
            10: 0.9,
            20: 1,
            30: 1.1,
            40: 1.2
        }

        for r in range(rows):
            for c in range(cols):
                base_x = c * self.__tile_size
                base_y = r * self.__tile_size

                tile_char, tile_val = self.__grid[r][c]

                ground_dest_y = base_y - y_offset_ground
                ground_area = None
                ground_img_h = self.__ground_image.get_height()
                if ground_dest_y < 0:
                    clip_amount = abs(ground_dest_y)
                    ground_dest_y = 0
                    ground_area = pygame.Rect(0, clip_amount, self.__tile_size, ground_img_h - clip_amount)
                screen.blit(self.__ground_image, (base_x, ground_dest_y), area=ground_area)

                if tile_char == 'B':
                    building_dest_y = base_y - y_offset_building
                    building_area = None
                    building_img_h = self.__building_image.get_height()
                    if building_dest_y < 0:
                        clip_amount = abs(building_dest_y)
                        building_dest_y = 0
                        building_area = pygame.Rect(0, clip_amount, self.__tile_size, building_img_h - clip_amount)
                    screen.blit(self.__building_image, (base_x, building_dest_y), area=building_area)

                if tile_char == 'S':
                    scale_factor = scale_map.get(tile_val, 1.0)
                    new_size = int(self.__tile_size * scale_factor)

                    scaled_trash = pygame.transform.smoothscale(self.__trash_image, (new_size, new_size))

                    pos_x = base_x + (self.__tile_size - new_size) // 2
                    pos_y = (base_y - y_offset_trash) + (self.__tile_size - new_size)

                    x_nudge = 0
                    y_nudge = 0
                    if scale_factor > 1.0:
                        x_nudge = int(self.__tile_size * 0.1)
                        y_nudge = int(self.__tile_size * 0.1)

                    pos_x += x_nudge
                    pos_y += y_nudge

                    trash_dest_y = pos_y
                    trash_area = None
                    if trash_dest_y < 0:
                        clip_amount = abs(trash_dest_y)
                        trash_dest_y = 0
                        trash_area = pygame.Rect(0, clip_amount, new_size, new_size - clip_amount)
                    screen.blit(scaled_trash, (pos_x, trash_dest_y), area=trash_area)

                    val_img = self.__value_images.get(tile_val)
                    if val_img:
                        img_surf, offset = val_img
                        original_num_width, original_num_height = img_surf.get_size()
                        new_num_width = int(original_num_width * scale_factor)
                        new_num_height = int(original_num_height * scale_factor)
                        scaled_num_surf = pygame.transform.smoothscale(img_surf, (new_num_width, new_num_height))

                        num_pos_x = pos_x + (new_size - new_num_width) // 2.5
                        num_pos_y = pos_y + (new_size - new_num_height) // 2.5

                        num_dest_y = num_pos_y
                        num_area = None
                        if num_dest_y < 0:
                            clip_amount = abs(num_dest_y)
                            num_dest_y = 0
                            num_area = pygame.Rect(0, clip_amount, new_num_width, new_num_height - clip_amount)
                        screen.blit(scaled_num_surf, (num_pos_x, num_dest_y), area=num_area)

                if tile_char == 'F':
                    flower_dest_y = base_y - y_offset_flower
                    flower_area = None
                    flower_img_h = self.__flower_image.get_height()
                    if flower_dest_y < 0:
                        clip_amount = abs(flower_dest_y)
                        flower_dest_y = 0
                        flower_area = pygame.Rect(0, clip_amount, self.__tile_size, flower_img_h - clip_amount)
                    screen.blit(self.__flower_image, (base_x, flower_dest_y), area=flower_area)

                if tile_char == 'E':
                    enemy_dest_y = base_y - y_offset_enemy
                    enemy_area = None
                    enemy_img_h = self.__enemy_image.get_height()
                    if enemy_dest_y < 0:
                        clip_amount = abs(enemy_dest_y)
                        enemy_dest_y = 0
                        enemy_area = pygame.Rect(0, clip_amount, self.__tile_size, enemy_img_h - clip_amount)
                    screen.blit(self.__enemy_image, (base_x, enemy_dest_y), area=enemy_area)

                if tile_char == 'TNT':
                    tnt_dest_y = base_y - y_offset_tnt
                    tnt_area = None
                    tnt_img_h = self.__tnt_image.get_height()
                    if tnt_dest_y < 0:
                        clip_amount = abs(tnt_dest_y)
                        tnt_dest_y = 0
                        tnt_area = pygame.Rect(0, clip_amount, self.__tile_size, tnt_img_h - clip_amount)
                    screen.blit(self.__tnt_image, (base_x, tnt_dest_y), area=tnt_area)

                if r == agent_r and c == agent_c:
                    agent_dest_y = base_y - y_offset_agent
                    agent_area = None
                    agent_img_h = self.__agent_image.get_height()
                    if agent_dest_y < 0:
                        clip_amount = abs(agent_dest_y)
                        agent_dest_y = 0
                        agent_area = pygame.Rect(0, clip_amount, self.__tile_size, agent_img_h - clip_amount)
                    screen.blit(self.__agent_image, (base_x, agent_dest_y), area=agent_area)

    @classmethod
    def __is_path_exists(cls, grid, start, goal):
        grid_size = len(grid)
        visited = set()

        def dfs(x, y):
            if (x, y) == goal:
                return True
            visited.add((x, y))

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < grid_size and 0 <= ny < grid_size and
                        (nx, ny) not in visited and grid[nx][ny][0] != 'B'):
                    if dfs(nx, ny):
                        return True
            return False

        return dfs(start[0], start[1])

    def __generate_probability_dict(self):
        probability_dict = {}

        for row in range(self.__grid_size):
            for col in range(self.__grid_size):
                state = (row, col)
                probability_dict[state] = {}

                for action in range(4):
                    intended_prob = random.uniform(0.55, 0.90)
                    remaining_prob = 1 - intended_prob
                    neighbor_prob = remaining_prob / 2

                    probability_dict[state][action] = {
                        'intended': intended_prob,
                        'neighbor': neighbor_prob}
        return probability_dict

    def __get_trash_state(self):
        grid = self.__grid
        states = [False for _ in range(self.__num_trash)]

        original_trash_positions = list(self.__trash_data.keys())

        for i, trash_coordinate in enumerate(original_trash_positions):
            x, y = trash_coordinate[0], trash_coordinate[1]
            if grid[x][y][0] == 'S':
                states[i] = True
        return states