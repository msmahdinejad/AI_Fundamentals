import numpy as np
import pygame
import random
import copy
import os
import matplotlib
import matplotlib.pyplot as plt
from environment import WallE, PygameInit

# Use non-interactive backend
matplotlib.use('Agg')

# ============================================================================
#                          CONFIG & PARAMETERS
# ============================================================================

GOOD_TRASH_REWARD = 250
GOAL_REWARD = 400
ENEMY_REWARD = -400
DEFAULT_REWARD = -1

# Tuning parameters
SAFETY_BUFFER = 15          
EATABLE_RISK_BUFFER = 10    
PARTIAL_RISK_BUFFER = 15    
COLLECTION_BUFFER = 18       

# Scoring weights
GOAL_DISTANCE_WEIGHT = 0.35
RISK_COST_MULTIPLIER = 3
LEVELING_BONUS = 2.2        

# Risk thresholds
MAX_RISK_FOR_EATABLE = 3
MAX_RISK_FOR_PARTIAL = 2.5
MAX_DIST_FOR_PARTIAL = 5

SAFETY_PADDING_PENALTY = -50 
STUCK_COUNT_TO_ESCAPE = 2
STUCK_HISTORY_SIZE = 12
STUCK_UNIQUE_THRESHOLD = 3

# Reward Map Values
# TRICK: We make GOAL very high so it stays Yellow in Heatmap
REWARD_GOAL_HIGH = 5000  
REWARD_ENEMY = -10000      
REWARD_BUILDING = -5000
REWARD_TRASH_ON_WAY = 50
REWARD_TARGET_TRASH = 1500
REWARD_EATABLE_BASE = 300
REWARD_EATABLE_MULTIPLIER = 2
REWARD_BIG_TRASH_PENALTY = -20


class MySolver(WallE):
    """
    Standard Value Iteration solver with CORRECTED Terminal State Logic.
    """
    
    def __init__(self):
        super().__init__()
        self.convergence_history = []
        self.last_value_table = None
    
    def actions_model(self, reward_map, track_convergence=False):
        """
        Runs Value Iteration.
        FIX: Handles Goal (F) as a terminal state correctly to avoid 'Blue Goal' issue.
        """
        grid_size = self._WallE__grid_size
        
        # Get transition probabilities
        transition_table = self._WallE__calculate_transition_model(
            grid_size,
            self._WallE__probability_dict,
            reward_map
        )
        
        # VI Settings
        gamma = 0.99
        epsilon = 1e-4 
        max_iter = 5000
        
        # Initialize V-table
        V = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        # Initialization
        for r in range(grid_size):
            for c in range(grid_size):
                cell_type = self.grid[r][c][0]
                if cell_type == 'B':
                    V[r][c] = float('-inf')
                elif cell_type == 'F':
                    # Fix: Initialize Goal with its map value
                    V[r][c] = reward_map[r][c] 

        if track_convergence:
            self.convergence_history = []
        
        # Main VI loop
        for iteration in range(max_iter):
            V_new = V.copy()
            delta = 0
            total_diff = 0
            
            for r in range(grid_size):
                for c in range(grid_size):
                    # Skip Buildings and Goal (Goal value is static)
                    cell_type = self.grid[r][c][0]
                    if cell_type == 'B': continue
                    if cell_type == 'F': 
                        # Force Goal to stay at its high value (Terminal State)
                        V_new[r][c] = reward_map[r][c]
                        continue

                    state = (r, c)
                    q_values = []
                    
                    # Check all 4 actions
                    for action in range(4):
                        q = 0.0
                        for prob, next_state, trans_reward in transition_table[state][action]:
                            nr, nc = next_state
                            
                            if self.grid[nr][nc][0] == 'B':
                                next_val = V[r][c]
                                if next_val == float('-inf'): next_val = -100
                                q += prob * (DEFAULT_REWARD + gamma * next_val)
                                
                            elif self.grid[nr][nc][0] == 'F':
                                # --- THE FIX ---
                                # If next state is Goal, we do NOT add gamma * V[Goal] if V[Goal] is already the reward.
                                # To get a smooth gradient (Yellow -> Green), we use the Value of Goal as the target.
                                # V(neighbor) = Step_Cost + Gamma * V(Goal)
                                # This ensures V(Neighbor) < V(Goal), so Goal stays Yellow.
                                
                                goal_val = V[nr][nc]
                                q += prob * (DEFAULT_REWARD + gamma * goal_val)
                                
                            else:
                                # Normal Update
                                next_val = V[nr][nc]
                                if next_val == float('-inf'): next_val = -100
                                q += prob * (trans_reward + gamma * next_val)
                        
                        q_values.append(q)
                    
                    best_value = max(q_values)
                    V_new[r][c] = best_value
                    
                    diff = abs(best_value - V[r][c])
                    delta = max(delta, diff)
                    total_diff += diff
            
            V = V_new
            
            if track_convergence:
                self.convergence_history.append(total_diff)
            
            if delta < epsilon:
                break
        
        self.last_value_table = V.copy()
        
        # Extract Policy
        policy = np.zeros((grid_size, grid_size), dtype=np.int8)
        
        for r in range(grid_size):
            for c in range(grid_size):
                if self.grid[r][c][0] in ['B', 'F']:
                    continue
                
                state = (r, c)
                best_action = 0
                best_q = float('-inf')
                
                for action in range(4):
                    q = 0.0
                    for prob, next_state, trans_reward in transition_table[state][action]:
                        nr, nc = next_state
                        
                        if self.grid[nr][nc][0] == 'B':
                            next_val = V[r][c]
                            if next_val == float('-inf'): next_val = -100
                            q += prob * (DEFAULT_REWARD + gamma * next_val)
                        elif self.grid[nr][nc][0] == 'F':
                            # Apply same logic as above for consistency
                            goal_val = V[nr][nc]
                            q += prob * (DEFAULT_REWARD + gamma * goal_val)
                        else:
                            next_val = V[nr][nc]
                            if next_val == float('-inf'): next_val = -100
                            q += prob * (trans_reward + gamma * next_val)
                    
                    if q > best_q:
                        best_q = q
                        best_action = action
                
                policy[r][c] = best_action
        
        return policy
    
    def get_value_table(self):
        return self.last_value_table
    
    def get_convergence_history(self):
        return self.convergence_history


# ============================================================================
#                            REWARD MAPS
# ============================================================================

def apply_safety_padding(env, reward_map):
    grid_size = 8
    for r in range(grid_size):
        for c in range(grid_size):
            if env.grid[r][c][0] == 'E':
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        if env.grid[nr][nc][0] not in ['B', 'E', 'F']:
                            reward_map[nr][nc] += SAFETY_PADDING_PENALTY

def build_reward_map_for_goal(env):
    """
    Map focused purely on reaching the Goal (F).
    """
    grid_size = 8
    reward_map = [[DEFAULT_REWARD for _ in range(grid_size)] for _ in range(grid_size)]
    
    for r in range(grid_size):
        for c in range(grid_size):
            cell_type = env.grid[r][c][0]
            
            if cell_type == 'F':
                reward_map[r][c] = REWARD_GOAL_HIGH # High value (5000)
            elif cell_type == 'E':
                reward_map[r][c] = REWARD_ENEMY
            elif cell_type == 'B':
                reward_map[r][c] = REWARD_BUILDING
            # No reward for trash when mode is GOAL

    apply_safety_padding(env, reward_map)
    return reward_map


def build_reward_map_for_trash(env, target_pos, current_threshold):
    grid_size = 8
    reward_map = [[DEFAULT_REWARD for _ in range(grid_size)] for _ in range(grid_size)]
    
    for r in range(grid_size):
        for c in range(grid_size):
            cell_type, cell_value = env.grid[r][c]
            
            if (r, c) == target_pos:
                reward_map[r][c] = REWARD_TARGET_TRASH
            elif cell_type == 'F':
                reward_map[r][c] = 200 # Lower attraction when hungry for trash
            elif cell_type == 'E':
                reward_map[r][c] = REWARD_ENEMY
            elif cell_type == 'B':
                reward_map[r][c] = REWARD_BUILDING
            elif cell_type == 'S':
                if cell_value <= current_threshold:
                    reward_map[r][c] = REWARD_EATABLE_BASE + cell_value * REWARD_EATABLE_MULTIPLIER
                else:
                    reward_map[r][c] = REWARD_BIG_TRASH_PENALTY
    
    apply_safety_padding(env, reward_map)
    return reward_map


def build_reward_map_for_collection(env, current_threshold, trash_to_avoid=None):
    grid_size = 8
    reward_map = [[DEFAULT_REWARD for _ in range(grid_size)] for _ in range(grid_size)]
    
    if trash_to_avoid is None: trash_to_avoid = set()
    
    for r in range(grid_size):
        for c in range(grid_size):
            cell_type, cell_value = env.grid[r][c]
            
            if cell_type == 'F':
                reward_map[r][c] = 200
            elif cell_type == 'E':
                reward_map[r][c] = REWARD_ENEMY
            elif cell_type == 'B':
                reward_map[r][c] = REWARD_BUILDING
            elif cell_type == 'S':
                if (r, c) in trash_to_avoid:
                    reward_map[r][c] = -100
                elif cell_value <= current_threshold:
                    reward_map[r][c] = 400 + cell_value * 5
                else:
                    partial_ratio = current_threshold / cell_value
                    reward_map[r][c] = int(100 * partial_ratio)
    
    apply_safety_padding(env, reward_map)
    return reward_map


# ============================================================================
#                            UTILITIES (NO CHANGE)
# ============================================================================

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_current_threshold(agent_value, thresholds):
    current = thresholds[0]
    for t in thresholds:
        if agent_value >= t:
            current = t
    return current

def get_all_trash(env):
    trash = []
    for r in range(8):
        for c in range(8):
            if env.grid[r][c][0] == 'S':
                trash.append(((r, c), env.grid[r][c][1]))
    return trash

def estimate_moves_to_reach(env, start, end, safety_factor=1.8):
    if start == end: return 0
    grid_size = 8
    visited = set()
    queue = [(start, 0)]
    visited.add(start)
    while queue:
        current, dist = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if (nr, nc) not in visited:
                    if env.grid[nr][nc][0] != 'B':
                        if (nr, nc) == end:
                            return int((dist + 1) * safety_factor) + 2
                        visited.add((nr, nc))
                        queue.append(((nr, nc), dist + 1))
    return int(manhattan_distance(start, end) * safety_factor * 2) + 10

def categorize_trash(trash_list, current_threshold):
    eatable = []
    partial = []
    for pos, value in trash_list:
        if value <= current_threshold:
            eatable.append((pos, value))
        else:
            partial.append((pos, value))
    return eatable, partial

def is_trash_accessible(env, trash_pos):
    r, c = trash_pos
    grid_size = 8
    neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    safe_neighbors = 0
    enemy_neighbors = 0
    for nr, nc in neighbors:
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            cell = env.grid[nr][nc][0]
            if cell == 'E':
                enemy_neighbors += 1
            elif cell != 'B':
                safe_neighbors += 1
    return safe_neighbors >= 1 and enemy_neighbors <= 1

def get_trash_risk_score(env, trash_pos):
    r, c = trash_pos
    grid_size = 8
    risk = 0.0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            if env.grid[nr][nc][0] == 'E': risk += 2.0
            elif env.grid[nr][nc][0] == 'B': risk += 0.3
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            if env.grid[nr][nc][0] == 'E': risk += 0.5
    return risk


# ============================================================================
#                            STUCK DETECTION
# ============================================================================

class StuckDetector:
    def __init__(self, history_size=STUCK_HISTORY_SIZE):
        self.position_history = []
        self.history_size = history_size
        self.stuck_count = 0
    
    def update(self, position):
        self.position_history.append(position)
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
    
    def is_stuck(self):
        if len(self.position_history) < self.history_size:
            return False
        unique_positions = len(set(self.position_history))
        if unique_positions <= STUCK_UNIQUE_THRESHOLD:
            self.stuck_count += 1
            return True
        self.stuck_count = 0
        return False
    
    def get_stuck_count(self):
        return self.stuck_count


# ============================================================================
#                            VISUALIZATION
# ============================================================================

def create_visualizations(value_table, convergence_history, env, sub_folder, plot_title):
    base_dir = "training_plots"
    full_dir = os.path.join(base_dir, sub_folder)
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
        
    # --- 1. Heat Map ---
    fig, ax = plt.subplots(figsize=(10, 8))
    display_table = value_table.copy()
    mask = display_table > float('-inf')
    if mask.any():
        min_valid = display_table[mask].min()
        display_table[~mask] = min_valid - 500
    
    im = ax.imshow(display_table, cmap='viridis', aspect='equal')
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xlabel('Col'); ax.set_ylabel('Row')
    
    for r in range(8):
        for c in range(8):
            cell_type = env.grid[r][c][0]
            val = value_table[r, c]
            if val > float('-inf'):
                text = f'{val:.0f}'
                color = 'white' if val < display_table.mean() else 'black'
            else:
                text = 'B'
                color = 'red'
            ax.text(c, r, text, ha='center', va='center', fontsize=7, color=color)
            
            if cell_type == 'F':
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False, edgecolor='gold', linewidth=3))
            elif cell_type == 'E':
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False, edgecolor='red', linewidth=2))
            elif cell_type == 'S':
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False, edgecolor='lime', linewidth=1))
    
    ax.set_title(f'Heat Map: {plot_title}')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(full_dir, "heatmap.png"), dpi=100)
    plt.close()
    
    # --- 2. Convergence ---
    if convergence_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        start_idx = 3
        data = convergence_history[start_idx:] if len(convergence_history) > start_idx else convergence_history
        ax.plot(data, 'b-', linewidth=2)
        ax.set_title(f'Convergence: {plot_title}')
        plt.tight_layout()
        plt.savefig(os.path.join(full_dir, "convergence.png"), dpi=100)
        plt.close()


# ============================================================================
#                        POLICY & MAIN LOGIC
# ============================================================================

def precompute_policies(env, thresholds):
    policies = {}
    print("\n[1/4] Building GOAL policy...")
    goal_reward_map = build_reward_map_for_goal(env)
    policies['GOAL'] = env.actions_model(goal_reward_map, track_convergence=True)
    create_visualizations(env.get_value_table(), env.get_convergence_history(), env, "Goal", "Goal Policy")
    
    trash_list = env.get_trash_points()
    trash_with_values = []
    for pos in trash_list:
        val = env.grid[pos[0]][pos[1]][1]
        trash_with_values.append((pos, val))
    trash_with_values.sort(key=lambda x: x[1])
    
    print(f"\n[2/4] Building policies for {len(trash_with_values)} trash items...")
    for pos, val in trash_with_values:
        for threshold in thresholds:
            if val <= threshold:
                reward_map = build_reward_map_for_trash(env, pos, threshold)
                key = (pos, threshold)
                policies[key] = env.actions_model(reward_map, track_convergence=True)
                folder = os.path.join("Trash", f"Pos_{pos[0]}_{pos[1]}", f"Threshold_{threshold}")
                create_visualizations(env.get_value_table(), env.get_convergence_history(), env, folder, f"Trash {pos}")
    
    print(f"\n[3/4] Building COLLECTION policies...")
    for threshold in thresholds:
        reward_map = build_reward_map_for_collection(env, threshold)
        key = f'COLLECT_{threshold}'
        policies[key] = env.actions_model(reward_map, track_convergence=True)
        folder = os.path.join("Collection", f"Threshold_{threshold}")
        create_visualizations(env.get_value_table(), env.get_convergence_history(), env, folder, f"Collection {threshold}")
    
    return policies, trash_with_values

def find_policy_for_target(policies, target_pos, target_val, thresholds, current_threshold):
    key = (target_pos, current_threshold)
    if key in policies: return policies[key]
    for t in thresholds:
        if target_val <= t:
            key = (target_pos, t)
            if key in policies: return policies[key]
            break
    key = (target_pos, thresholds[-1])
    if key in policies: return policies[key]
    return None

def select_optimal_policy(env, current_pos, policies, thresholds, remaining_moves, initial_trash, goal_pos=(7, 7), stuck_detector=None):
    """
    DECISION LOGIC - FIXED: Forces GOAL if no safe trash is found.
    """
    current_threshold = get_current_threshold(env.agent_value, thresholds)
    current_trash = get_all_trash(env)
    moves_to_goal = estimate_moves_to_reach(env, current_pos, goal_pos, safety_factor=2.0)
    
    # 1. Stuck?
    if stuck_detector and stuck_detector.is_stuck():
        if stuck_detector.get_stuck_count() >= STUCK_COUNT_TO_ESCAPE:
            print("! STUCK -> FORCING GOAL")
            return policies['GOAL'], "STUCK"
    
    # 2. Battery Low?
    if remaining_moves <= moves_to_goal + SAFETY_BUFFER:
        return policies['GOAL'], "LOW_BATTERY"
    
    # 3. All Clean?
    if not current_trash:
        return policies['GOAL'], "ALL_CLEAN"
    
    eatable, partial = categorize_trash(current_trash, current_threshold)
    
    # 4. Try Eatable
    if eatable:
        eatable_sorted = sorted(eatable, key=lambda x: (manhattan_distance(current_pos, x[0]), -x[1]))
        best_target = None
        best_score = float('-inf')
        
        for pos, val in eatable_sorted:
            if not is_trash_accessible(env, pos): continue
            risk = get_trash_risk_score(env, pos)
            if risk >= MAX_RISK_FOR_EATABLE: continue
            
            trip = estimate_moves_to_reach(env, current_pos, pos) + estimate_moves_to_reach(env, pos, goal_pos)
            if remaining_moves <= trip + EATABLE_RISK_BUFFER: continue
            
            reward = GOOD_TRASH_REWARD + val
            cost = manhattan_distance(current_pos, pos) + (GOAL_DISTANCE_WEIGHT * manhattan_distance(pos, goal_pos)) + (risk * RISK_COST_MULTIPLIER)
            score = reward / (cost + 1)
            
            try:
                idx = thresholds.index(current_threshold) + 1
                if idx < len(thresholds) and env.agent_value + val >= thresholds[idx]:
                    score *= LEVELING_BONUS
            except ValueError: pass
            
            if score > best_score:
                best_score = score
                best_target = (pos, val)
        
        if best_target:
            p = find_policy_for_target(policies, best_target[0], best_target[1], thresholds, current_threshold)
            if p is not None: return p, "EAT_TRASH"

    # 5. Try Partial
    if partial and remaining_moves > moves_to_goal + PARTIAL_RISK_BUFFER:
        best_partial = None
        best_p_score = float('-inf')
        for pos, val in partial:
            if not is_trash_accessible(env, pos): continue
            dist = manhattan_distance(current_pos, pos)
            risk = get_trash_risk_score(env, pos)
            if dist > MAX_DIST_FOR_PARTIAL or risk >= MAX_RISK_FOR_PARTIAL: continue
            
            p_reward = (current_threshold / val) * GOOD_TRASH_REWARD
            if p_reward > (dist * 2) + (risk * 5) + 10:
                score = p_reward / (dist + 1 + risk)
                if score > best_p_score:
                    best_p_score = score
                    best_partial = (pos, val)
        
        if best_partial:
            p = find_policy_for_target(policies, best_partial[0], best_partial[1], thresholds, current_threshold)
            if p is not None: return p, "PARTIAL_EAT"

    # 6. Fallback -> GOAL (Replaces the buggy General Collection)
    return policies['GOAL'], "NO_SAFE_TRASH_GO_HOME"


if __name__ == "__main__":
    THRESHOLDS = [5, 10, 20, 30, 40]
    MAX_MOVES = 150
    FPS = 15
    NUM_EPISODES = 5
    
    env = MySolver()
    
    # Goal check
    goal_pos = (7, 7)
    for r in range(8):
        for c in range(8):
            if env.grid[r][c][0] == 'F':
                goal_pos = (r, c); break
    
    policies, initial_trash = precompute_policies(env, THRESHOLDS)
    
    screen, clock = PygameInit.initialization()
    scores = []
    
    for ep in range(NUM_EPISODES):
        print(f"\n--- Episode {ep + 1} ---")
        state = env.reset()
        score = 0; moves = 0
        stuck_detector = StuckDetector()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); exit()
            
            env.render(screen)
            pygame.display.flip()
            clock.tick(FPS)
            
            stuck_detector.update(state)
            remaining = MAX_MOVES - moves
            
            policy, decision = select_optimal_policy(env, state, policies, THRESHOLDS, remaining, initial_trash, goal_pos, stuck_detector)
            
            action = policy[state[0]][state[1]]
            next_state, _, reward, done = env.step(action, THRESHOLDS)
            state = next_state
            score += reward
            moves += 1
            
            if done or moves >= MAX_MOVES:
                scores.append(score)
                trash_left = len(get_all_trash(env))
                print(f"  Result: Score={score:.1f}, Moves={moves}, Trash={8-trash_left}/8, Goal={'YES' if done else 'NO'}")
                break
    
    pygame.quit()