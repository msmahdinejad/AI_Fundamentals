import numpy as np
import pygame
import random
import copy
import os
import matplotlib
import matplotlib.pyplot as plt
from environment import WallE, PygameInit

# Use non-interactive backend to save plots without showing them
matplotlib.use('Agg')

# ============================================================================
#                         CONFIG & PARAMETERS
# ============================================================================

# Environment Rewards
GOOD_TRASH_REWARD = 250
GOAL_REWARD = 400
ENEMY_REWARD = -400
DEFAULT_REWARD = -1

# Tuning parameters for the decision logic
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

# Penalties used in Value Iteration
ENEMY_NEIGHBOR_PENALTY = -300
ENEMY_DIAGONAL_PENALTY = -80

# Stuck detection settings
STUCK_HISTORY_SIZE = 12
STUCK_UNIQUE_THRESHOLD = 3
STUCK_COUNT_TO_ESCAPE = 2

# Reward Map Values for VI
REWARD_GOAL_HIGH = 2000
REWARD_ENEMY = -10000
REWARD_BUILDING = -5000
REWARD_TRASH_ON_WAY = 50
REWARD_TARGET_TRASH = 1500
REWARD_EATABLE_BASE = 300
REWARD_EATABLE_MULTIPLIER = 2
REWARD_BIG_TRASH_PENALTY = -20


class MySolver(WallE):
    """
    Standard Value Iteration solver.
    """
    
    def __init__(self):
        super().__init__()
        self.convergence_history = []
        self.last_value_table = None
    
    def actions_model(self, reward_map, track_convergence=False):
        """
        Runs Value Iteration on the provided reward map.
        Returns the optimal policy grid.
        """
        grid_size = self._WallE__grid_size
        
        # Get transition probabilities from the environment logic
        transition_table = self._WallE__calculate_transition_model(
            grid_size,
            self._WallE__probability_dict,
            reward_map
        )
        
        # VI Settings
        gamma = 0.99
        epsilon = 1e-8
        max_iter = 5000
        
        # Initialize V-table (using -inf for buildings to block them)
        V = np.zeros((grid_size, grid_size), dtype=np.float64)
        for r in range(grid_size):
            for c in range(grid_size):
                if self.grid[r][c][0] == 'B':
                    V[r][c] = float('-inf')
        
        if track_convergence:
            self.convergence_history = []
        
        # Main VI loop
        for iteration in range(max_iter):
            V_new = V.copy()
            delta = 0
            total_diff = 0
            
            for r in range(grid_size):
                for c in range(grid_size):
                    if self.grid[r][c][0] == 'B':
                        continue
                    
                    state = (r, c)
                    q_values = []
                    
                    # Check all 4 actions
                    for action in range(4):
                        q = 0.0
                        for prob, next_state, reward in transition_table[state][action]:
                            nr, nc = next_state
                            
                            # Handle building collision logic (bounce back/stay)
                            if self.grid[nr][nc][0] == 'B':
                                next_val = V[r][c] if V[r][c] > float('-inf') else 0
                                # Fix: Use DEFAULT_REWARD when bouncing off building
                                q += prob * (DEFAULT_REWARD + gamma * next_val)
                            else:
                                next_val = V[nr][nc] if V[nr][nc] > float('-inf') else 0
                                
                                # Add extra penalty for being near enemies to be safe
                                neighbor_penalty = 0
                                # Orthogonal neighbors
                                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    nnr, nnc = nr + dr, nc + dc
                                    if 0 <= nnr < grid_size and 0 <= nnc < grid_size:
                                        if self.grid[nnr][nnc][0] == 'E':
                                            neighbor_penalty += ENEMY_NEIGHBOR_PENALTY
                                
                                # Diagonal neighbors
                                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                                    nnr, nnc = nr + dr, nc + dc
                                    if 0 <= nnr < grid_size and 0 <= nnc < grid_size:
                                        if self.grid[nnr][nnc][0] == 'E':
                                            neighbor_penalty += ENEMY_DIAGONAL_PENALTY
                                
                                q += prob * (reward + neighbor_penalty + gamma * next_val)
                        
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
                # Converged
                if track_convergence and iteration > 0: 
                     # Logic retained, output suppressed as requested
                     pass
                break
        
        self.last_value_table = V.copy()
        
        # Extract Policy
        policy = np.zeros((grid_size, grid_size), dtype=np.int8)
        
        for r in range(grid_size):
            for c in range(grid_size):
                if self.grid[r][c][0] == 'B':
                    continue
                
                state = (r, c)
                best_action = 0
                best_q = float('-inf')
                
                for action in range(4):
                    q = 0.0
                    for prob, next_state, reward in transition_table[state][action]:
                        nr, nc = next_state
                        
                        if self.grid[nr][nc][0] == 'B':
                            next_val = V[r][c] if V[r][c] > float('-inf') else 0
                            # Fix: Use DEFAULT_REWARD when bouncing off building
                            q += prob * (DEFAULT_REWARD + gamma * next_val)
                        else:
                            next_val = V[nr][nc] if V[nr][nc] > float('-inf') else 0
                            
                            # Re-apply neighbor penalties for policy extraction
                            neighbor_penalty = 0
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nnr, nnc = nr + dr, nc + dc
                                if 0 <= nnr < grid_size and 0 <= nnc < grid_size:
                                    if self.grid[nnr][nnc][0] == 'E':
                                        neighbor_penalty += ENEMY_NEIGHBOR_PENALTY
                            
                            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                                nnr, nnc = nr + dr, nc + dc
                                if 0 <= nnr < grid_size and 0 <= nnc < grid_size:
                                    if self.grid[nnr][nnc][0] == 'E':
                                        neighbor_penalty += ENEMY_DIAGONAL_PENALTY
                            
                            q += prob * (reward + neighbor_penalty + gamma * next_val)
                    
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
#                           REWARD MAP BUILDERS
# ============================================================================

def build_reward_map_for_goal(env):
    """Map focused purely on reaching the Goal (F)."""
    grid_size = 8
    reward_map = [[DEFAULT_REWARD for _ in range(grid_size)] for _ in range(grid_size)]
    
    for r in range(grid_size):
        for c in range(grid_size):
            cell_type = env.grid[r][c][0]
            
            if cell_type == 'F':
                reward_map[r][c] = REWARD_GOAL_HIGH
            elif cell_type == 'E':
                reward_map[r][c] = REWARD_ENEMY
            elif cell_type == 'B':
                reward_map[r][c] = REWARD_BUILDING
            elif cell_type == 'S':
                # Small incentive to grab trash if it's strictly on the way
                reward_map[r][c] = REWARD_TRASH_ON_WAY
    
    return reward_map


def build_reward_map_for_trash(env, target_pos, current_threshold):
    """Map focused on a specific trash item."""
    grid_size = 8
    reward_map = [[DEFAULT_REWARD for _ in range(grid_size)] for _ in range(grid_size)]
    
    for r in range(grid_size):
        for c in range(grid_size):
            cell_type, cell_value = env.grid[r][c]
            
            if (r, c) == target_pos:
                reward_map[r][c] = REWARD_TARGET_TRASH
            elif cell_type == 'F':
                # Fix: Goal should have reasonable value as fallback destination
                reward_map[r][c] = 300
            elif cell_type == 'E':
                reward_map[r][c] = REWARD_ENEMY
            elif cell_type == 'B':
                reward_map[r][c] = REWARD_BUILDING
            elif cell_type == 'S':
                if cell_value <= current_threshold:
                    reward_map[r][c] = REWARD_EATABLE_BASE + cell_value * REWARD_EATABLE_MULTIPLIER
                else:
                    reward_map[r][c] = REWARD_BIG_TRASH_PENALTY
    
    return reward_map


def build_reward_map_for_collection(env, current_threshold, trash_to_avoid=None):
    """General map for collecting whatever is edible around."""
    grid_size = 8
    reward_map = [[DEFAULT_REWARD for _ in range(grid_size)] for _ in range(grid_size)]
    
    if trash_to_avoid is None:
        trash_to_avoid = set()
    
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
    
    return reward_map


# ============================================================================
#                           UTILITIES
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
    """Returns list of ((row, col), value) for all trash currently on grid."""
    trash = []
    for r in range(8):
        for c in range(8):
            if env.grid[r][c][0] == 'S':
                trash.append(((r, c), env.grid[r][c][1]))
    return trash

def estimate_moves_to_reach(env, start, end, safety_factor=1.8):
    """Estimates steps needed considering slip probability and obstacles using BFS."""
    if start == end:
        return 0
    
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
    
    # No path found, return large number
    return int(manhattan_distance(start, end) * safety_factor * 2) + 10

def categorize_trash(trash_list, current_threshold):
    """Splits trash into what we can eat vs what we can't."""
    eatable = []
    partial = []
    
    for pos, value in trash_list:
        if value <= current_threshold:
            eatable.append((pos, value))
        else:
            partial.append((pos, value))
    
    return eatable, partial

def is_trash_accessible(env, trash_pos):
    """Checks if a trash is surrounded by walls or enemies."""
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
    
    # Needs at least one safe entry and not too many enemies nearby
    return safe_neighbors >= 1 and enemy_neighbors <= 1

def get_trash_risk_score(env, trash_pos):
    """Calculates how dangerous a trash position is."""
    r, c = trash_pos
    grid_size = 8
    risk = 0.0
    
    # Orthogonal neighbors
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            if env.grid[nr][nc][0] == 'E':
                risk += 2.0
            elif env.grid[nr][nc][0] == 'B':
                risk += 0.3
    
    # Diagonal neighbors
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            if env.grid[nr][nc][0] == 'E':
                risk += 0.5
    
    return risk


# ============================================================================
#                    STUCK DETECTION
# ============================================================================

class StuckDetector:
    """Detects if the robot is oscillating or trapped."""
    
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
        
        # If we barely moved unique tiles in the last X steps, we are likely stuck
        unique_positions = len(set(self.position_history))
        
        if unique_positions <= STUCK_UNIQUE_THRESHOLD:
            self.stuck_count += 1
            return True
        
        self.stuck_count = 0
        return False
    
    def get_stuck_count(self):
        return self.stuck_count


# ============================================================================
#                        VISUALIZATION
# ============================================================================

def create_visualizations(value_table, convergence_history, env, sub_folder, plot_title):
    """
    Saves Heatmap and Convergence graph into structured folders.
    structure: training_plots / sub_folder / [heatmap.png, convergence.png]
    """
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
        # Set buildings to a visibly lower value but not inf
        display_table[~mask] = min_valid - 500
    
    im = ax.imshow(display_table, cmap='viridis', aspect='equal')
    
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xlabel('Col')
    ax.set_ylabel('Row')
    
    # Draw grid values
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
            
            # Highlight special cells
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
    
    # --- 2. Convergence Graph (Linear Scale) ---
    if convergence_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Skip first few iterations to remove the initialization spike
        start_idx = 3
        if len(convergence_history) > start_idx:
            data = convergence_history[start_idx:]
            iterations = range(start_idx + 1, len(convergence_history) + 1)
        else:
            data = convergence_history
            iterations = range(1, len(convergence_history) + 1)
        
        ax.plot(iterations, data, 'b-', linewidth=2, label='Value Difference')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Sum of Differences')
        ax.set_title(f'Convergence: {plot_title}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.01, color='r', linestyle='--', label='Threshold (0.01)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(full_dir, "convergence.png"), dpi=100)
        plt.close()
        
    # Print saved confirmation only for GOAL to keep output clean and match requested format
    if sub_folder == "Goal":
        print(f"   Saved: heat_map_goal_table.png")
        print(f"   Saved: convergence_graph.png")


# ============================================================================
#                        POLICY PRE-COMPUTATION
# ============================================================================

def precompute_policies(env, thresholds):
    """
    Generates all necessary policies (Meta-States) before the game starts.
    """
    policies = {}
    
    print("=" * 60)
    print("                PRE-COMPUTATION PHASE")
    print("=" * 60)
    
    print("\n[1/4] Building GOAL policy...")
    goal_reward_map = build_reward_map_for_goal(env)
    policies['GOAL'] = env.actions_model(goal_reward_map, track_convergence=True)
    
    # Save Goal plots
    create_visualizations(env.get_value_table(), env.get_convergence_history(), env, "Goal", "Goal Policy")
    
    # Get trash info
    trash_list = env.get_trash_points()
    trash_with_values = []
    for pos in trash_list:
        val = env.grid[pos[0]][pos[1]][1]
        trash_with_values.append((pos, val))
    
    trash_with_values.sort(key=lambda x: x[1])
    
    print(f"\n[2/4] Building policies for {len(trash_with_values)} trash items...")
    
    for i, (pos, val) in enumerate(trash_with_values):
        accessible = is_trash_accessible(env, pos)
        risk = get_trash_risk_score(env, pos)
        
        # Fix: Build policy for each threshold level that can collect this trash
        for threshold in thresholds:
            if val <= threshold:
                status_symbol = "✓" if accessible else "⚠"
                print(f"   Trash {i+1}: pos={pos}, value={val}, threshold={threshold}, risk={risk:.1f} {status_symbol}")
                
                reward_map = build_reward_map_for_trash(env, pos, threshold)
                key = (pos, threshold)
                policies[key] = env.actions_model(reward_map, track_convergence=True)
                
                # Organize folders: Trash / Pos_r_c / Threshold_t
                folder_path = os.path.join("Trash", f"Pos_{pos[0]}_{pos[1]}", f"Threshold_{threshold}")
                create_visualizations(env.get_value_table(), env.get_convergence_history(), env, 
                                      folder_path, f"Trash {pos} (Thresh {threshold})")
    
    print(f"\n[3/4] Building COLLECTION policies for each threshold...")
    
    for threshold in thresholds:
        print(f"   Threshold: {threshold}")
        reward_map = build_reward_map_for_collection(env, threshold)
        key = f'COLLECT_{threshold}'
        policies[key] = env.actions_model(reward_map, track_convergence=True)
        
        # Organize folders: Collection / Threshold_t
        folder_path = os.path.join("Collection", f"Threshold_{threshold}")
        create_visualizations(env.get_value_table(), env.get_convergence_history(), env, 
                              folder_path, f"Collection (Thresh {threshold})")
    
    print(f"\n[4/4] Pre-computation complete. Total policies: {len(policies)}")
    
    return policies, trash_with_values


# ============================================================================
#                         POLICY SELECTION
# ============================================================================

def find_policy_for_target(policies, target_pos, target_val, thresholds, current_threshold):
    """Finds best matching policy key based on current threshold."""
    # First try to find policy for current threshold
    key = (target_pos, current_threshold)
    if key in policies:
        return policies[key]
    
    # Then try any threshold that can handle this trash
    for t in thresholds:
        if target_val <= t:
            key = (target_pos, t)
            if key in policies:
                return policies[key]
            break
    
    # Fallback to max threshold
    key = (target_pos, thresholds[-1])
    if key in policies:
        return policies[key]
    
    # Last resort: any policy for this position
    for t in thresholds:
        key = (target_pos, t)
        if key in policies:
            return policies[key]
            
    return None

def select_optimal_policy(env, current_pos, policies, thresholds, 
                          remaining_moves, initial_trash, goal_pos=(7, 7),
                          stuck_detector=None):
    """
    Decides the best move based on the current state.
    """
    current_threshold = get_current_threshold(env.agent_value, thresholds)
    current_trash = get_all_trash(env)
    
    # Fix: Use BFS-based estimation that considers obstacles
    moves_to_goal = estimate_moves_to_reach(env, current_pos, goal_pos, safety_factor=2.0)
    
    # 1. Stuck check
    if stuck_detector and stuck_detector.is_stuck():
        count = stuck_detector.get_stuck_count()
        if count >= STUCK_COUNT_TO_ESCAPE:
            return policies['GOAL'], f"STUCK"
    
    # 2. Emergency check
    if remaining_moves <= moves_to_goal + SAFETY_BUFFER:
        return policies['GOAL'], "LOW_BATTERY"
    
    # 3. Cleaned up everything?
    if not current_trash:
        return policies['GOAL'], "ALL_CLEAN"
    
    eatable, partial = categorize_trash(current_trash, current_threshold)
    
    # 4. Try to eat fully consumable trash
    if eatable:
        eatable_sorted = sorted(eatable, key=lambda x: (manhattan_distance(current_pos, x[0]), -x[1]))
        
        best_target = None
        best_score = float('-inf')
        
        for pos, val in eatable_sorted:
            if not is_trash_accessible(env, pos):
                continue
            
            risk = get_trash_risk_score(env, pos)
            if risk >= MAX_RISK_FOR_EATABLE:
                continue
            
            total_trip = estimate_moves_to_reach(env, current_pos, pos) + \
                         estimate_moves_to_reach(env, pos, goal_pos)
            
            if remaining_moves <= total_trip + EATABLE_RISK_BUFFER:
                continue
            
            reward_pot = GOOD_TRASH_REWARD + val
            dist_to_trash = manhattan_distance(current_pos, pos)
            dist_to_goal = manhattan_distance(pos, goal_pos)
            
            cost = dist_to_trash + (GOAL_DISTANCE_WEIGHT * dist_to_goal) + (risk * RISK_COST_MULTIPLIER)
            score = reward_pot / (cost + 1)
            
            try:
                idx = thresholds.index(current_threshold) + 1
                if idx < len(thresholds):
                    if env.agent_value + val >= thresholds[idx]:
                        score *= LEVELING_BONUS
            except ValueError:
                pass
            
            if score > best_score:
                best_score = score
                best_target = (pos, val)
        
        if best_target:
            t_pos, t_val = best_target
            policy = find_policy_for_target(policies, t_pos, t_val, thresholds, current_threshold)
            if policy is not None:
                return policy, f"EAT_TRASH"
            
            ckey = f'COLLECT_{current_threshold}'
            if ckey in policies:
                return policies[ckey], "GENERAL_COLLECT"
    
    # 5. Try partial trash if we have lots of time
    if partial and remaining_moves > moves_to_goal + PARTIAL_RISK_BUFFER:
        best_partial = None
        best_p_score = float('-inf')
        
        for pos, val in partial:
            if not is_trash_accessible(env, pos): continue
            
            dist = manhattan_distance(current_pos, pos)
            risk = get_trash_risk_score(env, pos)
            
            if dist > MAX_DIST_FOR_PARTIAL or risk >= MAX_RISK_FOR_PARTIAL:
                continue
            
            p_reward = (current_threshold / val) * GOOD_TRASH_REWARD
            
            if p_reward > (dist * 2) + (risk * 5) + 10:
                score = p_reward / (dist + 1 + risk)
                if score > best_p_score:
                    best_p_score = score
                    best_partial = (pos, val)
        
        if best_partial:
            t_pos, t_val = best_partial
            policy = find_policy_for_target(policies, t_pos, t_val, thresholds, current_threshold)
            if policy is not None:
                return policy, f"PARTIAL_EAT"
    
    # 6. Fallback to general collection
    ckey = f'COLLECT_{current_threshold}'
    if ckey in policies and remaining_moves > moves_to_goal + COLLECTION_BUFFER:
        return policies[ckey], "ROAMING"
    
    # 7. Default
    return policies['GOAL'], "GOING_HOME"


# ============================================================================
#                              MAIN
# ============================================================================

if __name__ == "__main__":
    
    THRESHOLDS = [5, 10, 20, 30, 40]
    MAX_MOVES = 150
    FPS = 15
    NUM_EPISODES = 5
    
    # Create Environment
    env = MySolver()
    
    # Goal check
    goal_pos = (7, 7)
    for r in range(8):
        for c in range(8):
            if env.grid[r][c][0] == 'F':
                goal_pos = (r, c)
                break
    
    # Pre-computation
    policies, initial_trash = precompute_policies(env, THRESHOLDS)
    
    print("\n" + "=" * 60)
    print("                 EXECUTION PHASE")
    print("=" * 60)
    
    # Visualization Setup
    screen, clock = PygameInit.initialization()

    # Stats
    scores = []
    total_trash_collected = 0
    total_moves = 0
    
    # Episodes Loop
    for ep in range(NUM_EPISODES):
        
        print(f"\n--- Episode {ep + 1} ---")
        state = env.reset()
        score = 0
        moves = 0
        stuck_detector = StuckDetector()
        running = True

        while running:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        if not os.path.exists("screenshots"):
                            os.makedirs("screenshots")
                        
                        filename = f"screenshots/ep_{ep+1}_move_{moves}.png"
                        pygame.image.save(screen, filename)
                        print(f"Screenshot saved: {filename}")
            
            # Render
            env.render(screen)
            pygame.display.flip()
            clock.tick(FPS)
            
            stuck_detector.update(state)
            remaining = MAX_MOVES - moves
            
            # Logic
            policy, decision = select_optimal_policy(
                env, state, policies, THRESHOLDS,
                remaining, initial_trash, goal_pos,
                stuck_detector
            )
            
            action = policy[state[0]][state[1]]
            
            # Act
            next_state, _, reward, done = env.step(action, THRESHOLDS)
            state = next_state
            score += reward
            moves += 1
            
            if done or moves >= MAX_MOVES:
                scores.append(score)
                remaining_t = get_all_trash(env)
                trash_collected = 8 - len(remaining_t)
                
                total_trash_collected += trash_collected
                total_moves += moves
                
                # Formatted Output
                print(f"\n  Results:")
                print(f"    Score:       {score:.1f}")
                print(f"    Moves:           {moves}")
                print(f"    Power:           {env.agent_value}")
                print(f"    Trash:            {trash_collected}/8")
                print(f"    Goal:          {'✓ YES' if done else '✗ NO'}")
                break
    
    avg_score = sum(scores) / len(scores)
    avg_trash = total_trash_collected / NUM_EPISODES
    avg_moves = total_moves / NUM_EPISODES
    
    print("\n" + "=" * 40)
    print(f"Episode Scores: {[f'{s:.1f}' for s in scores]}")
    print("\nStatistics over 5 episodes:")
    print(f"  Average Score:         {avg_score:.2f}")
    print(f"  Average Trash:            {avg_trash:.2f}/8")
    print(f"  Average Moves:           {avg_moves:.2f}")
    print(f"  Total Score:          {sum(scores):.2f}")
    print("=" * 40)
    
    pygame.quit()