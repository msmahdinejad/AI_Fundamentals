import heapq
from env.env import play

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def is_enemy_blocking(agent_pos, successors, enemy_pos):
    if enemy_pos is None:
        return False
    
    if manhattan_distance(agent_pos, enemy_pos) > 3:
        return False
    
    safe_moves = 0
    for action, cost, next_state in successors:
        if not next_state.is_collision_state():
            next_pos = next_state.get_agent_position()
            dist_before = manhattan_distance(agent_pos, enemy_pos)
            dist_after = manhattan_distance(next_pos, enemy_pos)
            if dist_after >= dist_before:
                safe_moves += 1
    
    return safe_moves == 0

def ucs(initial_state):
    priority_queue = []
    counter = 0
    visited = {}
    
    enemy_path = initial_state.get_enemy_path()
    cycle_length = len(enemy_path) if enemy_path else 1
    
    initial_enemy_pos = initial_state.get_enemy_position()
    initial_enemy_step = 0
    if initial_enemy_pos and enemy_path:
        try:
            initial_enemy_step = enemy_path.index(initial_enemy_pos)
        except ValueError:
            initial_enemy_step = 0
    
    start_position = initial_state.get_agent_position()
    targets = frozenset(initial_state.get_targets_positions())
    initial_visited_tuple = (start_position, targets, initial_enemy_step)
    
    heapq.heappush(priority_queue, (0, counter, initial_state, []))
    counter += 1
    visited[initial_visited_tuple] = 0
    
    while priority_queue:
        current_cost, _, current_state, path = heapq.heappop(priority_queue)
        
        current_position = current_state.get_agent_position()
        current_targets = frozenset(current_state.get_targets_positions())
        enemy_step = (initial_enemy_step + len(path)) % cycle_length
        current_visited_tuple = (current_position, current_targets, enemy_step)
        
        if current_cost > visited.get(current_visited_tuple, float('inf')):
            continue
        
        if current_state.is_goal_state():
            return path
        
        for action, step_cost, next_state in current_state.get_successors():
            if next_state.is_collision_state():
                continue
            
            new_cost = current_cost + step_cost
            next_position = next_state.get_agent_position()
            remaining_targets = frozenset(next_state.get_targets_positions())
            next_enemy_step = (initial_enemy_step + len(path) + 1) % cycle_length
            next_visited_tuple = (next_position, remaining_targets, next_enemy_step)
            
            if new_cost < visited.get(next_visited_tuple, float('inf')):
                visited[next_visited_tuple] = new_cost
                heapq.heappush(priority_queue, (new_cost, counter, next_state, path + [action]))
                counter += 1
    
    return []

def ucs_with_wait(initial_state):
    priority_queue = []
    counter = 0
    visited = {}
    
    enemy_path = initial_state.get_enemy_path()
    cycle_length = len(enemy_path) if enemy_path else 1
    
    initial_enemy_pos = initial_state.get_enemy_position()
    initial_enemy_step = 0
    if initial_enemy_pos and enemy_path:
        try:
            initial_enemy_step = enemy_path.index(initial_enemy_pos)
        except ValueError:
            initial_enemy_step = 0
    
    start_position = initial_state.get_agent_position()
    targets = frozenset(initial_state.get_targets_positions())
    initial_state_tuple = (start_position, targets, initial_enemy_step)
    
    heapq.heappush(priority_queue, (0, counter, initial_state, []))
    counter += 1
    visited[initial_state_tuple] = 0
    
    while priority_queue:
        current_cost, _, current_state, path = heapq.heappop(priority_queue)
        
        current_position = current_state.get_agent_position()
        current_targets = frozenset(current_state.get_targets_positions())
        enemy_step = (initial_enemy_step + len(path)) % cycle_length
        current_state_tuple = (current_position, current_targets, enemy_step)
        
        if current_cost > visited.get(current_state_tuple, float('inf')):
            continue
        
        if current_state.is_goal_state():
            return path
        
        enemy_position = current_state.get_enemy_position()
        successors = list(current_state.get_successors())
        all_successors = list(current_state.get_successors(toward_walls=True))
        has_wall = len(all_successors) > len(successors)
        
        if has_wall and enemy_position is not None:
            if is_enemy_blocking(current_position, successors, enemy_position):
                for action, step_cost, next_state in all_successors:
                    if next_state.get_agent_position() == current_position:
                        if not next_state.is_collision_state():
                            successors.append((action, step_cost, next_state))
        
        for action, step_cost, next_state in successors:
            if next_state.is_collision_state():
                continue
            
            new_cost = current_cost + step_cost
            next_position = next_state.get_agent_position()
            remaining_targets = frozenset(next_state.get_targets_positions())
            next_enemy_step = (initial_enemy_step + len(path) + 1) % cycle_length
            next_state_tuple = (next_position, remaining_targets, next_enemy_step)
            
            if new_cost < visited.get(next_state_tuple, float('inf')):
                visited[next_state_tuple] = new_cost
                heapq.heappush(priority_queue, (new_cost, counter, next_state, path + [action]))
                counter += 1
    
    return []

if __name__ == "__main__":
    play("easy", ucs, delay=50)