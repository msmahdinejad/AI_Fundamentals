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

def dls_recursive(state, path, depth, depth_limit, visited, initial_enemy_step, cycle_length):
    if state.is_goal_state():
        return path
    
    if depth >= depth_limit:
        return None
    
    current_position = state.get_agent_position()
    current_targets = frozenset(state.get_targets_positions())
    enemy_step = (initial_enemy_step + len(path)) % cycle_length
    visited_tuple = (current_position, current_targets, enemy_step)
    
    if visited_tuple in visited:
        return None
    
    visited.add(visited_tuple)
    
    for action, step_cost, next_state in state.get_successors():
        if next_state.is_collision_state():
            continue
        
        result = dls_recursive(next_state, path + [action], depth + 1, depth_limit, visited, initial_enemy_step, cycle_length)
        if result is not None:
            return result
    
    return None

def ids(initial_state, max_depth=100):
    enemy_path = initial_state.get_enemy_path()
    cycle_length = len(enemy_path) if enemy_path else 1
    
    initial_enemy_pos = initial_state.get_enemy_position()
    initial_enemy_step = 0
    if initial_enemy_pos and enemy_path:
        try:
            initial_enemy_step = enemy_path.index(initial_enemy_pos)
        except ValueError:
            initial_enemy_step = 0
    
    for depth_limit in range(1, max_depth):
        visited = set()
        result = dls_recursive(initial_state, [], 0, depth_limit, visited, initial_enemy_step, cycle_length)
        if result is not None:
            return result
    
    return []

def dls_recursive_with_wait(state, path, depth, depth_limit, visited, initial_enemy_step, cycle_length):
    if state.is_goal_state():
        return path
    
    if depth >= depth_limit:
        return None
    
    current_position = state.get_agent_position()
    current_targets = frozenset(state.get_targets_positions())
    enemy_step = (initial_enemy_step + len(path)) % cycle_length
    visited_tuple = (current_position, current_targets, enemy_step)
    
    if visited_tuple in visited:
        return None
    
    visited.add(visited_tuple)
    
    enemy_position = state.get_enemy_position()
    successors = list(state.get_successors())
    all_successors = list(state.get_successors(toward_walls=True))
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
        
        result = dls_recursive_with_wait(next_state, path + [action], depth + 1, depth_limit, visited, initial_enemy_step, cycle_length)
        if result is not None:
            return result
    
    return None

def ids_with_wait(initial_state, max_depth=100):
    enemy_path = initial_state.get_enemy_path()
    cycle_length = len(enemy_path) if enemy_path else 1
    
    initial_enemy_pos = initial_state.get_enemy_position()
    initial_enemy_step = 0
    if initial_enemy_pos and enemy_path:
        try:
            initial_enemy_step = enemy_path.index(initial_enemy_pos)
        except ValueError:
            initial_enemy_step = 0
    
    for depth_limit in range(1, max_depth):
        visited = set()
        result = dls_recursive_with_wait(initial_state, [], 0, depth_limit, visited, initial_enemy_step, cycle_length)
        if result is not None:
            return result
    
    return []

if __name__ == "__main__":
    play("easy", ids, delay=50)