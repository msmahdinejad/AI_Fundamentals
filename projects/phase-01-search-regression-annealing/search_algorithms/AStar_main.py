from env.env import play

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_obstacles_in_rectangle(pos1, pos2, bushes, rocks):
    min_x = min(pos1[0], pos2[0])
    max_x = max(pos1[0], pos2[0])
    min_y = min(pos1[1], pos2[1])
    max_y = max(pos1[1], pos2[1])
    
    bush_count = sum(1 for b in bushes if min_x <= b[0] <= max_x and min_y <= b[1] <= max_y)
    rock_count = sum(1 for r in rocks if min_x <= r[0] <= max_x and min_y <= r[1] <= max_y)
    
    return bush_count, rock_count

def weighted_distance(pos1, pos2, bushes, rocks):
    base_dist = manhattan_distance(pos1, pos2)
    bush_count, rock_count = get_obstacles_in_rectangle(pos1, pos2, bushes, rocks)
    return base_dist + bush_count * 2 + rock_count * 4

def calculate_mst_from_point(start_point, targets, bushes, rocks):
    if len(targets) == 0:
        return 0
    
    if len(targets) == 1:
        return weighted_distance(start_point, targets[0], bushes, rocks)
    
    min_to_first = min(weighted_distance(start_point, t, bushes, rocks) for t in targets)
    
    mst_cost = 0
    nearest = min(targets, key=lambda t: weighted_distance(start_point, t, bushes, rocks))
    visited = [nearest]
    remaining = [t for t in targets if t != nearest]
    
    while remaining:
        min_edge = float('inf')
        closest = None
        
        for v in visited:
            for r in remaining:
                dist = weighted_distance(v, r, bushes, rocks)
                if dist < min_edge:
                    min_edge = dist
                    closest = r
        
        if closest:
            mst_cost += min_edge
            visited.append(closest)
            remaining.remove(closest)
        else:
            break
    
    return min_to_first + mst_cost

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

def astar(initial_state):
    import heapq
    
    enemy_path = initial_state.get_enemy_path()
    cycle_length = len(enemy_path) if enemy_path else 1
    
    initial_enemy_pos = initial_state.get_enemy_position()
    initial_enemy_step = 0
    if initial_enemy_pos and enemy_path:
        try:
            initial_enemy_step = enemy_path.index(initial_enemy_pos)
        except ValueError:
            initial_enemy_step = 0
    
    def heuristic(state, weight=1):
        agent_pos = state.get_agent_position()
        targets = state.get_targets_positions()
        enemy_pos = state.get_enemy_position()
        bushes = state.get_bushes_positions()
        rocks = state.get_rocks_positions()
        
        if len(targets) == 0:
            return 0
        
        h = calculate_mst_from_point(agent_pos, targets, bushes, rocks)
        
        if enemy_pos is not None:
            dist_to_enemy = manhattan_distance(agent_pos, enemy_pos)
            if dist_to_enemy <= 2:
                h += 1000 * (3 - dist_to_enemy)
        
        return h * weight
    
    def search_to_next_target(start_state, base_actions_count):
        frontier = []
        counter = 0
        heapq.heappush(frontier, (heuristic(start_state), counter, 0, start_state, []))
        counter += 1
        
        visited = {}
        nodes_expanded = 0
        
        while frontier:
            f, _, g, current_state, actions = heapq.heappop(frontier)
            
            agent_pos = current_state.get_agent_position()
            targets_tuple = tuple(sorted(current_state.get_targets_positions()))
            enemy_step = (initial_enemy_step + base_actions_count + len(actions)) % cycle_length
            visited_tuple = (agent_pos, targets_tuple, enemy_step)
            
            if visited_tuple in visited and visited[visited_tuple] <= g:
                continue
            
            visited[visited_tuple] = g
            nodes_expanded += 1
            
            if current_state.is_collision_state():
                continue
            
            current_targets = len(current_state.get_targets_positions())
            start_targets = len(start_state.get_targets_positions())
            if current_targets < start_targets:
                return actions, current_state, nodes_expanded
            
            for action, cost, next_state in current_state.get_successors():
                if next_state.is_collision_state():
                    continue
                
                next_agent_pos = next_state.get_agent_position()
                next_targets_tuple = tuple(sorted(next_state.get_targets_positions()))
                next_enemy_step = (initial_enemy_step + base_actions_count + len(actions) + 1) % cycle_length
                next_visited_tuple = (next_agent_pos, next_targets_tuple, next_enemy_step)
                
                new_g = g + cost
                
                if next_visited_tuple not in visited or visited[next_visited_tuple] > new_g:
                    new_f = new_g + heuristic(next_state)
                    heapq.heappush(frontier, (new_f, counter, new_g, next_state, actions + [action]))
                    counter += 1
        
        return None, None, nodes_expanded
    
    total_actions = []
    current_state = initial_state
    total_nodes = 0
    
    initial_targets_count = len(initial_state.get_targets_positions())
    
    for target_num in range(initial_targets_count):
        actions, new_state, nodes = search_to_next_target(current_state, len(total_actions))
        
        if actions is None:
            return []
        
        total_actions.extend(actions)
        current_state = new_state
        total_nodes += nodes
    
    return total_actions

def astar_with_wait(initial_state):
    import heapq
    
    enemy_path = initial_state.get_enemy_path()
    cycle_length = len(enemy_path) if enemy_path else 1
    
    initial_enemy_pos = initial_state.get_enemy_position()
    initial_enemy_step = 0
    if initial_enemy_pos and enemy_path:
        try:
            initial_enemy_step = enemy_path.index(initial_enemy_pos)
        except ValueError:
            initial_enemy_step = 0
    
    def heuristic(state, weight=1):
        agent_pos = state.get_agent_position()
        targets = state.get_targets_positions()
        enemy_pos = state.get_enemy_position()
        bushes = state.get_bushes_positions()
        rocks = state.get_rocks_positions()
        
        if len(targets) == 0:
            return 0
        
        h = calculate_mst_from_point(agent_pos, targets, bushes, rocks)
        
        if enemy_pos is not None:
            dist_to_enemy = manhattan_distance(agent_pos, enemy_pos)
            if dist_to_enemy <= 2:
                h += 1000 * (3 - dist_to_enemy)
        
        return h * weight
    
    def search_to_next_target(start_state, base_actions_count):
        frontier = []
        counter = 0
        heapq.heappush(frontier, (heuristic(start_state), counter, 0, start_state, []))
        counter += 1
        
        visited = {}
        nodes_expanded = 0
        
        while frontier:
            f, _, g, current_state, actions = heapq.heappop(frontier)
            
            agent_pos = current_state.get_agent_position()
            targets_tuple = tuple(sorted(current_state.get_targets_positions()))
            enemy_step = (initial_enemy_step + base_actions_count + len(actions)) % cycle_length
            visited_tuple = (agent_pos, targets_tuple, enemy_step)
            
            if visited_tuple in visited and visited[visited_tuple] <= g:
                continue
            
            visited[visited_tuple] = g
            nodes_expanded += 1
            
            if current_state.is_collision_state():
                continue
            
            current_targets = len(current_state.get_targets_positions())
            start_targets = len(start_state.get_targets_positions())
            if current_targets < start_targets:
                return actions, current_state, nodes_expanded
            
            enemy_pos = current_state.get_enemy_position()
            successors = list(current_state.get_successors())
            all_successors = list(current_state.get_successors(toward_walls=True))
            has_wall = len(all_successors) > len(successors)
            
            if has_wall and enemy_pos is not None:
                if is_enemy_blocking(agent_pos, successors, enemy_pos):
                    for action, cost, next_state in all_successors:
                        if next_state.get_agent_position() == agent_pos:
                            if not next_state.is_collision_state():
                                successors.append((action, cost, next_state))
            
            for action, cost, next_state in successors:
                if next_state.is_collision_state():
                    continue
                
                next_agent_pos = next_state.get_agent_position()
                next_targets_tuple = tuple(sorted(next_state.get_targets_positions()))
                next_enemy_step = (initial_enemy_step + base_actions_count + len(actions) + 1) % cycle_length
                next_visited_tuple = (next_agent_pos, next_targets_tuple, next_enemy_step)
                
                new_g = g + cost
                
                if next_visited_tuple not in visited or visited[next_visited_tuple] > new_g:
                    new_f = new_g + heuristic(next_state)
                    heapq.heappush(frontier, (new_f, counter, new_g, next_state, actions + [action]))
                    counter += 1
        
        return None, None, nodes_expanded
    
    total_actions = []
    current_state = initial_state
    total_nodes = 0
    
    initial_targets_count = len(initial_state.get_targets_positions())
    
    for target_num in range(initial_targets_count):
        actions, new_state, nodes = search_to_next_target(current_state, len(total_actions))
        
        if actions is None:
            return []
        
        total_actions.extend(actions)
        current_state = new_state
        total_nodes += nodes
    
    return total_actions

if __name__ == "__main__":
    play("hard", astar, delay=5000)