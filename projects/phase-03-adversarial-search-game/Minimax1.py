from src.agent import Agent
from src.env import PBState, TEAM1, TEAM2
from collections import deque
import numpy as np


class MinimaxAgent(Agent):
    
    def __init__(self, max_depth=3):
        super().__init__(
            name="Star",
            avatar_path="src/icons/Han.png",
            color=(100, 200, 255)
        )
        self.max_depth = max_depth
        self.my_team = None
        self.enemy_team = None
        self.transposition_table = {}
        self.killer_moves = {}
    
    def set_team(self, team):
        super().set_team(team)
        self.my_team = team
        self.enemy_team = TEAM2 if team == TEAM1 else TEAM1
    
    def state_hash(self, state):
        return hash((
            tuple(state.paint.flatten()),
            state.p1,
            state.p2,
            state.alive1,
            state.alive2
        ))
    
    def quick_evaluate_move(self, state, action, new_state):
        my_pos = state.p1 if self.my_team == TEAM1 else state.p2
        new_my_pos = new_state.p1 if self.my_team == TEAM1 else new_state.p2
        enemy_pos = state.p2 if self.my_team == TEAM1 else state.p1
        
        score = 0
        
        enemy_alive = new_state.alive2 if self.my_team == TEAM1 else new_state.alive1
        if not enemy_alive:
            return 100000
        
        s1, s2 = new_state.get_scores()
        my_score = s1 if self.my_team == TEAM1 else s2
        enemy_score = s2 if self.my_team == TEAM1 else s1
        
        old_s1, old_s2 = state.get_scores()
        old_my_score = old_s1 if self.my_team == TEAM1 else old_s2
        old_enemy_score = old_s2 if self.my_team == TEAM1 else old_s1
        
        score_gain = (my_score - old_my_score) - (enemy_score - old_enemy_score)
        score += score_gain * 200
        
        if new_my_pos and enemy_pos:
            old_distance = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
            new_distance = abs(new_my_pos[0] - enemy_pos[0]) + abs(new_my_pos[1] - enemy_pos[1])
            
            if my_score > enemy_score:
                score += (old_distance - new_distance) * 10
            else:
                score -= (old_distance - new_distance) * 10
        
        return score
    
    def order_moves(self, state, successors, depth):
        if len(successors) <= 1:
            return successors
        
        killer = self.killer_moves.get(depth)
        
        scored_moves = []
        for successor in successors:
            action, paint, p1, p2, step, alive1, alive2 = successor
            new_state = PBState(state.grid, paint, p1, p2, step, alive1, alive2)
            
            if killer and action == killer:
                quick_score = 50000
            else:
                quick_score = self.quick_evaluate_move(state, action, new_state)
            
            scored_moves.append((quick_score, successor))
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored_moves]
    
    def get_action(self, state):
        successors = state.get_successors(self.my_team)
        
        if not successors:
            return 'U'
        
        if len(successors) == 1:
            return successors[0][0]
        
        if len(self.transposition_table) > 100000:
            self.transposition_table.clear()
        
        self.killer_moves.clear()
        
        ordered_successors = self.order_moves(state, successors, self.max_depth)
        
        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for successor in ordered_successors:
            action, paint, p1, p2, step, alive1, alive2 = successor
            
            new_state = PBState(state.grid, paint, p1, p2, step, alive1, alive2)
            value = self.minimax(new_state, self.max_depth - 1, False, alpha, beta)
            
            if value > best_value:
                best_value = value
                best_action = action
            
            alpha = max(alpha, value)
        
        return best_action
    
    def minimax(self, state, depth, is_maximizing, alpha, beta):
        if state.is_terminal() or depth == 0:
            return self.evaluate(state)
        
        state_key = self.state_hash(state)
        original_alpha = alpha
        original_beta = beta
        
        if state_key in self.transposition_table:
            entry = self.transposition_table[state_key]
            if entry['depth'] >= depth:
                if entry['flag'] == 'EXACT':
                    return entry['value']
                elif entry['flag'] == 'LOWERBOUND':
                    alpha = max(alpha, entry['value'])
                elif entry['flag'] == 'UPPERBOUND':
                    beta = min(beta, entry['value'])
                
                if alpha >= beta:
                    return entry['value']
        
        if is_maximizing:
            max_value = float('-inf')
            successors = state.get_successors(self.my_team)
            ordered_successors = self.order_moves(state, successors, depth)
            
            for successor in ordered_successors:
                action, paint, p1, p2, step, alive1, alive2 = successor
                new_state = PBState(state.grid, paint, p1, p2, step, alive1, alive2)
                value = self.minimax(new_state, depth - 1, False, alpha, beta)
                
                if value > max_value:
                    max_value = value
                
                alpha = max(alpha, value)
                
                if beta <= alpha:
                    self.killer_moves[depth] = action
                    break
            
            flag = 'EXACT'
            if max_value <= original_alpha:
                flag = 'UPPERBOUND'
            elif max_value >= original_beta:
                flag = 'LOWERBOUND'
            
            self.transposition_table[state_key] = {
                'value': max_value,
                'depth': depth,
                'flag': flag
            }
            
            return max_value
        else:
            min_value = float('inf')
            successors = state.get_successors(self.enemy_team)
            
            if not successors:
                return self.evaluate(state)
            
            ordered_successors = self.order_moves(state, successors, depth)
            
            for successor in ordered_successors:
                action, paint, p1, p2, step, alive1, alive2 = successor
                new_state = PBState(state.grid, paint, p1, p2, step, alive1, alive2)
                value = self.minimax(new_state, depth - 1, True, alpha, beta)
                
                if value < min_value:
                    min_value = value
                
                beta = min(beta, value)
                
                if beta <= alpha:
                    self.killer_moves[depth] = action
                    break
            
            flag = 'EXACT'
            if min_value <= original_alpha:
                flag = 'UPPERBOUND'
            elif min_value >= original_beta:
                flag = 'LOWERBOUND'
            
            self.transposition_table[state_key] = {
                'value': min_value,
                'depth': depth,
                'flag': flag
            }
            
            return min_value
    
    def bfs_distance(self, state, start, end):
        if start is None or end is None:
            return float('inf')
        
        R, C = state.grid.shape
        q = deque([(start, 0)])
        visited = {start}
        
        while q:
            (r, c), dist = q.popleft()
            
            if (r, c) == end:
                return dist
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < R and 0 <= nc < C:
                    is_obstacle = (state.grid[nr, nc] == 'R') or (state.paint[nr, nc] == self.my_team)
                    
                    if not is_obstacle and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append(((nr, nc), dist + 1))
                        
        return float('inf')

    def count_mobility(self, state, pos, team):
        if pos is None:
            return 0
            
        count = 0
        R, C = state.grid.shape
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < R and 0 <= nc < C:
                if state.grid[nr, nc] != 'R' and state.paint[nr, nc] != team:
                    count += 1
        return count

    def evaluate(self, state):
        my_alive = state.alive1 if self.my_team == TEAM1 else state.alive2
        enemy_alive = state.alive2 if self.my_team == TEAM1 else state.alive1
        my_pos = state.p1 if self.my_team == TEAM1 else state.p2
        enemy_pos = state.p2 if self.my_team == TEAM1 else state.p1
        
        s1, s2 = state.get_scores()
        my_score = s1 if self.my_team == TEAM1 else s2
        enemy_score = s2 if self.my_team == TEAM1 else s1
        
        if not my_alive and not enemy_alive: 
            return -1000
        elif not my_alive: 
            return -100000 + (my_score - enemy_score) * 500
        elif not enemy_alive: 
            return 100000 + (my_score - enemy_score) * 500
        
        if state.is_terminal():
            if my_score > enemy_score: 
                return 100000 + (my_score - enemy_score) * 500
            elif my_score < enemy_score: 
                return -100000 + (my_score - enemy_score) * 500
            else: 
                return -1000
        
        score_diff = my_score - enemy_score
        score = score_diff * 500
        
        my_mobility = self.count_mobility(state, my_pos, self.my_team)
        enemy_mobility = self.count_mobility(state, enemy_pos, self.enemy_team)

        if my_mobility == 0:
            score -= 1000
        if enemy_mobility == 0:
            score += 1000

        if my_pos and enemy_pos:
            distance = self.bfs_distance(state, my_pos, enemy_pos)
            
            if distance == float('inf'):
                distance = 1000 

            if abs(score_diff) != 0:
                score += score_diff / distance * abs(score_diff) * 225
            else:
                current_turn_team = TEAM1 if state.step % 2 == 0 else TEAM2
                is_my_turn = (current_turn_team == self.my_team)
                parity_bonus = 2000 if (is_my_turn == (distance % 2 == 0)) else -2000
                score += parity_bonus + 100 / distance
        
        if my_pos:
            R, C = state.grid.shape
            
            empty_mask = (state.grid != 'R') & (state.paint == 0)
            enemy_mask = (state.grid != 'R') & (state.paint == self.enemy_team)
            
            empty_positions = np.argwhere(empty_mask)
            enemy_positions = np.argwhere(enemy_mask)
            
            if len(empty_positions) > 0:
                distances = np.abs(empty_positions - np.array(my_pos)).sum(axis=1)
                avg_distance = distances.mean()
                score += avg_distance * (-20)
            
            if len(enemy_positions) > 0:
                distances = np.abs(enemy_positions - np.array(my_pos)).sum(axis=1)
                avg_distance = distances.mean()
                if self.my_team == TEAM1:
                    score += avg_distance * (-50)
                else:
                    score += avg_distance * (-400)
        
        return score