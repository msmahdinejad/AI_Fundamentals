import os
import numpy as np
import pygame
import matplotlib.pyplot as plt
from environment import WallE_RL, PygameInit, ACTIONS, THRESHOLDS
import random
import time

# =============================================================================
# STANDARD Q-LEARNING AGENT
# =============================================================================
class QLearningAgent:
    def __init__(self, actions, 
                 learning_rate=0.2,    
                 discount_factor=0.9,     
                 epsilon=1.0, 
                 epsilon_decay=0.9995,    
                 min_epsilon=0.01, 
                 lr_decay=0.99995,        
                 min_lr=0.01):            
        
        self.actions = actions
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}
        
    def get_state_key(self, position, power, trash_state):
        power_level = 0
        for i, threshold in enumerate(THRESHOLDS):
            if power >= threshold:
                power_level = i
            else:
                break
        
        trash_mask = 0
        for i, has_trash in enumerate(trash_state):
            if has_trash:
                trash_mask |= (1 << i)
        
        # 3D State: (X,Y), Power_Index, Trash_Mask_Int
        return (position, power_level, trash_mask)

    def get_q_value(self, state):
        # Lazy initialization
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state, is_training=True):
        # Epsilon-Greedy Strategy
        if is_training and np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        
        q_values = self.get_q_value(state)
        max_q = np.max(q_values)
        # Random choice among max values to break ties randomly
        best_actions = [i for i, val in enumerate(q_values) if val == max_q]
        return np.random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        """
        Standard Q-Learning Update Rule (Online):
        Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.get_q_value(state)[action]
        
        if done:
            target = reward
        else:
            max_next_q = np.max(self.get_q_value(next_state))
            target = reward + self.gamma * max_next_q
        
        # Single Update
        new_q = current_q + self.lr * (target - current_q)
        self.q_table[state][action] = new_q

    def decay_parameters(self, episode, total_episodes):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.lr = max(self.min_lr, self.lr * self.lr_decay)

def calculate_value_difference(current_q_table, prev_q_table):
    if not prev_q_table: return 0.0
    total_diff = 0.0
    all_states = set(current_q_table.keys()) | set(prev_q_table.keys())
    for state in all_states:
        curr = current_q_table.get(state, np.zeros(4))
        prev = prev_q_table.get(state, np.zeros(4))
        total_diff += np.sum(np.abs(curr - prev))
    return total_diff

def create_policy_map(agent, reference_power=5):
    policy_map = np.zeros((8, 8), dtype=int)
    arrow_map = np.empty((8, 8), dtype=str)
    # Reference trash state: All trash present
    trash_state = tuple([True] * 6)
    action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    for x in range(8):
        for y in range(8):
            state_key = agent.get_state_key((x, y), reference_power, trash_state)
            q_values = agent.get_q_value(state_key)
            action = np.argmax(q_values)
            policy_map[y, x] = action
            arrow_map[y, x] = action_symbols[action]
    return policy_map, arrow_map

def plot_convergence(convergence_history, output_path='outputs/convergence.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(convergence_history)), convergence_history, alpha=0.7, color='blue')
    plt.xlabel('Episode'); plt.ylabel('Value Difference'); plt.title('Convergence Analysis')
    plt.grid(True, alpha=0.3); plt.yscale('log')
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_policy_map(policy_map, arrow_map, output_path='outputs/policy_map.png'):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(policy_map, cmap='coolwarm', alpha=0.6)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, arrow_map[i, j], ha="center", va="center", color="black", fontsize=20, fontweight='bold')
    ax.set_title('Policy Map (Power=5)'); plt.colorbar(im, ax=ax)
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def print_report(agent, training_rewards, test_rewards, training_time):
    print("\n" + "="*60)
    print(f"FINAL REPORT (Standard Q-Learning - Fixed)")
    print("="*60)
    print(f"Episodes: {len(training_rewards)}")
    print(f"States Discovered: {len(agent.q_table)}")
    print(f"Training Time: {training_time/60:.2f} min")
    print(f"Final Epsilon: {agent.epsilon:.5f}")
    print(f"Final LR: {agent.lr:.5f}")
    print("-" * 30)
    print(f"Test Mean Score: {np.mean(test_rewards):.2f}")
    
    mean_score = np.mean(test_rewards)
    if mean_score > 1500: grade = "100% (EXCELLENT)"
    elif mean_score > 1200: grade = "70% (GOOD)"
    else: grade = "<50% (FAIL)"
    print(f"Estimated Grade: {grade}")
    print("="*60 + "\n")
    
    with open('outputs/results_summary.txt', 'w') as f:
        f.write(f"Mean Score: {mean_score:.2f}\nGrade: {grade}\nStates: {len(agent.q_table)}")

if __name__ == "__main__":
    if not os.path.exists('outputs'): os.makedirs('outputs')
    
    # 30,000 is enough for Standard Q-Learning
    TRAINING_EPISODES = 30000
    
    print(f"\nSTARTING STANDARD TRAINING ({TRAINING_EPISODES} episodes)...")
    
    env = WallE_RL()
    try: MAX_ACTIONS = env.max_actions if hasattr(env, 'max_actions') else 150
    except: MAX_ACTIONS = 150

    # Initialize Agent with STANDARD parameters
    agent = QLearningAgent(ACTIONS, 
                           learning_rate=0.2,    
                           discount_factor=0.9,   
                           epsilon_decay=0.9995,  
                           min_epsilon=0.01,
                           lr_decay=0.99995)      
    
    training_rewards = []
    convergence_history = []
    prev_q_table = {}
    start_time = time.time()
    
    for episode in range(TRAINING_EPISODES):
        pos, power = env.reset()
        trash_state = tuple([True] * 6)
        state_key = agent.get_state_key(pos, power, trash_state)
        
        done = False; episode_reward = 0; steps = 0
        
        while not done and steps < MAX_ACTIONS:
            # 1. Choose Action
            action = agent.choose_action(state_key, is_training=True)
            
            # 2. Act
            next_pos, reward, next_trash, done, next_pwr = env.step(action)
            steps += 1
            
            # 3. Observe Next State
            if next_trash is not None: trash_state = next_trash
            next_state_key = agent.get_state_key(next_pos, next_pwr, trash_state)
            
            # 4. Update Q-Table (Online)
            agent.learn(state_key, action, reward, next_state_key, done)
            
            # 5. Transition
            state_key = next_state_key
            pos = next_pos; power = next_pwr
            episode_reward += reward
        
        training_rewards.append(episode_reward)
        agent.decay_parameters(episode, TRAINING_EPISODES)
        
        if episode % 1000 == 0:
            if episode > 0:
                diff = calculate_value_difference(agent.q_table, prev_q_table)
                convergence_history.append(diff)
                prev_q_table = {s: v.copy() for s, v in agent.q_table.items()}
            
            recent_avg = np.mean(training_rewards[-1000:]) if len(training_rewards) > 0 else 0
            print(f"Ep {episode:5d} | Rew: {recent_avg:7.1f} | ε: {agent.epsilon:.4f} | States: {len(agent.q_table):5d}")

    total_time = time.time() - start_time
    
    # Visualize
    plot_convergence(convergence_history)
    p_map, a_map = create_policy_map(agent)
    plot_policy_map(p_map, a_map)
    
    # Test
    print("\nTESTING (ε=0)...")
    
    # Enable Pygame window for tests
    screen, clock = PygameInit.initialization()
    
    test_rewards = []
    for test_ep in range(5):
        pos, power = env.reset()
        trash_state = tuple([True] * 6)
        state_key = agent.get_state_key(pos, power, trash_state)
        done = False; total_reward = 0; steps = 0
        
        print(f"Test {test_ep+1}...")
        while not done and steps < MAX_ACTIONS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); exit()
            
            env.render(screen)
            pygame.display.flip()
            clock.tick(20)
            
            action = agent.choose_action(state_key, is_training=False)
            next_pos, reward, next_trash, done, next_pwr = env.step(action)
            if next_trash is not None: trash_state = next_trash
            state_key = agent.get_state_key(next_pos, next_pwr, trash_state)
            total_reward += reward; steps += 1
        test_rewards.append(total_reward)
        print(f"Score: {total_reward}")

    pygame.quit()
    print_report(agent, training_rewards, test_rewards, total_time)