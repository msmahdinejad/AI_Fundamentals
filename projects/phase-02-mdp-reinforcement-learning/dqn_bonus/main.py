import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from environment import WallE_RL, PygameInit, ACTIONS

# Dueling DQN Architecture
class DuelingDQN(nn.Module):
    # state size -> 32 features
    # action size -> 4 (u/d/r/l)
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()

        # Feature extraction layers
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # Value stream (V)
        # Output: A single scalar V(s) representing the value of the state
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream (A)
        # Output: A vector of size 'action_size' representing A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Parallel calculation of Value and Advantage
        val = self.value_stream(x)
        adv = self.advantage_stream(x)

        # Combine V and A to get Q
        # Formula: Q(s,a) = V(s) + (A(s,a) - Mean(A))
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        
        return q_values

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        
        # Copy weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.0003
        self.batch_size = 64
        self.target_update = 10
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(10000)

        # Using SmoothL1Loss (Huber Loss)
        self.criterion = nn.SmoothL1Loss()
        
    def get_state_representation(self, state, agent_power, trash_state):
        x, y = state
        
        # Normalize inputs
        pos_features = np.array([x / 4.0, y / 4.0])
        
        # One-hot position (5x5 grid = 25 features)
        agent_pos_onehot = np.zeros(25)
        agent_pos_onehot[y * 5 + x] = 1
        
        power_norm = agent_power / 40.0

        # Trash Existence Flags (3 features)
        trash_binary = np.array([1 if t else 0 for t in trash_state])

        # Manhattan Distance to Goal Normalized
        goal_distance = (abs(x - 4) + abs(y - 4)) / 8.0
        
        # Final state vector (32 features)
        state_vector = np.concatenate([
            pos_features,
            agent_pos_onehot,
            [power_norm],
            trash_binary,
            [goal_distance]
        ])
        
        return state_vector
    
    def select_action(self, state):
        # Random action (Explore)
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        
        # Best action (Exploit)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return ACTIONS[q_values.argmax().item()]
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        # Get batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current Q
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Double DQN target
        with torch.no_grad():
            # Action selection: Policy Net
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            # Action eval: Target Net
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss and optimize
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

if __name__ == "__main__":
    NUM_TRAIN_EPISODES = 1000
    NUM_TEST_EPISODES = 5
    MAX_ACTIONS_PER_EPISODE = 200
    
    env = WallE_RL()
    screen, clock = PygameInit.initialization()
    
    state_size = 32
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size)
    
    episode_rewards = []
    losses = []
    avg_rewards = []
    
    print(f"Training on {agent.device}...")
    
    for episode in range(NUM_TRAIN_EPISODES):
        state, agent_power = env.reset()
        trash_state = [True] * 3 
        state_vector = agent.get_state_representation(state, agent_power, trash_state)
        
        total_reward = 0
        episode_loss = []
        
        for step in range(MAX_ACTIONS_PER_EPISODE):
            action = agent.select_action(state_vector)
            next_state, reward, next_trash_state, done, next_agent_power = env.step(action)
            next_state_vector = agent.get_state_representation(next_state, next_agent_power, next_trash_state)
            
            agent.memory.push(state_vector, action, reward, next_state_vector, float(done))
            
            loss = agent.train_step()
            if loss > 0:
                episode_loss.append(loss)
            
            state_vector = next_state_vector
            total_reward += reward
            trash_state = next_trash_state
            
            if done:
                break
        
        # Update target net
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        agent.decay_epsilon()
        
        episode_rewards.append(total_reward)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        if episode >= 99:
            avg_rew = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_rew)
        
        if (episode + 1) % 100 == 0:
            avg_rew = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1} | Reward: {avg_rew:.2f} | Eps: {agent.epsilon:.3f}")
    
    print("Done!")
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    # 1. Rewards
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    if avg_rewards:
        plt.plot(range(99, NUM_TRAIN_EPISODES), avg_rewards, label='Avg Reward (100 eps)', linewidth=2, color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards (Dueling DDQN)')
    plt.legend()
    plt.grid(True)
    
    # 2. Loss
    plt.subplot(1, 3, 2)
    if losses:
        plt.plot(losses, color='red', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Loss (SmoothL1)')
        plt.title('Training Loss')
        plt.grid(True)
    
    # 3. Epsilon
    plt.subplot(1, 3, 3)
    epsilons = [agent.epsilon_min + (1.0 - agent.epsilon_min) * (agent.epsilon_decay ** i) for i in range(NUM_TRAIN_EPISODES)]
    plt.plot(epsilons, color='green')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Decay')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dqn_training_metrics.png')
    print("Metrics saved to 'dqn_training_metrics.png'")
    # plt.show()
    plt.close()
    
    # Testing
    print(f"\nStarting evaluation for {NUM_TEST_EPISODES} episodes (Greedy)...")
    agent.epsilon = 0.0  
    
    test_rewards = []
    FPS = 10
    
    for episode in range(NUM_TEST_EPISODES):
        state, agent_power = env.reset()
        trash_state = [True] * 3
        state_vector = agent.get_state_representation(state, agent_power, trash_state)
        total_reward = 0
        
        for step in range(MAX_ACTIONS_PER_EPISODE):
            # Visualization
            env.render(screen)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); exit()
            
            # Action (Greedy)
            action = agent.select_action(state_vector)
            
            # Step
            next_state, reward, next_trash_state, done, next_agent_power = env.step(action)
            
            state_vector = agent.get_state_representation(next_state, next_agent_power, next_trash_state)
            total_reward += reward
            
            if done:
                break
                
            pygame.display.flip()
            clock.tick(FPS)
        
        test_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Reward = {total_reward:.0f}")
    
    print("-" * 50)
    mean_score = np.mean(test_rewards)
    print(f"MEAN TEST REWARD: {mean_score:.2f}")
    
    # Save Model
    torch.save(agent.policy_net.state_dict(), 'dueling_ddqn_model.pth')
    print("Model saved to 'dueling_ddqn_model.pth'")
    
    pygame.quit()