# Phase 2: MDP and Reinforcement Learning

This phase studies a stochastic WALL-E grid environment.

- `mdp/` implements value iteration, reward-map design, policy precomputation, heat-map visualization, and execution over multiple episodes.
- `unknown_environment/` implements tabular Q-learning with epsilon-greedy exploration and convergence plots.
- `dqn_bonus/` implements a Dueling DQN with replay buffer and target network updates.

Run each script from its own directory because the environments load assets through relative paths.
