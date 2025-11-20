"""
Deep Nash Q-Learning Algorithm for Competitive Snake Game

Uses neural networks to approximate Q-values for joint actions
instead of tabular Q-tables.
"""

import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import random


class DQNetwork(nn.Module):
    """
    Deep Q-Network for Nash Q-Learning
    
    Network architecture:
    - Input: state features (18 dimensions per agent)
    - Hidden layers: fully connected with ReLU
    - Output: Q-values for all joint action pairs (3x3 = 9)
    """
    
    def __init__(self, state_size=18, action_size=3, hidden_size=128):
        super(DQNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.joint_action_size = action_size * action_size  # 9 joint actions
        
        # Network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, self.joint_action_size)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: Tensor of shape (batch_size, state_size)
            
        Returns:
            Q-values of shape (batch_size, action_size, action_size)
        """
        # Use batch norm only if batch size > 1
        if state.size(0) > 1:
            x = F.relu(self.bn1(self.fc1(state)))
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            # Skip batch norm for single samples
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        
        # Reshape to (batch_size, action_size, action_size)
        batch_size = state.shape[0]
        q_values = q_values.view(batch_size, self.action_size, self.action_size)
        
        return q_values


class ReplayBuffer:
    """Experience replay buffer for training stability"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, my_action, opp_action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, my_action, opp_action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        states, my_actions, opp_actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(my_actions), np.array(opp_actions),
                np.array(rewards), np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DeepNashQLearningAgent:
    """
    Deep Nash Q-Learning agent using neural networks
    """
    
    def __init__(self, agent_id, state_size=18, action_size=3,
                 learning_rate=0.001, discount_factor=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_capacity=50000, batch_size=64,
                 target_update_freq=100, hidden_size=128):
        """
        Initialize Deep Nash Q-Learning agent
        
        Args:
            agent_id: 1 or 2, identifying which agent this is
            state_size: Dimension of state vector
            action_size: Number of actions per agent
            learning_rate: Learning rate for network optimization
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_capacity: Size of replay buffer
            batch_size: Mini-batch size for training
            target_update_freq: Steps between target network updates
            hidden_size: Size of hidden layers
        """
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.hidden_size = hidden_size
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent {agent_id} using device: {self.device}")
        
        # Q-Networks
        self.policy_net = DQNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity)
        
        # Nash equilibrium strategies cache
        self.nash_strategies = {}
        
        # Training statistics
        self.total_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self.steps_done = 0
        
    def state_to_tensor(self, state):
        """Convert state to PyTorch tensor"""
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def get_q_values(self, state, use_target=False):
        """
        Get Q-values for a state
        
        Args:
            state: State vector
            use_target: Whether to use target network
            
        Returns:
            Q-values matrix (action_size x action_size)
        """
        state_tensor = self.state_to_tensor(state)
        
        with torch.no_grad():
            if use_target:
                q_values = self.target_net(state_tensor)
            else:
                q_values = self.policy_net(state_tensor)
        
        return q_values.squeeze(0).cpu().numpy()
    
    def get_action(self, state, training=True):
        """
        Select action using epsilon-greedy with Nash equilibrium strategy
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; else use pure Nash strategy
            
        Returns:
            Selected action (0, 1, or 2)
        """
        # Exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Get Q-values
        q_matrix = self.get_q_values(state, use_target=False)
        
        # Compute Nash equilibrium strategy
        state_key = tuple(state)
        if state_key not in self.nash_strategies:
            strategy = self._compute_nash_strategy(q_matrix)
            self.nash_strategies[state_key] = strategy
        else:
            strategy = self.nash_strategies[state_key]
        
        # Sample action according to mixed strategy
        return np.random.choice(self.action_size, p=strategy)
    
    def _compute_nash_strategy(self, q_matrix):
        """
        Compute Nash equilibrium mixed strategy from Q-values
        
        Uses maxmin strategy computation via linear programming
        """
        try:
            from scipy.optimize import linprog
            
            n = self.action_size
            
            # Objective: maximize v (minimize -v)
            c = np.zeros(n + 1)
            c[-1] = -1
            
            # Constraints: -sum_i p_i * q[i,j] + v <= 0 for all j
            A_ub = []
            b_ub = []
            for j in range(n):
                constraint = np.zeros(n + 1)
                constraint[:n] = -q_matrix[:, j]
                constraint[-1] = 1
                A_ub.append(constraint)
                b_ub.append(0)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # Equality: sum p_i = 1
            A_eq = np.zeros((1, n + 1))
            A_eq[0, :n] = 1
            b_eq = np.array([1])
            
            # Bounds: p_i >= 0, v unbounded
            bounds = [(0, None)] * n + [(None, None)]
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                           bounds=bounds, method='highs')
            
            if result.success:
                strategy = result.x[:n]
                strategy = strategy / strategy.sum()  # Normalize
                return strategy
        except:
            pass
        
        # Fallback: softmax over Q-values
        q_avg = q_matrix.mean(axis=1)
        exp_q = np.exp(q_avg - q_avg.max())
        return exp_q / exp_q.sum()
    
    def update(self, state, my_action, opp_action, reward, next_state, done):
        """
        Store experience and perform learning update
        
        Args:
            state: Current state
            my_action: Action taken by this agent
            opp_action: Action taken by opponent
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store in replay buffer
        self.memory.push(state, my_action, opp_action, reward, next_state, done)
        
        # Update counter
        self.steps_done += 1
        
        # Learn from batch if enough samples
        if len(self.memory) >= self.batch_size:
            self._learn()
        
        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _learn(self):
        """
        Sample from replay buffer and update network
        """
        # Sample batch
        states, my_actions, opp_actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        my_actions = torch.LongTensor(my_actions).to(self.device)
        opp_actions = torch.LongTensor(opp_actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q-values
        current_q_values = self.policy_net(states)
        current_q = current_q_values[range(self.batch_size), my_actions, opp_actions]
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            
            # Nash equilibrium value for next state
            nash_values = []
            for i in range(self.batch_size):
                q_matrix = next_q_values[i].cpu().numpy()
                strategy = self._compute_nash_strategy(q_matrix)
                
                # Expected value under Nash strategy
                value = 0.0
                for a1 in range(self.action_size):
                    for a2 in range(self.action_size):
                        value += strategy[a1] * strategy[a2] * q_matrix[a1, a2]
                nash_values.append(value)
            
            nash_values = torch.FloatTensor(nash_values).to(self.device)
            target_q = rewards + (1 - dones) * self.discount_factor * nash_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Record loss
        self.losses.append(loss.item())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save agent to file"""
        data = {
            'agent_id': self.agent_id,
            'hidden_size': self.hidden_size,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'total_rewards': self.total_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses[-1000:],  # Keep last 1000 losses
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'draw_count': self.draw_count,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(data, filepath)
        print(f"Agent {self.agent_id} saved to {filepath}")
    
    def load(self, filepath):
        """Load agent from file"""
        try:
            data = torch.load(filepath, map_location=self.device)
            
            # Try to detect hidden_size from the saved model architecture
            saved_hidden_size = data.get('hidden_size', None)
            if saved_hidden_size is None:
                # Try to infer from the weight shapes
                try:
                    fc1_weight_shape = data['policy_net_state']['fc1.weight'].shape
                    saved_hidden_size = fc1_weight_shape[0]  # Output size of fc1 = hidden_size
                    print(f"Detected hidden_size={saved_hidden_size} from model weights")
                except:
                    saved_hidden_size = self.hidden_size
            
            # Check if architecture matches
            if saved_hidden_size != self.hidden_size:
                print(f"Warning: Model was trained with hidden_size={saved_hidden_size}, but current hidden_size={self.hidden_size}")
                print(f"Recreating networks with hidden_size={saved_hidden_size}...")
                # Recreate networks with correct size
                self.hidden_size = saved_hidden_size
                self.policy_net = DQNetwork(self.state_size, self.action_size, saved_hidden_size).to(self.device)
                self.target_net = DQNetwork(self.state_size, self.action_size, saved_hidden_size).to(self.device)
                self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
            
            self.policy_net.load_state_dict(data['policy_net_state'])
            self.target_net.load_state_dict(data['target_net_state'])
            self.optimizer.load_state_dict(data['optimizer_state'])
            self.learning_rate = data['learning_rate']
            self.discount_factor = data['discount_factor']
            self.epsilon = data.get('epsilon', self.epsilon_min)
            self.steps_done = data.get('steps_done', 0)
            self.total_rewards = data.get('total_rewards', [])
            self.episode_lengths = data.get('episode_lengths', [])
            self.losses = data.get('losses', [])
            self.win_count = data.get('win_count', 0)
            self.loss_count = data.get('loss_count', 0)
            self.draw_count = data.get('draw_count', 0)
            
            print(f"Agent {self.agent_id} loaded from {filepath} (hidden_size={self.hidden_size})")
            return True
        except FileNotFoundError:
            print(f"No saved model found at {filepath}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def train_deep_nash_q_learning(width=15, height=15, num_episodes=5000,
                               save_interval=500, model_dir='models',
                               learning_rate=0.001, discount_factor=0.95,
                               batch_size=64, buffer_capacity=50000,
                               hidden_size=128):
    """
    Train two agents using Deep Nash Q-Learning
    
    Args:
        width: Game board width
        height: Game board height
        num_episodes: Number of training episodes
        save_interval: Save models every N episodes
        model_dir: Directory to save models
        learning_rate: Learning rate for networks
        discount_factor: Discount factor for future rewards
        batch_size: Mini-batch size
        buffer_capacity: Replay buffer size
        hidden_size: Hidden layer size
    """
    from game import CompetitiveSnakeGame
    
    # Initialize environment
    env = CompetitiveSnakeGame(width, height)
    
    # Initialize agents
    agent1 = DeepNashQLearningAgent(
        agent_id=1,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        hidden_size=hidden_size
    )
    agent2 = DeepNashQLearningAgent(
        agent_id=2,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        hidden_size=hidden_size
    )
    
    # Try to load existing models
    agent1.load(f"{model_dir}/agent1_deep_nash.pth")
    agent2.load(f"{model_dir}/agent2_deep_nash.pth")
    
    print(f"\nTraining Deep Nash Q-Learning agents for {num_episodes} episodes")
    print(f"Game size: {width}x{height}")
    print(f"Learning rate: {learning_rate}, Discount: {discount_factor}")
    print(f"Batch size: {batch_size}, Buffer: {buffer_capacity}")
    print(f"Hidden size: {hidden_size}")
    print("="*70)
    
    for episode in range(num_episodes):
        state1, state2 = env.reset()
        
        episode_reward1 = 0
        episode_reward2 = 0
        steps = 0
        
        while True:
            # Get actions from both agents
            action1 = agent1.get_action(state1, training=True)
            action2 = agent2.get_action(state2, training=True)
            
            # Take step in environment
            reward1, reward2, (next_state1, next_state2), done, info = \
                env.step(action1, action2)
            
            # Update both agents
            agent1.update(state1, action1, action2, reward1, next_state1, done)
            agent2.update(state2, action2, action1, reward2, next_state2, done)
            
            episode_reward1 += reward1
            episode_reward2 += reward2
            steps += 1
            
            state1 = next_state1
            state2 = next_state2
            
            if done:
                break
        
        # Decay exploration
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        # Record statistics
        agent1.total_rewards.append(episode_reward1)
        agent2.total_rewards.append(episode_reward2)
        agent1.episode_lengths.append(steps)
        agent2.episode_lengths.append(steps)
        
        # Record wins/losses
        if info['snake1_score'] > info['snake2_score']:
            agent1.win_count += 1
            agent2.loss_count += 1
        elif info['snake2_score'] > info['snake1_score']:
            agent2.win_count += 1
            agent1.loss_count += 1
        else:
            agent1.draw_count += 1
            agent2.draw_count += 1
        
        # Print progress
        if episode % 50 == 0:
            avg_reward1 = np.mean(agent1.total_rewards[-100:]) if len(agent1.total_rewards) >= 100 else np.mean(agent1.total_rewards)
            avg_reward2 = np.mean(agent2.total_rewards[-100:]) if len(agent2.total_rewards) >= 100 else np.mean(agent2.total_rewards)
            avg_loss1 = np.mean(agent1.losses[-100:]) if len(agent1.losses) >= 100 else 0
            avg_loss2 = np.mean(agent2.losses[-100:]) if len(agent2.losses) >= 100 else 0
            
            win_rate1 = agent1.win_count / (episode + 1) * 100
            win_rate2 = agent2.win_count / (episode + 1) * 100
            
            print(f"Ep {episode:4d} | "
                  f"A1: S={info['snake1_score']:2d} R={episode_reward1:6.1f} AvgR={avg_reward1:6.1f} "
                  f"Win%={win_rate1:5.1f} Loss={avg_loss1:6.3f} | "
                  f"A2: S={info['snake2_score']:2d} R={episode_reward2:6.1f} AvgR={avg_reward2:6.1f} "
                  f"Win%={win_rate2:5.1f} Loss={avg_loss2:6.3f} | "
                  f"Steps={steps:3d} ε={agent1.epsilon:.3f} Buf={len(agent1.memory):5d}")
        
        # Save models periodically
        if episode % save_interval == 0 and episode > 0:
            agent1.save(f"{model_dir}/agent1_deep_nash.pth")
            agent2.save(f"{model_dir}/agent2_deep_nash.pth")
    
    # Final save
    agent1.save(f"{model_dir}/agent1_deep_nash.pth")
    agent2.save(f"{model_dir}/agent2_deep_nash.pth")
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Agent 1 - Wins: {agent1.win_count}, Losses: {agent1.loss_count}, Draws: {agent1.draw_count}")
    print(f"Agent 2 - Wins: {agent2.win_count}, Losses: {agent2.loss_count}, Draws: {agent2.draw_count}")
    print(f"Replay buffer sizes - Agent1: {len(agent1.memory)}, Agent2: {len(agent2.memory)}")
    print("="*70)
    
    return agent1, agent2


if __name__ == "__main__":
    train_deep_nash_q_learning(
        width=15,
        height=15,
        num_episodes=3000,
        save_interval=300,
        hidden_size=128
    )