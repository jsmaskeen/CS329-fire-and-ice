import os
import random
import pickle
from collections import deque, defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise ImportError(
        "PyTorch is required for Deep Q-Learning. Install it with 'pip install torch'."
    ) from e

import matplotlib.pyplot as plt

import sys
# Add Corporative directory to path to find Game
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Game.game import Game


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DDQNAgent:
    """
    A Double Deep Q-Network agent.
    """
    def __init__(self, state_size=22, action_size=9, hidden_size=128,
                 lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=20000, batch_size=64, target_update=500, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update = target_update

        self.device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        self.policy_net = MLP(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.target_net = MLP(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        self.steps_done = 0
        self.scores = []
        self.episode_lengths = []
        self.epsilon_history = []

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        state_arr = np.array(state, dtype=np.float32)

        if random.random() <= self.epsilon:
            action = random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                s = torch.from_numpy(state_arr).float().to(self.device).unsqueeze(0)
                q_vals = self.policy_net(s)
                action = int(torch.argmax(q_vals, dim=1).item())

        return action

    def decode_action(self, action):
        snake1_action = action // 3
        snake2_action = action % 3
        return snake1_action, snake2_action

    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(np.array(state, dtype=np.float32), action, reward, np.array(next_state, dtype=np.float32), done)

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device).unsqueeze(1)
        rewards = torch.from_numpy(rewards).float().to(self.device).unsqueeze(1)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device).unsqueeze(1)

        # Compute Q(s_t, a)
        q_values = self.policy_net(states).gather(1, actions)

        # Double DQN Target:
        # 1. Select best action a' using Policy Network: argmax_a' Q_policy(s_{t+1}, a')
        # 2. Evaluate that action using Target Network: Q_target(s_{t+1}, a')
        with torch.no_grad():
            # Select action using policy net
            next_state_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Evaluate action using target net
            max_next_q_values = self.target_net(next_states).gather(1, next_state_actions)

        expected_q_values = rewards + (1.0 - dones) * (self.gamma * max_next_q_values)

        loss = nn.functional.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'policy_state': self.policy_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'params': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'hidden_size': self.hidden_size,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
            },
            'scores': self.scores,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history
        }
        torch.save(data, filepath)
        print(f"DDQN model saved to {filepath}")

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            print(f"No saved model found at {filepath}")
            return False

        data = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(data['policy_state'])
        try:
            self.target_net.load_state_dict(data.get('target_state', data['policy_state']))
        except Exception:
            pass
        try:
            self.optimizer.load_state_dict(data.get('optimizer_state', self.optimizer.state_dict()))
        except Exception:
            pass

        params = data.get('params', {})
        self.epsilon = params.get('epsilon', self.epsilon)
        self.epsilon_min = params.get('epsilon_min', self.epsilon_min)
        self.epsilon_decay = params.get('epsilon_decay', self.epsilon_decay)

        self.scores = data.get('scores', [])
        self.episode_lengths = data.get('episode_lengths', [])
        self.epsilon_history = data.get('epsilon_history', [])

        print(f"DDQN model loaded from {filepath}")
        return True

    def plot_training_progress(self, save_path=None, show=True):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        ax1.plot(self.scores)
        ax1.set_title('Scores over Episodes')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.grid(True)

        if len(self.scores) > 100:
            moving_avg = np.convolve(self.scores, np.ones(100)/100, mode='valid')
            ax1.plot(range(99, len(self.scores)), moving_avg, 'r-', label='Moving Average (100)')
            ax1.legend()

        ax2.plot(self.episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)

        ax3.plot(self.epsilon_history)
        ax3.set_title('Exploration Rate (Epsilon)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # print(f"Training progress plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)


def train_ddqn_agent(width=15, height=15, num_episodes=2000,
                           save_interval=500, model_path='models/ddqn_snake.pth',
                           load_existing=False, experiment_dir=None, train_every=4, **agent_kwargs):
    env = Game(width, height)

    # Try to infer state size from env.get_state()
    env.reset()
    env.spawn_food()
    sample_state = env.get_state()
    state_size = len(np.array(sample_state).ravel())

    agent = DDQNAgent(state_size=state_size, action_size=9, **agent_kwargs)

    if load_existing and os.path.exists(model_path):
        agent.load_model(model_path)

    print(f"Starting DDQN training for {num_episodes} episodes...")
    best_score = -float('inf')
    target_update_counter = 0

    # Setup logging
    log_file = None
    if experiment_dir:
        log_path = os.path.join(experiment_dir, 'training_log.csv')
        log_file = open(log_path, 'w')
        log_file.write("Episode,Score,Steps,AvgScore,Epsilon\n")
        
        # Setup plot path
        plot_path = os.path.join(experiment_dir, 'training_plot.png')
        
        # Setup checkpoints dir
        checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Setup best model path
        best_model_path = os.path.join(experiment_dir, 'best_model.pth')

    for episode in range(1, num_episodes + 1):
        env.reset()
        env.spawn_food()
        state = env.get_state()

        total_reward = 0
        steps = 0
        max_steps = width * height * 4

        while steps < max_steps:
            action = agent.get_action(state)
            snake1_action, snake2_action = agent.decode_action(action)

            reward, next_state, done = env.step(snake1_action, snake2_action)

            agent.push_transition(state, action, reward, next_state, done)
            
            # Optimize model only every 'train_every' steps
            if agent.steps_done % train_every == 0:
                agent.optimize_model()

            state = next_state
            total_reward += reward
            steps += 1
            agent.steps_done += 1
            target_update_counter += 1

            if target_update_counter >= agent.target_update:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                target_update_counter = 0

            if done:
                break

        agent.decay_epsilon()
        agent.scores.append(env.score)
        agent.episode_lengths.append(steps)
        agent.epsilon_history.append(agent.epsilon)

        # Calculate average score
        avg_score = np.mean(agent.scores[-100:]) if len(agent.scores) >= 100 else np.mean(agent.scores)

        # Log to CSV
        if log_file:
            log_file.write(f"{episode},{env.score},{steps},{avg_score:.2f},{agent.epsilon:.4f}\n")
            log_file.flush()

        # Save best model
        if env.score > best_score:
            best_score = env.score
            if experiment_dir:
                agent.save_model(best_model_path)

        # Console logging
        if episode % 100 == 0:
            avg_length = np.mean(agent.episode_lengths[-100:]) if len(agent.episode_lengths) >= 100 else np.mean(agent.episode_lengths)
            print(f"Episode {episode:6d} | Score: {env.score:3d} | Avg Score: {avg_score:6.2f} | Best: {best_score:3d} | Steps: {steps:4d} | Avg Steps: {avg_length:6.1f} | Eps: {agent.epsilon:.3f}")
            
            # Update plot in real-time
            if experiment_dir:
                agent.plot_training_progress(save_path=plot_path, show=False)

        # Save checkpoint
        if episode % save_interval == 0:
            if experiment_dir:
                checkpoint_path = os.path.join(checkpoints_dir, f'episode_{episode}.pth')
                agent.save_model(checkpoint_path)
            else:
                agent.save_model(model_path)

    if log_file:
        log_file.close()

    # Final save
    if experiment_dir:
        final_path = os.path.join(experiment_dir, 'final_model.pth')
        agent.save_model(final_path)
    else:
        agent.save_model(model_path)
        
    print("\nDDQN training completed!")
    print(f"Best score achieved: {best_score}")
    return agent


def test_ddqn_agent(model_path='models/ddqn_snake.pth', width=15, height=15, num_test_episodes=100):
    env = Game(width, height)

    env.reset()
    env.spawn_food()
    sample_state = env.get_state()
    state_size = len(np.array(sample_state).ravel())

    agent = DDQNAgent(state_size=state_size, action_size=9)

    if not agent.load_model(model_path):
        print("No trained model found!")
        return None

    agent.epsilon = 0.0

    test_scores = []
    test_lengths = []

    for episode in range(num_test_episodes):
        env.reset()
        env.spawn_food()
        state = env.get_state()

        steps = 0
        max_steps = width * height * 4

        while steps < max_steps:
            action = agent.get_action(state)
            snake1_action, snake2_action = agent.decode_action(action)

            reward, next_state, done = env.step(snake1_action, snake2_action)
            state = next_state
            steps += 1
            if done:
                break

        test_scores.append(env.score)
        test_lengths.append(steps)

        if episode % 10 == 0:
            print(f"Test Episode {episode:3d} | Score: {env.score:3d} | Steps: {steps:4d}")

    print("\nTest Results:")
    print(f"Average Score: {np.mean(test_scores):.2f} ± {np.std(test_scores):.2f}")
    print(f"Max Score: {max(test_scores)}")
    print(f"Min Score: {min(test_scores)}")
    print(f"Average Episode Length: {np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f}")

    return test_scores, test_lengths
