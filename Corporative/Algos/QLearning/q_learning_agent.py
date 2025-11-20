import numpy as np
import random
import pickle
import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Try importing Game module with different casings/locations to avoid "No module named 'Game'" errors
try:
    from Game.game import Game
except ModuleNotFoundError:
    try:
        from game.game import Game
    except ModuleNotFoundError:
        try:
            from game import Game
        except ModuleNotFoundError:
            # Attempt to add repository root to sys.path and retry imports
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if repo_root not in sys.path:
                sys.path.append(repo_root)
            try:
                from Game.game import Game
            except ModuleNotFoundError:
                try:
                    from game.game import Game
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        "Could not import Game module. Ensure there is a 'Game' package (with __init__.py) "
                        "or a 'game.py' file somewhere in the project and that the correct path is on sys.path."
                    ) from e

class QLearningAgent:
    def __init__(self, state_size=22, action_size=9, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.scores = []
        self.episode_lengths = []
        self.epsilon_history = []

    def state_to_key(self, state):
        return tuple(state)

    def get_action(self, state):
        state_key = self.state_to_key(state)

        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state_key])

    def decode_action(self, action):
        snake1_action = action // 3
        snake2_action = action % 3
        return snake1_action, snake2_action

    def update_q_table(self, state, action, reward, next_state, done):
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        current_q = self.q_table[state_key][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state_key])

        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath):
        """Save Q-table and parameters"""
        model_data = {
            'q_table': dict(self.q_table),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'scores': self.scores,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.q_table = defaultdict(lambda: np.zeros(self.action_size), model_data['q_table'])
            self.learning_rate = model_data['learning_rate']
            self.discount_factor = model_data['discount_factor']
            self.epsilon = model_data['epsilon']
            self.epsilon_min = model_data['epsilon_min']
            self.epsilon_decay = model_data['epsilon_decay']
            self.scores = model_data.get('scores', [])
            self.episode_lengths = model_data.get('episode_lengths', [])
            self.epsilon_history = model_data.get('epsilon_history', [])

            print(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"No saved model found at {filepath}")
            return False

    def plot_training_progress(self, save_path=None, show=True):
        """Plot training statistics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

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

        q_table_sizes = [len(self.q_table) for _ in range(len(self.scores))]
        ax4.plot(q_table_sizes)
        ax4.set_title('Q-table Size Growth')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Number of States')
        ax4.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

def train_q_learning_agent(width=15, height=15, num_episodes=10000,
                         save_interval=1000, model_path='models/qlearning_snake.pkl',
                         load_existing=True, learning_rate=0.1, discount_factor=0.95,
                         epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                         experiment_dir=None):
    env = Game(width, height)
    agent = QLearningAgent(learning_rate=learning_rate, discount_factor=discount_factor,
                           epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)

    if load_existing and os.path.exists(model_path):
        agent.load_model(model_path)

    print(f"Starting Q-Learning training for {num_episodes} episodes...")
    print(f"Game size: {width}x{height}")
    print(f"Initial epsilon: {agent.epsilon}")

    # Setup logging
    log_file = None
    if experiment_dir:
        log_path = os.path.join(experiment_dir, 'training_log.csv')
        log_file = open(log_path, 'w')
        log_file.write("Episode,Score,Steps,AvgScore,Epsilon,QStates\n")

        checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        best_model_path = os.path.join(experiment_dir, 'best_model.pkl')
        plot_path = os.path.join(experiment_dir, 'training_plot.png')

    best_score = -float('inf')

    for episode in range(num_episodes):
        env.reset()
        env.spawn_food()
        state = env.get_state()

        total_reward = 0
        steps = 0
        max_steps = width * height * 2

        while steps < max_steps:
            action = agent.get_action(state)
            snake1_action, snake2_action = agent.decode_action(action)

            reward, next_state, done = env.step(snake1_action, snake2_action)

            agent.update_q_table(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

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
            log_file.write(f"{episode},{env.score},{steps},{avg_score:.2f},{agent.epsilon:.4f},{len(agent.q_table)}\n")
            log_file.flush()

        # Save best model
        if env.score > best_score:
            best_score = env.score
            if experiment_dir:
                agent.save_model(best_model_path)

        if episode % 100 == 0:
            avg_length = np.mean(agent.episode_lengths[-100:]) if len(agent.episode_lengths) >= 100 else np.mean(agent.episode_lengths)

            print(f"Episode {episode:6d} | "
                  f"Score: {env.score:3d} | "
                  f"Avg Score: {avg_score:6.2f} | "
                  f"Best Score: {best_score:3d} | "
                  f"Steps: {steps:4d} | "
                  f"Avg Steps: {avg_length:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Q-states: {len(agent.q_table):6d}")

            # Update plot in real-time
            if experiment_dir:
                agent.plot_training_progress(save_path=plot_path, show=False)

        if episode % save_interval == 0 and episode > 0:
            if experiment_dir:
                checkpoint_path = os.path.join(checkpoints_dir, f'episode_{episode}.pkl')
                agent.save_model(checkpoint_path)
            else:
                agent.save_model(model_path)

    if log_file:
        log_file.close()

    # Final save
    if experiment_dir:
        final_path = os.path.join(experiment_dir, 'final_model.pkl')
        agent.save_model(final_path)
    else:
        agent.save_model(model_path)

    print(f"\nTraining completed!")
    print(f"Best score achieved: {best_score}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Q-table size: {len(agent.q_table)} states")

    return agent

def test_q_learning_agent(model_path='models/qlearning_snake.pkl', width=15, height=15,
                         num_test_episodes=100, render=False):
    env = Game(width, height)
    agent = QLearningAgent()

    if not agent.load_model(model_path):
        print("No trained model found!")
        return

    agent.epsilon = 0.0

    print(f"Testing Q-Learning agent for {num_test_episodes} episodes...")

    test_scores = []
    test_lengths = []

    for episode in range(num_test_episodes):
        env.reset()
        env.spawn_food()
        state = env.get_state()

        steps = 0
        max_steps = width * height * 2

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

    print(f"\nTest Results:")
    print(f"Average Score: {np.mean(test_scores):.2f} ± {np.std(test_scores):.2f}")
    print(f"Max Score: {max(test_scores)}")
    print(f"Min Score: {min(test_scores)}")
    print(f"Average Episode Length: {np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f}")

    return test_scores, test_lengths