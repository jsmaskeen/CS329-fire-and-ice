import numpy as np
import random
import pickle
import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

# Add parent directory to path for imports (e.g., from Game.game import Game)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


class MonteCarloAgent:
    """
    On-Policy First-Visit Monte Carlo Agent for Cooperative Snake Game

    This agent learns a joint Q-function Q(s, (a1, a2)).
    State size = 22, Joint Action size = 9 (3 actions per snake).
    """

    def __init__(self, state_size=22, action_size=9,
                 discount_factor=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.9999):

        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: maps state_key -> array of Q-values for 9 joint actions
        self.q_table = defaultdict(lambda: np.zeros(action_size))

        # Total returns S(s, a): stores the sum of returns for each (state, action) pair
        self.total_returns = defaultdict(lambda: np.zeros(action_size))

        # N(s, a): stores the count of visits for each (state, action) pair
        self.returns_count = defaultdict(lambda: np.zeros(action_size))

        # Training statistics
        self.scores = []
        self.episode_lengths = []
        self.epsilon_history = []

    def state_to_key(self, state):
        """Convert state array to hashable key"""
        return tuple(state)

    def get_action(self, state, training=True):
        """
        Select action using the epsilon-greedy policy over the 9 joint actions.
        """
        state_key = self.state_to_key(state)

        if training and random.random() <= self.epsilon:
            # Exploration: Choose a random joint action (0-8)
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: Choose the best joint action based on Q-table
            q_values = self.q_table[state_key]
            best_actions = np.where(q_values == np.max(q_values))[0]
            # Break ties randomly
            return np.random.choice(best_actions)

    def decode_action(self, action):
        """Decodes joint action (0-8) into individual snake actions (0-2)"""
        snake1_action = action // 3
        snake2_action = action % 3
        return snake1_action, snake2_action

    def update_q_values(self, episode_history):
        """
        Update Q-values using the First-Visit Monte Carlo method.

        Args:
            episode_history: A list of tuples (state, action, reward) for the episode.
        """
        G = 0.0 # Return (cumulative discounted reward)

        # Keep track of visited (s, a) pairs in this episode
        sa_visited = set()

        # Iterate backwards through the episode history
        for t in reversed(range(len(episode_history))):
            state, action, reward = episode_history[t]
            state_key = self.state_to_key(state)

            # Update the return G
            G = reward + self.discount_factor * G

            sa_pair = (state_key, action)

            # First-Visit MC: Only update if this is the first time (s, a) is encountered
            if sa_pair not in sa_visited:
                sa_visited.add(sa_pair)

                # Update total returns
                self.total_returns[state_key][action] += G

                # Update visit count
                self.returns_count[state_key][action] += 1

                # Update Q-value: Average of returns
                self.q_table[state_key][action] = \
                    self.total_returns[state_key][action] / self.returns_count[state_key][action]

    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath):
        """Save Q-table and parameters"""
        model_data = {
            'q_table': dict(self.q_table),
            'returns_count': dict(self.returns_count),
            'total_returns': dict(self.total_returns),
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'scores': self.scores,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load agent from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.q_table = defaultdict(lambda: np.zeros(self.action_size), model_data['q_table'])
            self.returns_count = defaultdict(lambda: np.zeros(self.action_size), model_data.get('returns_count', {}))
            self.total_returns = defaultdict(lambda: np.zeros(self.action_size), model_data.get('total_returns', {}))

            self.discount_factor = model_data['discount_factor']
            self.epsilon = model_data.get('epsilon', self.epsilon_min)
            self.scores = model_data.get('scores', [])
            self.episode_lengths = model_data.get('episode_lengths', [])
            self.epsilon_history = model_data.get('epsilon_history', [])

            print(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"No saved model found at {filepath}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def plot_training_progress(self, save_path=None, show=True):
        """Plot training statistics (similar to Q-Learning plot)"""
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

# --- Trainer and Tester Functions ---

def train_monte_carlo_agent(width=15, height=15, num_episodes=10000,
                         save_interval=1000, model_path='models/mc_snake.pkl',
                         load_existing=True, discount_factor=0.95,
                         epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9999,
                         experiment_dir=None):
    env = Game(width, height)
    agent = MonteCarloAgent(discount_factor=discount_factor, epsilon=epsilon,
                            epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)

    if load_existing and os.path.exists(model_path):
        agent.load_model(model_path)

    print(f"Starting Monte Carlo training for {num_episodes} episodes...")
    print(f"Game size: {width}x{height}")
    print(f"Discount factor: {discount_factor}, Initial epsilon: {agent.epsilon}")
    print(f"Epsilon decay: {epsilon_decay}, Epsilon min: {epsilon_min}")

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

    best_score = max(agent.scores) if agent.scores else -float('inf')

    for episode in range(num_episodes):
        env.reset()
        env.spawn_food()
        state = env.get_state()

        # History is a list of tuples (state, joint_action, reward) for this episode
        episode_history = []
        steps = 0
        max_steps = width * height * 2 # Safety limit

        while steps < max_steps:
            # Step 1: Generate an episode using the current policy
            joint_action = agent.get_action(state, training=True)
            snake1_action, snake2_action = agent.decode_action(joint_action)

            reward, next_state, done = env.step(snake1_action, snake2_action)

            # Record the transition
            episode_history.append((state, joint_action, reward))

            state = next_state
            steps += 1

            if done:
                break

        # Step 2: Policy Evaluation (Update Q-values)
        agent.update_q_values(episode_history)

        # Step 3: Policy Improvement (Decay epsilon)
        agent.decay_epsilon()

        # Record statistics
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

def test_monte_carlo_agent(model_path='models/mc_snake.pkl', width=15, height=15,
                         num_test_episodes=100):

    env = Game(width, height)
    agent = MonteCarloAgent()

    if not agent.load_model(model_path):
        print("No trained model found!")
        return [], []

    # Test on pure exploitation
    agent.epsilon = 0.0

    print(f"Testing Monte Carlo agent for {num_test_episodes} episodes...")

    test_scores = []
    test_lengths = []

    for episode in range(num_test_episodes):
        env.reset()
        env.spawn_food()
        state = env.get_state()

        steps = 0
        max_steps = width * height * 2

        while steps < max_steps:
            joint_action = agent.get_action(state, training=False)
            snake1_action, snake2_action = agent.decode_action(joint_action)

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
    print(f"Average Episode Length: {np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f}")

    return test_scores, test_lengths