"""
Nash Q-Learning Algorithm for Competitive Snake Game

Nash Q-Learning extends Q-learning to multi-agent settings by computing
Nash equilibria of the stage games defined by the agents' Q-values.
"""

import numpy as np
import pickle
import os
from collections import defaultdict
from scipy.optimize import linprog
import itertools
import matplotlib.pyplot as plt


class NashQLearningAgent:
    """
    Nash Q-Learning agent for competitive two-player game
    
    Each agent maintains Q-values for joint actions and computes
    Nash equilibrium strategies at each state.
    """
    
    def __init__(self, agent_id, state_size=18, action_size=3, 
                 learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialize Nash Q-Learning agent
        
        Args:
            agent_id: 1 or 2, identifying which agent this is
            state_size: Dimension of state vector
            action_size: Number of actions per agent
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
        """
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: maps (state, my_action, opponent_action) -> Q-value
        # Using defaultdict for sparse representation
        self.q_table = defaultdict(lambda: np.zeros((action_size, action_size)))
        
        # Nash equilibrium strategies: maps state -> mixed strategy (probability distribution)
        self.nash_strategies = {}
        
        # Training statistics
        self.total_rewards = []
        self.episode_lengths = []
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        
    def state_to_key(self, state):
        """Convert state array to hashable key"""
        return tuple(state)
    
    def get_action(self, state, training=True):
        """
        Select action using epsilon-greedy with Nash equilibrium strategy
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; else use pure Nash strategy
            
        Returns:
            Selected action (0, 1, or 2)
        """
        state_key = self.state_to_key(state)
        
        # Exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        
        # Exploitation: use Nash equilibrium strategy
        if state_key not in self.nash_strategies:
            # If no Nash strategy computed yet, use uniform random
            return np.random.randint(0, self.action_size)
        
        strategy = self.nash_strategies[state_key]
        
        # Sample action according to mixed strategy
        return np.random.choice(self.action_size, p=strategy)
    
    def update(self, state, my_action, opp_action, reward, next_state, done):
        """
        Update Q-values and compute Nash equilibrium
        
        Args:
            state: Current state
            my_action: Action taken by this agent
            opp_action: Action taken by opponent
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Get current Q-value
        current_q = self.q_table[state_key][my_action, opp_action]
        
        # Compute target Q-value
        if done:
            target_q = reward
        else:
            # Value is the Nash equilibrium value of next state
            nash_value = self._compute_nash_value(next_state_key)
            target_q = reward + self.discount_factor * nash_value
        
        # Q-learning update
        self.q_table[state_key][my_action, opp_action] += \
            self.learning_rate * (target_q - current_q)
        
        # Recompute Nash equilibrium for this state
        self._compute_nash_equilibrium(state_key)
    
    def _compute_nash_equilibrium(self, state_key):
        """
        Compute Nash equilibrium strategy for a state
        
        For 2-player zero-sum games, we can use linear programming to find
        the Nash equilibrium mixed strategy.
        """
        q_matrix = self.q_table[state_key]
        
        try:
            # Use maxmin strategy (works for general-sum games as approximation)
            strategy = self._solve_maxmin(q_matrix)
            self.nash_strategies[state_key] = strategy
        except:
            # If optimization fails, use uniform strategy
            self.nash_strategies[state_key] = np.ones(self.action_size) / self.action_size
    
    def _solve_maxmin(self, payoff_matrix):
        """
        Solve for maxmin strategy using linear programming
        
        max v
        s.t. sum_i p_i * payoff[i,j] >= v for all j
             sum_i p_i = 1
             p_i >= 0
        """
        n = payoff_matrix.shape[0]
        
        # Objective: maximize v (equivalently, minimize -v)
        # Variables: [p_1, ..., p_n, v]
        c = np.zeros(n + 1)
        c[-1] = -1  # Maximize v
        
        # Inequality constraints: -sum_i p_i * payoff[i,j] + v <= 0 for all j
        A_ub = []
        b_ub = []
        for j in range(payoff_matrix.shape[1]):
            constraint = np.zeros(n + 1)
            constraint[:n] = -payoff_matrix[:, j]
            constraint[-1] = 1
            A_ub.append(constraint)
            b_ub.append(0)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Equality constraint: sum p_i = 1
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1
        b_eq = np.array([1])
        
        # Bounds: p_i >= 0, v unbounded
        bounds = [(0, None)] * n + [(None, None)]
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if result.success:
            strategy = result.x[:n]
            # Normalize to ensure it sums to 1 (numerical stability)
            strategy = strategy / strategy.sum()
            return strategy
        else:
            # Fallback to uniform distribution
            return np.ones(n) / n
    
    def _compute_nash_value(self, state_key):
        """
        Compute the Nash equilibrium value for a state
        
        This is the expected payoff when both players play their Nash strategies
        """
        if state_key not in self.nash_strategies:
            return 0.0
        
        q_matrix = self.q_table[state_key]
        strategy = self.nash_strategies[state_key]
        
        # Expected value: sum over all action pairs weighted by strategy probabilities
        # For opponent, we assume they also play their Nash strategy
        # In practice, we use our strategy as approximation for opponent's
        value = 0.0
        for i in range(self.action_size):
            for j in range(self.action_size):
                value += strategy[i] * strategy[j] * q_matrix[i, j]
        
        return value
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save agent to file"""
        data = {
            'agent_id': self.agent_id,
            'q_table': dict(self.q_table),
            'nash_strategies': self.nash_strategies,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'total_rewards': self.total_rewards,
            'episode_lengths': self.episode_lengths,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'draw_count': self.draw_count,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent {self.agent_id} saved to {filepath}")
    
    def load(self, filepath):
        """Load agent from file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Reconstruct Q-table as defaultdict
            q_table_dict = data['q_table']
            self.q_table = defaultdict(lambda: np.zeros((self.action_size, self.action_size)))
            for key, value in q_table_dict.items():
                self.q_table[key] = value
            
            self.nash_strategies = data['nash_strategies']
            self.learning_rate = data['learning_rate']
            self.discount_factor = data['discount_factor']
            self.epsilon = data.get('epsilon', self.epsilon_min)
            self.total_rewards = data.get('total_rewards', [])
            self.episode_lengths = data.get('episode_lengths', [])
            self.win_count = data.get('win_count', 0)
            self.loss_count = data.get('loss_count', 0)
            self.draw_count = data.get('draw_count', 0)
            
            print(f"Agent {self.agent_id} loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"No saved model found at {filepath}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def train_nash_q_learning(width=15, height=15, num_episodes=10000,
                         save_interval=1000, model_dir='models',
                         learning_rate=0.1, discount_factor=0.95,
                         experiment_dir=None):
    """
    Train two agents using Nash Q-Learning in competitive environment
    
    Args:
        width: Game board width
        height: Game board height
        num_episodes: Number of training episodes
        save_interval: Save models every N episodes
        model_dir: Directory to save models
        learning_rate: Learning rate for both agents
        discount_factor: Discount factor for both agents
    """
    from Game.game import CompetitiveSnakeGame
    
    # Initialize environment
    env = CompetitiveSnakeGame(width, height)
    
    # Initialize agents
    agent1 = NashQLearningAgent(
        agent_id=1,
        learning_rate=learning_rate,
        discount_factor=discount_factor
    )
    agent2 = NashQLearningAgent(
        agent_id=2,
        learning_rate=learning_rate,
        discount_factor=discount_factor
    )
    
    # Try to load existing models
    agent1.load(f"{model_dir}/agent1_nash.pkl")
    agent2.load(f"{model_dir}/agent2_nash.pkl")

    # Setup experiment directory logging if requested (matches corporative structure)
    log_file = None
    checkpoints_dir = None
    best_model_path = None
    final_model_path = None
    if experiment_dir:
        os.makedirs(experiment_dir, exist_ok=True)
        # CSV log
        log_path = os.path.join(experiment_dir, 'training_log.csv')
        log_file = open(log_path, 'w', encoding='utf-8')
        log_file.write("Episode,Agent1Reward,Agent2Reward,Agent1AvgR,Agent2AvgR,Agent1Wins,Agent2Wins,Steps,Epsilon\n")

        # Hyperparameters
        hyper_path = os.path.join(experiment_dir, 'hyperparameters.txt')
        with open(hyper_path, 'w', encoding='utf-8') as hf:
            hf.write(f"algorithm: nash_q\n")
            hf.write(f"width: {width}\nheight: {height}\n")
            hf.write(f"num_episodes: {num_episodes}\n")
            hf.write(f"save_interval: {save_interval}\n")
            hf.write(f"learning_rate: {learning_rate}\n")
            hf.write(f"discount_factor: {discount_factor}\n")

        # Checkpoints dir and model paths
        checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        best_model_path = os.path.join(experiment_dir, 'best_model.pkl')
        final_model_path = os.path.join(experiment_dir, 'final_model.pkl')
        # Plot path
        plot_path = os.path.join(experiment_dir, 'training_plot.png')
    
    print(f"Training Nash Q-Learning agents for {num_episodes} episodes")
    print(f"Game size: {width}x{height}")
    print(f"Learning rate: {learning_rate}, Discount: {discount_factor}")
    
    best_score = -float('inf')
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
        if episode % 20 == 0:
            avg_reward1 = np.mean(agent1.total_rewards[-100:]) if len(agent1.total_rewards) >= 100 else np.mean(agent1.total_rewards)
            avg_reward2 = np.mean(agent2.total_rewards[-100:]) if len(agent2.total_rewards) >= 100 else np.mean(agent2.total_rewards)
            
            win_rate1 = agent1.win_count / (episode + 1) * 100
            win_rate2 = agent2.win_count / (episode + 1) * 100
            
            print(f"Episode {episode:5d} | "
                  f"Agent1: Score={info['snake1_score']:2d} Reward={episode_reward1:6.1f} AvgR={avg_reward1:6.1f} Win%={win_rate1:5.1f} | "
                  f"Agent2: Score={info['snake2_score']:2d} Reward={episode_reward2:6.1f} AvgR={avg_reward2:6.1f} Win%={win_rate2:5.1f} | "
                  f"Steps={steps:3d} ε={agent1.epsilon:.3f} Q-states={len(agent1.q_table):5d}")
        
        # Log to CSV if requested
        if log_file:
            log_file.write(f"{episode},{episode_reward1},{episode_reward2},{avg_reward1:.3f},{avg_reward2:.3f},{agent1.win_count},{agent2.win_count},{steps},{agent1.epsilon:.4f}\n")
            log_file.flush()

        # Save models periodically
        if episode % save_interval == 0 and episode > 0:
            # Default model dir save
            agent1.save(f"{model_dir}/agent1_nash.pkl")
            agent2.save(f"{model_dir}/agent2_nash.pkl")

            # Also save checkpoint copies if experiment_dir provided
            if checkpoints_dir:
                cp1 = os.path.join(checkpoints_dir, f'agent1_episode_{episode}.pkl')
                cp2 = os.path.join(checkpoints_dir, f'agent2_episode_{episode}.pkl')
                agent1.save(cp1)
                agent2.save(cp2)
            # Update training plot
            try:
                if plot_path:
                    plt.figure(figsize=(10, 6))
                    # rewards
                    plt.plot(agent1.total_rewards, label='Agent1 Reward')
                    plt.plot(agent2.total_rewards, label='Agent2 Reward')
                    # moving averages
                    if len(agent1.total_rewards) > 100:
                        ma1 = np.convolve(agent1.total_rewards, np.ones(100)/100, mode='valid')
                        plt.plot(range(99, 99+len(ma1)), ma1, 'k--', label='A1 MA(100)')
                    if len(agent2.total_rewards) > 100:
                        ma2 = np.convolve(agent2.total_rewards, np.ones(100)/100, mode='valid')
                        plt.plot(range(99, 99+len(ma2)), ma2, 'r--', label='A2 MA(100)')
                    plt.xlabel('Episode')
                    plt.ylabel('Reward')
                    plt.title('Training Rewards')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=150)
                    plt.close()
            except Exception:
                pass

        # Update best model based on top score seen this episode
        max_episode_score = max(info.get('snake1_score', 0), info.get('snake2_score', 0))
        if max_episode_score > best_score:
            best_score = max_episode_score
            # Save best models in experiment folder
            if best_model_path:
                agent1.save(best_model_path.replace('.pkl', '_agent1.pkl'))
                agent2.save(best_model_path.replace('.pkl', '_agent2.pkl'))
    
    # Final save
    agent1.save(f"{model_dir}/agent1_nash.pkl")
    agent2.save(f"{model_dir}/agent2_nash.pkl")
    if final_model_path:
        agent1.save(final_model_path.replace('.pkl', '_agent1.pkl'))
        agent2.save(final_model_path.replace('.pkl', '_agent2.pkl'))

    if log_file:
        log_file.close()
    # Final plot
    try:
        if experiment_dir:
            plt.figure(figsize=(10, 6))
            plt.plot(agent1.total_rewards, label='Agent1 Reward')
            plt.plot(agent2.total_rewards, label='Agent2 Reward')
            if len(agent1.total_rewards) > 100:
                ma1 = np.convolve(agent1.total_rewards, np.ones(100)/100, mode='valid')
                plt.plot(range(99, 99+len(ma1)), ma1, 'k--', label='A1 MA(100)')
            if len(agent2.total_rewards) > 100:
                ma2 = np.convolve(agent2.total_rewards, np.ones(100)/100, mode='valid')
                plt.plot(range(99, 99+len(ma2)), ma2, 'r--', label='A2 MA(100)')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards')
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path.replace('.png', '_final.png'), dpi=150)
            plt.close()
    except Exception:
        pass
    
    print("\nTraining completed!")
    print(f"Agent 1 - Wins: {agent1.win_count}, Losses: {agent1.loss_count}, Draws: {agent1.draw_count}")
    print(f"Agent 2 - Wins: {agent2.win_count}, Losses: {agent2.loss_count}, Draws: {agent2.draw_count}")
    print(f"Q-table sizes - Agent1: {len(agent1.q_table)}, Agent2: {len(agent2.q_table)}")
    
    return agent1, agent2


if __name__ == "__main__":
    train_nash_q_learning(
        width=15,
        height=15,
        num_episodes=5000,
        save_interval=500
    )