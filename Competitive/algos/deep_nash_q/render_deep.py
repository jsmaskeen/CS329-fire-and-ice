#!/usr/bin/env python3
"""
Rendering demo for trained Deep Nash Q-Learning agents
"""

import sys
import os
import argparse
import pygame
import time

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deep_nash_qlearning import DeepNashQLearningAgent
from game import CompetitiveSnakeGame


class DeepNashRenderer:
    """Pygame renderer for Deep Nash Q-Learning agents"""
    
    def __init__(self, width=15, height=15, cell_size=30):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size + 140  # Extra space for info
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 100, 255)
        self.DARK_GREEN = (0, 150, 0)
        self.DARK_BLUE = (0, 50, 150)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (40, 40, 40)
        self.LIGHT_GRAY = (100, 100, 100)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (200, 0, 255)
        self.CYAN = (0, 255, 255)
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Deep Nash Q-Learning: Competitive Snake")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)
        
        self.running = True
        self.paused = False
        self.game_speed = 8  # FPS
        
    def draw_cell(self, x, y, color, border=True):
        """Draw a single cell on the grid"""
        rect = pygame.Rect(
            x * self.cell_size,
            y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, color, rect)
        if border:
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, rect, 1)
    
    def draw_snake(self, snake, head_color, body_color):
        """Draw a snake"""
        if not snake.alive:
            return
            
        for i, segment in enumerate(snake.body):
            if i == 0:  # Head
                self.draw_cell(segment[0], segment[1], head_color)
                # Draw direction indicator
                center_x = segment[0] * self.cell_size + self.cell_size // 2
                center_y = segment[1] * self.cell_size + self.cell_size // 2
                dir_x, dir_y = snake.direction
                end_x = center_x + dir_x * self.cell_size // 3
                end_y = center_y + dir_y * self.cell_size // 3
                pygame.draw.line(self.screen, self.WHITE, 
                               (center_x, center_y), (end_x, end_y), 3)
            else:  # Body
                self.draw_cell(segment[0], segment[1], body_color)
    
    def draw_game(self, game, agent1, agent2, episode, total_episodes):
        """Draw the game state"""
        # Background
        self.screen.fill(self.GRAY)
        
        # Draw grid
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, self.BLACK, rect)
                pygame.draw.rect(self.screen, self.LIGHT_GRAY, rect, 1)
        
        # Draw food
        if game.food:
            self.draw_cell(game.food[0], game.food[1], self.RED, False)
            # Pulsing effect
            pulse_size = int(self.cell_size * 0.3)
            center_x = game.food[0] * self.cell_size + self.cell_size // 2
            center_y = game.food[1] * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, self.ORANGE, (center_x, center_y), pulse_size)
        
        # Draw snakes
        self.draw_snake(game.snake1, self.GREEN, self.DARK_GREEN)
        self.draw_snake(game.snake2, self.BLUE, self.DARK_BLUE)
        
        # Info panel
        info_y = self.screen_height - 130
        
        # Divider line
        pygame.draw.line(self.screen, self.WHITE, 
                        (0, info_y - 5), (self.screen_width, info_y - 5), 2)
        
        # Title
        title = self.font_medium.render("DEEP NASH Q-LEARNING", True, self.CYAN)
        self.screen.blit(title, (10, info_y))
        
        # Episode info
        episode_text = self.font_small.render(
            f"Episode: {episode}/{total_episodes}  Speed: {self.game_speed} FPS",
            True, self.WHITE
        )
        self.screen.blit(episode_text, (10, info_y + 25))
        
        # Steps and buffer info
        stats_text = self.font_small.render(
            f"Steps: {game.steps}/{game.max_steps}  "
            f"Buffer: A1={len(agent1.memory)} A2={len(agent2.memory)}",
            True, self.YELLOW
        )
        self.screen.blit(stats_text, (10, info_y + 45))
        
        # Agent 1 info (GREEN)
        agent1_title = self.font_medium.render("Agent 1 (Green) - DQN", True, self.GREEN)
        self.screen.blit(agent1_title, (10, info_y + 65))
        
        agent1_info = self.font_small.render(
            f"Score: {game.snake1_score}  Len: {len(game.snake1.body)}  "
            f"Status: {'ALIVE' if game.snake1.alive else 'DEAD'}  "
            f"W:{agent1.win_count} L:{agent1.loss_count} D:{agent1.draw_count}",
            True, self.WHITE
        )
        self.screen.blit(agent1_info, (10, info_y + 90))
        
        # Agent 2 info (BLUE)
        agent2_title = self.font_medium.render("Agent 2 (Blue) - DQN", True, self.BLUE)
        self.screen.blit(agent2_title, (self.screen_width // 2, info_y + 65))
        
        agent2_info = self.font_small.render(
            f"Score: {game.snake2_score}  Len: {len(game.snake2.body)}  "
            f"Status: {'ALIVE' if game.snake2.alive else 'DEAD'}  "
            f"W:{agent2.win_count} L:{agent2.loss_count} D:{agent2.draw_count}",
            True, self.WHITE
        )
        self.screen.blit(agent2_info, (self.screen_width // 2, info_y + 90))
        
        # Controls
        controls_text = self.font_small.render(
            "SPACE: Pause  R: Reset  +/-: Speed  N: Next  ESC: Quit",
            True, self.YELLOW
        )
        self.screen.blit(controls_text, (10, info_y + 110))
        
        # Pause overlay
        if self.paused:
            overlay = pygame.Surface((self.screen_width, self.screen_height))
            overlay.set_alpha(128)
            overlay.fill(self.BLACK)
            self.screen.blit(overlay, (0, 0))
            
            pause_text = self.font_large.render("PAUSED", True, self.YELLOW)
            pause_rect = pause_text.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2)
            )
            self.screen.blit(pause_text, pause_rect)
    
    def draw_game_over(self, game, agent1, agent2):
        """Draw game over screen"""
        overlay = pygame.Surface((self.screen_width, self.screen_height - 140))
        overlay.set_alpha(200)
        overlay.fill(self.BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Game Over text
        game_over_text = self.font_large.render("GAME OVER", True, self.RED)
        game_over_rect = game_over_text.get_rect(
            center=(self.screen_width // 2, self.screen_height // 2 - 80)
        )
        self.screen.blit(game_over_text, game_over_rect)
        
        # Winner determination
        if game.snake1_score > game.snake2_score:
            winner_text = "Agent 1 (Green) WINS!"
            winner_color = self.GREEN
        elif game.snake2_score > game.snake1_score:
            winner_text = "Agent 2 (Blue) WINS!"
            winner_color = self.BLUE
        else:
            winner_text = "It's a DRAW!"
            winner_color = self.YELLOW
        
        winner = self.font_large.render(winner_text, True, winner_color)
        winner_rect = winner.get_rect(
            center=(self.screen_width // 2, self.screen_height // 2 - 20)
        )
        self.screen.blit(winner, winner_rect)
        
        # Final scores
        scores = self.font_medium.render(
            f"Final Score - Green: {game.snake1_score}  Blue: {game.snake2_score}",
            True, self.WHITE
        )
        scores_rect = scores.get_rect(
            center=(self.screen_width // 2, self.screen_height // 2 + 30)
        )
        self.screen.blit(scores, scores_rect)
        
        # Instructions
        instructions = self.font_small.render(
            "Press SPACE to continue or ESC to quit",
            True, self.YELLOW
        )
        instructions_rect = instructions.get_rect(
            center=(self.screen_width // 2, self.screen_height // 2 + 70)
        )
        self.screen.blit(instructions, instructions_rect)
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return 'quit'
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    return 'pause'
                elif event.key == pygame.K_r:
                    return 'reset'
                elif event.key == pygame.K_n:
                    return 'next'
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.game_speed = min(30, self.game_speed + 2)
                elif event.key == pygame.K_MINUS:
                    self.game_speed = max(1, self.game_speed - 2)
        return None
    
    def cleanup(self):
        """Cleanup pygame"""
        pygame.quit()


def run_demo(model_dir='models', width=15, height=15, 
             num_episodes=10, cell_size=30, speed=8):
    """
    Run visual demo of trained Deep Nash Q-Learning agents
    
    Args:
        model_dir: Directory containing trained models
        width: Game board width
        height: Game board height
        num_episodes: Number of episodes to demonstrate
        cell_size: Size of each grid cell in pixels
        speed: Initial game speed (FPS)
    """
    # Load agents
    print("Loading trained Deep Nash Q-Learning agents...")
    agent1 = DeepNashQLearningAgent(agent_id=1)
    agent2 = DeepNashQLearningAgent(agent_id=2)
    
    if not agent1.load(f"{model_dir}/agent1_deep_nash.pth"):
        print("Error: Agent 1 model not found!")
        print(f"Please train agents first using: python train_deep.py --train")
        return
    
    if not agent2.load(f"{model_dir}/agent2_deep_nash.pth"):
        print("Error: Agent 2 model not found!")
        print(f"Please train agents first using: python train_deep.py --train")
        return
    
    # Set to pure exploitation
    agent1.epsilon = 0.0
    agent2.epsilon = 0.0
    
    print("Agents loaded successfully!")
    print(f"Agent 1 - Steps: {agent1.steps_done}, Buffer: {len(agent1.memory)}")
    print(f"Agent 2 - Steps: {agent2.steps_done}, Buffer: {len(agent2.memory)}")
    print(f"\nStarting demo with {num_episodes} episodes...")
    print("Controls: SPACE=Pause, R=Reset, N=Next Episode, +/-=Speed, ESC=Quit")
    
    # Initialize renderer
    renderer = DeepNashRenderer(width, height, cell_size)
    renderer.game_speed = speed
    
    # Initialize game
    game = CompetitiveSnakeGame(width, height)
    
    episode = 0
    
    try:
        while renderer.running and episode < num_episodes:
            # Start new episode
            state1, state2 = game.reset()
            game_over = False
            
            while renderer.running:
                # Handle events
                event_result = renderer.handle_events()
                
                if event_result == 'quit':
                    break
                elif event_result == 'reset':
                    state1, state2 = game.reset()
                    game_over = False
                elif event_result == 'next':
                    break
                
                if not renderer.paused and not game_over:
                    # Get actions from agents
                    action1 = agent1.get_action(state1, training=False)
                    action2 = agent2.get_action(state2, training=False)
                    
                    # Step environment
                    reward1, reward2, (next_state1, next_state2), done, info = \
                        game.step(action1, action2)
                    
                    state1 = next_state1
                    state2 = next_state2
                    
                    if done:
                        game_over = True
                
                # Draw everything
                renderer.draw_game(game, agent1, agent2, episode + 1, num_episodes)
                
                if game_over:
                    renderer.draw_game_over(game, agent1, agent2)
                
                pygame.display.flip()
                renderer.clock.tick(renderer.game_speed)
                
                # Auto-advance after game over
                if game_over:
                    time.sleep(2)  # Pause to show result
                    break
            
            episode += 1
        
    finally:
        renderer.cleanup()
    
    print("\nDemo completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Visual demo of Deep Nash Q-Learning agents'
    )
    
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--width', type=int, default=15,
                       help='Game board width (default: 15)')
    parser.add_argument('--height', type=int, default=15,
                       help='Game board height (default: 15)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to demonstrate (default: 10)')
    parser.add_argument('--cell-size', type=int, default=30,
                       help='Size of grid cells in pixels (default: 30)')
    parser.add_argument('--speed', type=int, default=8,
                       help='Initial game speed in FPS (default: 8)')
    
    args = parser.parse_args()
    
    run_demo(
        model_dir=args.model_dir,
        width=args.width,
        height=args.height,
        num_episodes=args.episodes,
        cell_size=args.cell_size,
        speed=args.speed
    )


if __name__ == "__main__":
    main()