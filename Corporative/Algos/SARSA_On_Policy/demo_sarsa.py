import os
import sys
import time
import pygame

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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

from sarsa_agent import SARSAAgent

class SARSAGameDemo:
    def __init__(self, model_path='models/sarsa_snake.pkl', width=15, height=15, cell_size=30):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size + 100  # Extra space for info

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        self.DARK_GREEN = (0, 150, 0)
        self.DARK_BLUE = (0, 0, 150)

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Q-Learning Agent Playing 2-Player Snake")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)

        # Initialize game and agent
        self.game = Game(width, height)
        self.agent = SARSAAgent()

        # Load trained model
        if not self.agent.load_model(model_path):
            print(f"No trained model found at {model_path}")
            print("Please train the agent first using train_sarsa.py")
            sys.exit(1)

        # Set to pure exploitation (no exploration)
        self.agent.epsilon = 0.0

        self.running = True
        self.paused = False
        self.game_speed = 5  # FPS

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.game_speed = min(20, self.game_speed + 1)
                elif event.key == pygame.K_MINUS:
                    self.game_speed = max(1, self.game_speed - 1)
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

    def reset_game(self):
        self.game.reset()
        self.game.spawn_food()

    def draw_cell(self, x, y, color, border_color=None):
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                          self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect)
        if border_color:
            pygame.draw.rect(self.screen, border_color, rect, 1)

    def draw_game(self):
        # Clear screen
        self.screen.fill(self.BLACK)

        # Draw grid background
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.GRAY, rect, 1)

        # Draw food
        if self.game.food:
            self.draw_cell(self.game.food[0], self.game.food[1], self.RED, self.WHITE)

        # Draw snake 1 (Green)
        for i, segment in enumerate(self.game.snake1.body):
            if i == 0:  # Head
                self.draw_cell(segment[0], segment[1], self.GREEN, self.WHITE)
                # Draw direction indicator
                head_center_x = segment[0] * self.cell_size + self.cell_size // 2
                head_center_y = segment[1] * self.cell_size + self.cell_size // 2
                dir_x, dir_y = self.game.snake1.direction
                end_x = head_center_x + dir_x * self.cell_size // 3
                end_y = head_center_y + dir_y * self.cell_size // 3
                pygame.draw.line(self.screen, self.WHITE,
                               (head_center_x, head_center_y), (end_x, end_y), 3)
            else:  # Body
                color = self.DARK_GREEN
                self.draw_cell(segment[0], segment[1], color, self.WHITE)

        # Draw snake 2 (Blue)
        for i, segment in enumerate(self.game.snake2.body):
            if i == 0:  # Head
                self.draw_cell(segment[0], segment[1], self.BLUE, self.WHITE)
                # Draw direction indicator
                head_center_x = segment[0] * self.cell_size + self.cell_size // 2
                head_center_y = segment[1] * self.cell_size + self.cell_size // 2
                dir_x, dir_y = self.game.snake2.direction
                end_x = head_center_x + dir_x * self.cell_size // 3
                end_y = head_center_y + dir_y * self.cell_size // 3
                pygame.draw.line(self.screen, self.WHITE,
                               (head_center_x, head_center_y), (end_x, end_y), 3)
            else:  # Body
                color = self.DARK_BLUE
                self.draw_cell(segment[0], segment[1], color, self.WHITE)

        info_y = self.screen_height - 90

        score_text = self.font.render(f"Total Score: {self.game.score}", True, self.WHITE)
        self.screen.blit(score_text, (10, info_y))

        snake1_len = self.font.render(f"Green Snake Length: {len(self.game.snake1.body)}", True, self.GREEN)
        self.screen.blit(snake1_len, (10, info_y + 20))

        snake2_len = self.font.render(f"Blue Snake Length: {len(self.game.snake2.body)}", True, self.BLUE)
        self.screen.blit(snake2_len, (10, info_y + 40))

        q_states_text = self.font.render(f"Q-table States: {len(self.agent.q_table)}", True, self.WHITE)
        self.screen.blit(q_states_text, (250, info_y))

        speed_text = self.font.render(f"Speed: {self.game_speed} FPS", True, self.WHITE)
        self.screen.blit(speed_text, (250, info_y + 20))

        # Controls
        controls = [
            "SPACE: Pause/Resume",
            "R: Reset Game",
            "+/-: Speed",
            "ESC: Quit"
        ]
        for i, control in enumerate(controls):
            control_text = self.font.render(control, True, self.YELLOW)
            self.screen.blit(control_text, (400, info_y + i * 15))

        if self.paused:
            pause_text = self.big_font.render("PAUSED", True, self.RED)
            pause_rect = pause_text.get_rect(center=(self.screen_width // 2, 50))
            self.screen.blit(pause_text, pause_rect)

    def get_action_description(self, action):
        """Get human-readable description of the action"""
        snake1_action, snake2_action = self.agent.decode_action(action)

        action_names = ["Straight", "Left", "Right"]
        return f"Green: {action_names[snake1_action]}, Blue: {action_names[snake2_action]}"

    def run(self):
        print("Q-Learning Agent Demo")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  R: Reset game")
        print("  +/-: Adjust speed")
        print("  ESC: Quit")
        print(f"Agent loaded with {len(self.agent.q_table)} states in Q-table")

        steps = 0
        max_steps = self.width * self.height * 4  # Prevent infinite games

        while self.running:
            self.handle_events()

            if not self.paused and steps < max_steps:
                # Get current state
                state = self.game.get_state()

                # Get action from Q-learning agent
                action = self.agent.get_action(state)
                snake1_action, snake2_action = self.agent.decode_action(action)

                # Take step in game
                reward, next_state, done = self.game.step(snake1_action, snake2_action)
                steps += 1

                if done:
                    print(f"Game Over! Final score: {self.game.score}, Steps: {steps}")
                    steps = max_steps  # Stop updating game

            # Draw everything
            self.draw_game()
            pygame.display.flip()

            # Control game speed
            self.clock.tick(self.game_speed)

        pygame.quit()
        sys.exit()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Watch Q-Learning agent play Snake')
    parser.add_argument('--model-path', type=str, default='models/qlearning_snake.pkl',
                       help='Path to trained Q-learning model')
    parser.add_argument('--width', type=int, default=15, help='Game board width')
    parser.add_argument('--height', type=int, default=15, help='Game board height')
    parser.add_argument('--cell-size', type=int, default=30, help='Cell size in pixels')
    parser.add_argument('--speed', type=int, default=5, help='Initial game speed (FPS)')

    args = parser.parse_args()

    demo = SARSAGameDemo(
        model_path=args.model_path,
        width=args.width,
        height=args.height,
        cell_size=args.cell_size
    )
    demo.game_speed = args.speed
    demo.run()

if __name__ == "__main__":
    main()