import pygame
import sys
from Game.game import Game

class SnakeGameRenderer:
    def __init__(self, width=20, height=20, cell_size=25):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size + 60  # Extra space for score
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("2-Player Snake Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Initialize game
        self.game = Game(width, height)
        
        # Game state
        self.running = True
        self.game_over = False
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and self.game_over:
                    # Restart game
                    self.game.reset()
                    self.game_over = False
                elif not self.game_over:
                    # Snake 1 controls (WASD)
                    if event.key == pygame.K_w and self.game.snake1.direction != (0, 1):
                        self.game.snake1.change_direction((0, -1))
                    elif event.key == pygame.K_s and self.game.snake1.direction != (0, -1):
                        self.game.snake1.change_direction((0, 1))
                    elif event.key == pygame.K_a and self.game.snake1.direction != (1, 0):
                        self.game.snake1.change_direction((-1, 0))
                    elif event.key == pygame.K_d and self.game.snake1.direction != (-1, 0):
                        self.game.snake1.change_direction((1, 0))
                    
                    # Snake 2 controls (Arrow keys)
                    elif event.key == pygame.K_UP and self.game.snake2.direction != (0, 1):
                        self.game.snake2.change_direction((0, -1))
                    elif event.key == pygame.K_DOWN and self.game.snake2.direction != (0, -1):
                        self.game.snake2.change_direction((0, 1))
                    elif event.key == pygame.K_LEFT and self.game.snake2.direction != (1, 0):
                        self.game.snake2.change_direction((-1, 0))
                    elif event.key == pygame.K_RIGHT and self.game.snake2.direction != (-1, 0):
                        self.game.snake2.change_direction((1, 0))
    
    def draw_cell(self, x, y, color):
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                          self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, self.BLACK, rect, 1)
    
    def draw_game(self):
        # Clear screen
        self.screen.fill(self.BLACK)
        
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.GRAY, rect, 1)
        
        if self.game.food:
            self.draw_cell(self.game.food[0], self.game.food[1], self.RED)
        
        for i, segment in enumerate(self.game.snake1.body):
            if i == 0:
                self.draw_cell(segment[0], segment[1], self.GREEN)
            else:
                color = (0, max(100, 255 - i * 10), 0)
                self.draw_cell(segment[0], segment[1], color)
        
        for i, segment in enumerate(self.game.snake2.body):
            if i == 0:
                self.draw_cell(segment[0], segment[1], self.BLUE)
            else:
                color = (0, 0, max(100, 255 - i * 10))
                self.draw_cell(segment[0], segment[1], color)
        
        # Draw score and info
        score_text = self.font.render(f"Score: {self.game.score}", True, self.WHITE)
        self.screen.blit(score_text, (10, self.screen_height - 50))
        
        if self.game_over:
            # Game over screen
            game_over_text = pygame.font.Font(None, 72).render("GAME OVER", True, self.RED)
            text_rect = game_over_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
            self.screen.blit(game_over_text, text_rect)
            
            restart_text = self.font.render("Press R to Restart", True, self.WHITE)
            restart_rect = restart_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(restart_text, restart_rect)
            
            winner = self.determine_winner()
            if winner:
                winner_text = self.font.render(winner, True, self.YELLOW)
                winner_rect = winner_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 30))
                self.screen.blit(winner_text, winner_rect)
    
    def run(self):
        while self.running:
            self.handle_events()
            
            if not self.game_over:
                self.game_over = self.game.step()
            
            self.draw_game()
            pygame.display.flip()
            
            self.clock.tick(5)
        
        pygame.quit()
        sys.exit()

def main():
    renderer = SnakeGameRenderer(width=25, height=20, cell_size=25)
    renderer.run()

if __name__ == "__main__":
    main()
