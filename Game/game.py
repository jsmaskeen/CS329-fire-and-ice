import random
from .snake import Snake

class Game:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.score = 0
        self.snake1 = Snake(startx=width//4, starty=height//2, direction=(1, 0))
        self.snake2 = Snake(startx=3*width//4, starty=height//2, direction=(-1, 0))
        self.food = None
        self.spawn_food()
        
    def reset(self):
        self.score = 0
        self.snake1 = Snake(startx=self.width//4, starty=self.height//2)
        self.snake2 = Snake(startx=3*self.width//4, starty=self.height//2)
        
    def step(self):
        
        if self.snake1.body[0] == self.food:
            self.snake1.eatAndStep()
            self.score += 1
            self.spawn_food()
        else:
            self.snake1.step()
            
        if self.snake2.body[0] == self.food:
            self.snake2.eatAndStep()
            self.score += 1
            self.spawn_food()
        else:
            self.snake2.step()
        
        if self.check_collision():
            return True  # Game over
        
        return False  # Game continues
        
    def get_state(self):
        pass
    
    def spawn_food(self):
        while True:
            self.food = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if self.food not in self.snake1.body and self.food not in self.snake2.body:
                break
            
    def check_collision(self):
        head1 = self.snake1.body[0]
        head2 = self.snake2.body[0]
        
        # Check wall collisions
        if not (0 <= head1[0] < self.width and 0 <= head1[1] < self.height):
            return True
        if not (0 <= head2[0] < self.width and 0 <= head2[1] < self.height):
            return True
        
        # Check self collisions
        if head1 in self.snake1.body[1:]:
            return True
        if head2 in self.snake2.body[1:]:
            return True
        
        # Check snake collisions
        if head1 in self.snake2.body:
            return True
        if head2 in self.snake1.body:
            return True
        
        return False