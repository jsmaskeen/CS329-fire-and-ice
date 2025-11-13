import random
from .snake import Snake

SURVIVE_REWARD = -0.1
EAT_REWARD = 10
DIE_REWARD = -10

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
        self.snake1 = Snake(startx=self.width//4, starty=self.height//2, direction=(1, 0))
        self.snake2 = Snake(startx=3*self.width//4, starty=self.height//2, direction=(-1, 0))
        
    def step(self, action1, action2):
        reward = 0
        
        if action1 == 0:
            pass
        elif action1 == 1:
            if self.snake1.direction == (0, -1):
                self.snake1.change_direction((-1, 0))
            elif self.snake1.direction == (0, 1):
                self.snake1.change_direction((1, 0))
            elif self.snake1.direction == (1, 0):
                self.snake1.change_direction((0, -1))
            elif self.snake1.direction == (-1, 0):
                self.snake1.change_direction((0, 1))
        elif action1 == 2:
            if self.snake1.direction == (0, -1):
                self.snake1.change_direction((1, 0))
            elif self.snake1.direction == (0, 1):
                self.snake1.change_direction((-1, 0))
            elif self.snake1.direction == (1, 0):
                self.snake1.change_direction((0, 1))
            elif self.snake1.direction == (-1, 0):
                self.snake1.change_direction((0, -1))
                
        if action2 == 0:
            pass
        elif action2 == 1:
            if self.snake2.direction == (0, -1):
                self.snake2.change_direction((-1, 0))
            elif self.snake2.direction == (0, 1):
                self.snake2.change_direction((1, 0))
            elif self.snake2.direction == (1, 0):
                self.snake2.change_direction((0, -1))
            elif self.snake2.direction == (-1, 0):
                self.snake2.change_direction((0, 1))
        elif action2 == 2:
            if self.snake2.direction == (0, -1):
                self.snake2.change_direction((1, 0))
            elif self.snake2.direction == (0, 1):
                self.snake2.change_direction((-1, 0))
            elif self.snake2.direction == (1, 0):
                self.snake2.change_direction((0, 1))
            elif self.snake2.direction == (-1, 0):
                self.snake2.change_direction((0, -1))
        
        if self.snake1.body[0] == self.food:
            self.snake1.eatAndStep()
            self.score += 1
            self.spawn_food()
            reward = EAT_REWARD
        else:
            self.snake1.step()
            
        if self.snake2.body[0] == self.food:
            self.snake2.eatAndStep()
            self.score += 1
            self.spawn_food()
            reward = EAT_REWARD
        else:
            self.snake2.step()

        if self.check_collision(self.snake1.body[0], 0) or self.check_collision(self.snake2.body[0], 1):
            reward = DIE_REWARD
            return reward, self.get_state(), True  # Game over
        elif reward == 0:
            reward = SURVIVE_REWARD

        return reward, self.get_state(), False  # Game continues
    
    # TODO: Get a better state
    def get_state(self):
        head1 = self.snake1.body[0]
        head2 = self.snake2.body[0]

        dir_north1 = self.snake1.direction == (0, -1)
        dir_south1 = self.snake1.direction == (0, 1)
        dir_east1 = self.snake1.direction == (1, 0)
        dir_west1 = self.snake1.direction == (-1, 0)
        
        dir_north2 = self.snake2.direction == (0, -1)
        dir_south2 = self.snake2.direction == (0, 1)
        dir_east2 = self.snake2.direction == (1, 0)
        dir_west2 = self.snake2.direction == (-1, 0)
        
        point_north1 = (head1[0], head1[1] - 1)
        point_south1 = (head1[0], head1[1] + 1)
        point_east1 = (head1[0] + 1, head1[1])
        point_west1 = (head1[0] - 1, head1[1])
        
        point_north2 = (head2[0], head2[1] - 1)
        point_south2 = (head2[0], head2[1] + 1)
        point_east2 = (head2[0] + 1, head2[1])
        point_west2 = (head2[0] - 1, head2[1])
        
        danger_left1 = (dir_north1 and self.check_collision(point_west1, 0)) or \
                       (dir_south1 and self.check_collision(point_east1, 0)) or \
                       (dir_east1 and self.check_collision(point_north1, 0)) or \
                       (dir_west1 and self.check_collision(point_south1, 0))
        danger_right1 = (dir_north1 and self.check_collision(point_east1, 0)) or \
                        (dir_south1 and self.check_collision(point_west1, 0)) or \
                        (dir_east1 and self.check_collision(point_south1, 0)) or \
                        (dir_west1 and self.check_collision(point_north1, 0))
        danger_straight1 = (dir_north1 and self.check_collision(point_north1, 0)) or \
                           (dir_south1 and self.check_collision(point_south1, 0)) or \
                           (dir_east1 and self.check_collision(point_east1, 0)) or \
                           (dir_west1 and self.check_collision(point_west1, 0))
        
        danger_left2 = (dir_north2 and self.check_collision(point_west2, 1)) or \
                       (dir_south2 and self.check_collision(point_east2, 1)) or \
                       (dir_east2 and self.check_collision(point_north2, 1)) or \
                       (dir_west2 and self.check_collision(point_south2, 1))
        danger_right2 = (dir_north2 and self.check_collision(point_east2, 1)) or \
                        (dir_south2 and self.check_collision(point_west2, 1)) or \
                        (dir_east2 and self.check_collision(point_south2, 1)) or \
                        (dir_west2 and self.check_collision(point_north2, 1))
        danger_straight2 = (dir_north2 and self.check_collision(point_north2, 1)) or \
                           (dir_south2 and self.check_collision(point_south2, 1)) or \
                           (dir_east2 and self.check_collision(point_east2, 1)) or \
                           (dir_west2 and self.check_collision(point_west2, 1))

        food_north1 = self.food[1] < head1[1]
        food_south1 = self.food[1] > head1[1]
        food_east1 = self.food[0] > head1[0]
        food_west1 = self.food[0] < head1[0]
        
        food_north2 = self.food[1] < head2[1]
        food_south2 = self.food[1] > head2[1]
        food_east2 = self.food[0] > head2[0]
        food_west2 = self.food[0] < head2[0]
        
        return [danger_left1, danger_right1, danger_straight1, dir_north1, dir_south1, dir_east1, dir_west1, food_north1, food_south1, food_east1, food_west1,
                danger_left2, danger_right2, danger_straight2, dir_north2, dir_south2, dir_east2, dir_west2, food_north2, food_south2, food_east2, food_west2]

    def spawn_food(self):
        while True:
            self.food = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if self.food not in self.snake1.body and self.food not in self.snake2.body:
                break
            
    def check_collision(self, pos, ignore_snake_head):
        if not (0 <= pos[0] < self.width and 0 <= pos[1] < self.height):
            return True
        
        if ignore_snake_head == 0:
            if pos in self.snake2.body:
                return True
        else:
            if pos in self.snake1.body:
                return True
        
        if pos in self.snake1.body[1:]:
            return True
        if pos in self.snake2.body[1:]:
            return True        
        return False