"""
Competitive Snake Game - Two snakes compete for food
Only the snake that eats the food gets the reward
"""

import random

# Reward structure for competitive game
FOOD_REWARD = 10.0      # Reward for eating food
DEATH_PENALTY = -10.0   # Penalty for dying
SURVIVE_REWARD = -0.1    # No reward for just surviving
OPPONENT_DEATH = 5.0    # Bonus when opponent dies

class Snake:
    """Snake agent in the competitive game"""
    
    def __init__(self, startx, starty, direction=(1, 0), snake_id=1):
        self.body = [(startx, starty)]
        self.direction = direction
        self.snake_id = snake_id
        self.alive = True
        
    def step(self):
        """Move snake one step forward"""
        if not self.alive:
            return
            
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)
        self.body = [new_head] + self.body[:-1]
        
    def eat_and_step(self):
        """Move forward and grow by one segment"""
        if not self.alive:
            return
            
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)
        # Keep the tail when eating
        self.body = [new_head] + self.body
        
    def change_direction(self, new_direction):
        """Change snake's direction"""
        if self.alive:
            self.direction = new_direction
            
    def get_head(self):
        """Get head position"""
        return self.body[0] if self.body else None
        
    def kill(self):
        """Mark snake as dead"""
        self.alive = False


class CompetitiveSnakeGame:
    """
    Competitive 2-player Snake Game
    - Two snakes compete for the same food
    - Only the snake that eats gets the reward
    - Game ends when one or both snakes die
    """
    
    def __init__(self, width=15, height=15):
        self.width = width
        self.height = height
        self.snake1 = None
        self.snake2 = None
        self.food = None
        self.snake1_score = 0
        self.snake2_score = 0
        self.steps = 0
        self.max_steps = width * height * 3  # Prevent infinite games
        self.reset()
        
    def reset(self):
        """Reset the game to initial state"""
        # Place snakes on opposite sides
        self.snake1 = Snake(
            startx=self.width // 4, 
            starty=self.height // 2,
            direction=(1, 0),
            snake_id=1
        )
        self.snake2 = Snake(
            startx=3 * self.width // 4,
            starty=self.height // 2, 
            direction=(-1, 0),
            snake_id=2
        )
        
        self.snake1_score = 0
        self.snake2_score = 0
        self.steps = 0
        self.spawn_food()
        
        return self.get_state()
        
    def spawn_food(self):
        """Spawn food at random location not occupied by snakes"""
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            food_pos = (x, y)
            
            # Check if position is free
            if (food_pos not in self.snake1.body and 
                food_pos not in self.snake2.body):
                self.food = food_pos
                return
                
            attempts += 1
            
        # If can't find free spot, place anywhere
        self.food = (random.randint(0, self.width - 1), 
                     random.randint(0, self.height - 1))
        
    def step(self, action1, action2):
        """
        Execute one step of the game
        
        Args:
            action1: Action for snake 1 (0=straight, 1=left, 2=right)
            action2: Action for snake 2 (0=straight, 1=left, 2=right)
            
        Returns:
            reward1: Reward for snake 1
            reward2: Reward for snake 2
            next_state: New state after the step
            done: Whether game is over
            info: Additional information dictionary
        """
        self.steps += 1
        reward1 = SURVIVE_REWARD
        reward2 = SURVIVE_REWARD
        
        # Apply actions to change directions
        if self.snake1.alive:
            self._apply_action(self.snake1, action1)
        if self.snake2.alive:
            self._apply_action(self.snake2, action2)
        
        # Check if snakes will eat food
        snake1_ate = False
        snake2_ate = False
        
        if self.snake1.alive:
            next_head1 = self._get_next_head(self.snake1)
            if next_head1 == self.food:
                snake1_ate = True
                
        if self.snake2.alive:
            next_head2 = self._get_next_head(self.snake2)
            if next_head2 == self.food:
                snake2_ate = True
        
        # Move snakes (growing if they ate)
        if self.snake1.alive:
            if snake1_ate:
                self.snake1.eat_and_step()
                reward1 += FOOD_REWARD
                self.snake1_score += 1
            else:
                self.snake1.step()
                
        if self.snake2.alive:
            if snake2_ate:
                self.snake2.eat_and_step()
                reward2 += FOOD_REWARD
                self.snake2_score += 1
            else:
                self.snake2.step()
        
        # Spawn new food if any snake ate
        if snake1_ate or snake2_ate:
            self.spawn_food()
        
        # Check collisions
        snake1_died = False
        snake2_died = False
        
        if self.snake1.alive:
            if self._check_collision(self.snake1):
                self.snake1.kill()
                reward1 += DEATH_PENALTY
                snake1_died = True
                
        if self.snake2.alive:
            if self._check_collision(self.snake2):
                self.snake2.kill()
                reward2 += DEATH_PENALTY
                snake2_died = True
        
        # Bonus reward if opponent dies
        if snake2_died and self.snake1.alive:
            reward1 += OPPONENT_DEATH
        if snake1_died and self.snake2.alive:
            reward2 += OPPONENT_DEATH
        
        # Check if game is over
        done = (not self.snake1.alive or 
                not self.snake2.alive or 
                self.steps >= self.max_steps)
        
        next_state = self.get_state()
        
        info = {
            'snake1_score': self.snake1_score,
            'snake2_score': self.snake2_score,
            'snake1_alive': self.snake1.alive,
            'snake2_alive': self.snake2.alive,
            'steps': self.steps,
            'snake1_length': len(self.snake1.body),
            'snake2_length': len(self.snake2.body),
        }
        
        return reward1, reward2, next_state, done, info
    
    def _apply_action(self, snake, action):
        """Apply action to change snake direction"""
        if action == 0:
            # Straight - no change
            pass
        elif action == 1:
            # Turn left
            if snake.direction == (0, -1):  # North -> West
                snake.change_direction((-1, 0))
            elif snake.direction == (0, 1):  # South -> East
                snake.change_direction((1, 0))
            elif snake.direction == (1, 0):  # East -> North
                snake.change_direction((0, -1))
            elif snake.direction == (-1, 0):  # West -> South
                snake.change_direction((0, 1))
        elif action == 2:
            # Turn right
            if snake.direction == (0, -1):  # North -> East
                snake.change_direction((1, 0))
            elif snake.direction == (0, 1):  # South -> West
                snake.change_direction((-1, 0))
            elif snake.direction == (1, 0):  # East -> South
                snake.change_direction((0, 1))
            elif snake.direction == (-1, 0):  # West -> North
                snake.change_direction((0, -1))
    
    def _get_next_head(self, snake):
        """Get where snake's head will be after moving"""
        head = snake.get_head()
        if head is None:
            return None
        dir_x, dir_y = snake.direction
        return (head[0] + dir_x, head[1] + dir_y)
    
    def _check_collision(self, snake):
        """Check if snake collided with wall or any snake body"""
        head = snake.get_head()
        if head is None:
            return True
            
        # Wall collision
        if not (0 <= head[0] < self.width and 0 <= head[1] < self.height):
            return True
            
        # Self collision (head hit own body)
        if head in snake.body[1:]:
            return True
            
        # Collision with other snake
        other_snake = self.snake2 if snake.snake_id == 1 else self.snake1
        if other_snake.alive and head in other_snake.body:
            return True
            
        return False
    
    def get_state(self):
        """
        Get current state representation for both agents
        
        Returns two separate state vectors (one for each snake):
        Each state vector contains:
        - Danger detection (straight, left, right) - 3 features
        - Current direction (N, S, E, W) - 4 features  
        - Food direction relative to snake (N, S, E, W) - 4 features
        - Opponent head direction relative to snake (N, S, E, W) - 4 features
        - Length comparison (longer, equal, shorter) - 3 features
        Total: 18 features per snake
        """
        state1 = self._get_snake_state(self.snake1, self.snake2)
        state2 = self._get_snake_state(self.snake2, self.snake1)
        
        return state1, state2
    
    def _get_snake_state(self, snake, opponent):
        """Get state vector for a specific snake"""
        if not snake.alive:
            return [0] * 18
            
        head = snake.get_head()
        
        # Direction features
        dir_north = int(snake.direction == (0, -1))
        dir_south = int(snake.direction == (0, 1))
        dir_east = int(snake.direction == (1, 0))
        dir_west = int(snake.direction == (-1, 0))
        
        # Danger detection
        point_straight = self._get_next_head(snake)
        point_left = self._get_point_left(snake)
        point_right = self._get_point_right(snake)
        
        danger_straight = int(self._is_dangerous_position(point_straight, snake))
        danger_left = int(self._is_dangerous_position(point_left, snake))
        danger_right = int(self._is_dangerous_position(point_right, snake))
        
        # Food direction
        food_north = int(self.food[1] < head[1])
        food_south = int(self.food[1] > head[1])
        food_east = int(self.food[0] > head[0])
        food_west = int(self.food[0] < head[0])
        
        # Opponent head direction
        if opponent.alive:
            opp_head = opponent.get_head()
            opp_north = int(opp_head[1] < head[1])
            opp_south = int(opp_head[1] > head[1])
            opp_east = int(opp_head[0] > head[0])
            opp_west = int(opp_head[0] < head[0])
        else:
            opp_north = opp_south = opp_east = opp_west = 0
        
        # Length comparison
        my_length = len(snake.body)
        opp_length = len(opponent.body) if opponent.alive else 0
        
        length_longer = int(my_length > opp_length)
        length_equal = int(my_length == opp_length)
        length_shorter = int(my_length < opp_length)
        
        return [
            danger_straight, danger_left, danger_right,
            dir_north, dir_south, dir_east, dir_west,
            food_north, food_south, food_east, food_west,
            opp_north, opp_south, opp_east, opp_west,
            length_longer, length_equal, length_shorter
        ]
    
    def _get_point_left(self, snake):
        """Get position to the left of snake's current direction"""
        head = snake.get_head()
        if snake.direction == (0, -1):  # North
            return (head[0] - 1, head[1])
        elif snake.direction == (0, 1):  # South
            return (head[0] + 1, head[1])
        elif snake.direction == (1, 0):  # East
            return (head[0], head[1] - 1)
        elif snake.direction == (-1, 0):  # West
            return (head[0], head[1] + 1)
    
    def _get_point_right(self, snake):
        """Get position to the right of snake's current direction"""
        head = snake.get_head()
        if snake.direction == (0, -1):  # North
            return (head[0] + 1, head[1])
        elif snake.direction == (0, 1):  # South
            return (head[0] - 1, head[1])
        elif snake.direction == (1, 0):  # East
            return (head[0], head[1] + 1)
        elif snake.direction == (-1, 0):  # West
            return (head[0], head[1] - 1)
    
    def _is_dangerous_position(self, pos, snake):
        """Check if a position would cause collision"""
        if pos is None:
            return True
            
        # Wall
        if not (0 <= pos[0] < self.width and 0 <= pos[1] < self.height):
            return True
            
        # Own body (excluding tail as it will move)
        if pos in snake.body[:-1]:
            return True
            
        # Opponent body
        opponent = self.snake2 if snake.snake_id == 1 else self.snake1
        if opponent.alive and pos in opponent.body:
            return True
            
        return False
