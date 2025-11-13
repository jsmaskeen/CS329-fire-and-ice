class Snake:
    def __init__(self, startx, starty, direction=(1, 0)):
        self.body = [(startx, starty)]
        self.direction = direction
    
    def eatAndStep(self):
        old_tail = self.body[-1]
        self.step()
        self.body.append(old_tail)
        
    def step(self):
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)
        self.body = [new_head] + self.body[:-1]
        
    def change_direction(self, new_direction):
        self.direction = new_direction
        
    def get_action(self, dir):
        if self.direction == dir:
            return 0  # Straight
        
        if (self.direction == (0, -1) and dir == (-1, 0)) or \
           (self.direction == (0, 1) and dir == (1, 0)) or \
           (self.direction == (1, 0) and dir == (0, -1)) or \
           (self.direction == (-1, 0) and dir == (0, 1)):
            return 1  # Left turn
        
        if (self.direction == (0, -1) and dir == (1, 0)) or \
           (self.direction == (0, 1) and dir == (-1, 0)) or \
           (self.direction == (1, 0) and dir == (0, 1)) or \
           (self.direction == (-1, 0) and dir == (0, -1)):
            return 2  # Right turn
        
        return -1