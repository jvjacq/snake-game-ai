import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font(None, 25)
Point = namedtuple('Point', 'x, y')

# Colours.
WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0, 255, 0)

# Constants.
BLOCK_SIZE = 20
SPEED = 10000

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class Snake:
    def __init__(self, w=500, h=500):
        # Initialise the game.
        self.w = w
        self.h = h
        pygame.display.set_caption('Snake | Score: 0')
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode((self.w, self.h))
        self.reset()
        self.clockwise = [Direction.RIGHT, Direction. DOWN, Direction.LEFT, Direction.UP]

    def reset(self):
        # Reset the game state.
        self.head = Point((self.w // 2) // BLOCK_SIZE * BLOCK_SIZE,
                         (self.h // 2) // BLOCK_SIZE * BLOCK_SIZE)
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)] 
        self.score = 0
        self.food = None       
        self.frame_iteration = 0
        self.direction = Direction.RIGHT
        self._place_food()
        
    def _place_food(self):
        # Place food in a random, but valid, location.
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break
        
    def play_step(self, action):
        # Gather user input.
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move the snake based on the chosen action.
        self._move(action)
        self.snake.insert(0, self.head)
        
        # Check game state.
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > ((self.w + self.h) / 2) * 0.05 * len(self.snake):
            game_over = True
            reward = -1
            return reward, game_over, self.score
                    
        # Check whether to place food.
        if self.head == self.food:
            self.score += 1
            reward = 1
            self._place_food()
        else:
            self.snake.pop()
        
        # Update the interface and continue.
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        # Check for boundary or body collision.
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        elif pt in self.snake[1:]:
            return True
        return False
        
    def _update_ui(self):
        # Display all updated elements.
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.set_caption('Snake | Score: ' + str(self.score))
        pygame.display.flip()
        
    def _move(self, action):
        # Move the snake straight, right, or left.
        index = self.clockwise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_direction = self.clockwise[index]
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (index + 1) % 4
            new_direction = self.clockwise[next_index]
        else:
            next_index = (index - 1) % 4
            new_direction = self.clockwise[next_index]
        
        # Set the snake's new head location.
        self.direction = new_direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        self.head = Point(x, y)