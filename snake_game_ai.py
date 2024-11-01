import pygame
import random
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
SPEED = 10

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class SnakeGame:
    def __init__(self, w=500, h=500):
        self.w = w
        self.h = h

        # Initialise the game.
        pygame.display.set_caption('Snake | Score: 0')
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode((self.w, self.h))
        self.reset()
        
    def _place_food(self):
        # Place food in a random, but valid, location.
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break
        
    def play_step(self):
        # Gather user input.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                self.previous_direction = self.direction
                if event.key == pygame.K_DOWN and self.previous_direction != Direction.UP:
                    self.direction = Direction.DOWN
                elif event.key == pygame.K_UP and self.previous_direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_RIGHT and self.previous_direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_LEFT and self.previous_direction != Direction.RIGHT:
                    self.direction = Direction.LEFT

        # Move the snake.
        self._move(self.direction)
        self.snake.insert(0, self.head)
        
        # Check game state.
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
            
        # Check whether to place food.
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # Update the interface.
        self._update_ui()
        self.clock.tick(SPEED)

        return game_over, self.score
    
    def _is_collision(self):
        # Check for boundary or body collision.
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        elif self.head in self.snake[1:]:
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
        
    def _move(self, direction):
        # Move the snake.
        x = self.head.x
        y = self.head.y
        if direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.RIGHT:
            x += BLOCK_SIZE
        self.head = Point(x, y)

    def loop(self):
        # Game loop.
        while True:
            game_over, self.score = game.play_step()
            if game_over == True:
                break
        game.game_over_screen()

    def reset(self):
        # Start a new game.
        self.head = Point((self.w // 2) // BLOCK_SIZE * BLOCK_SIZE,
                         (self.h // 2) // BLOCK_SIZE * BLOCK_SIZE)
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)] 
        self.score = 0
        self.food = None       
        self.previous_direction = None
        self.direction = Direction.RIGHT
        self._place_food()

    def game_over_screen(self):
        # Display game over message.
        self.display.fill(BLACK)
        print('Final Score: ', self.score)
        text = font.render("Game Over! Final Score: " + str(self.score), True, WHITE)
        retry_text = font.render("Press SPACE to try again or ESC to quit", True, WHITE)
        self.display.blit(text, [self.w // 4, self.h // 2])
        self.display.blit(retry_text, [self.w // 6, self.h // 2 + 30])
        pygame.display.flip()

        # Wait for user input.
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                        self.reset()
                        self.loop()
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()
                                    
if __name__ == '__main__':
    game = SnakeGame()
    game.loop()