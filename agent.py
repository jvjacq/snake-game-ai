import torch
import random
import numpy as np
from collections import deque
from game import Snake, Direction, Point, BLOCK_SIZE
from model import LinearNet,  Trainer
from plotter import plot

# Constants.
MAX_MEMORY = 100000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
DECAY_RATE = 0.99
INPUT_SIZE = 11
HIDDEN_SIZE = 512
OUTPUT_SIZE = 3
GAMMA = 0.9
EPSILON = 1

class Agent:
    def __init__(self):
        # Initialise the agent with a model, trainer, and memory.
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.number_of_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = Trainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game):
        # Generate the current state of the game for the agent.
        head = game.snake[0]
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)
        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)

        # Check which direction the agent is moving.
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN
        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT

        state = [
            # The direction the agent is moving.
            direction_up,
            direction_down,
            direction_left,
            direction_right,

            # The direction the food is located in.
            game.food.y < game.head.y,
            game.food.y > game.head.y,
            game.food.x < game.head.x,
            game.food.x > game.head.x,

            # Whether there is danger to the front.
            (direction_up and game.is_collision(point_up)) or
            (direction_down and game.is_collision(point_down)) or
            (direction_left and game.is_collision(point_left)) or
            (direction_right and game.is_collision(point_right)),

            # Whether there is danger to the right.
            (direction_up and game.is_collision(point_right)) or
            (direction_down and game.is_collision(point_left)) or
            (direction_left and game.is_collision(point_up)) or
            (direction_right and game.is_collision(point_down)),

            # Whether there is danger to the left.
            (direction_up and game.is_collision(point_left)) or
            (direction_down and game.is_collision(point_right)) or
            (direction_left and game.is_collision(point_down)) or
            (direction_right and game.is_collision(point_up))
            ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        # Store the agent's experience in memory.
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        # Train the model using experiences from memory.
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory
        states, actions, rewards, next_states, game_overs = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        # Train the model with a single experience.
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # Get the agent's action based on the current state.
        move = [0, 0, 0]
        if random.uniform(0, 1) < self.epsilon:
            random_move = random.randint(0, 2)
            move[random_move] = 1
        else:
            state_zero = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_zero)
            predicted_move = torch.argmax(prediction).item()
            move[predicted_move] = 1
        return move
    
    def update_epsilon(self):
        # Update epsilon to decay exploration over time.
        if self.epsilon > 0.1:
            self.epsilon *= DECAY_RATE
        elif self.epsilon > 0:
            self.epsilon = 0

def train():
    # Main training loop for the agent.
    record = 0
    total_scores = 0
    plot_scores = []
    plot_mean_scores = []
    agent = Agent()
    game = Snake()
    
    while True:
        # Get the current state, action, and perform a game step.
        current_state = agent.get_state(game)
        move = agent.get_action(current_state)
        reward, game_over, score = game.play_step(move)
        new_state = agent.get_state(game)

        # Train agent on the short-term memory and store the experience.
        agent.train_short_memory(current_state, move, reward, new_state, game_over)
        agent.remember(current_state, move, reward, new_state, game_over)

        if game_over:
            # Update epsilon, train on long-term memory, and save the model.
            agent.update_epsilon()
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            # Save the model if a new score record is set.
            if score > record:
                record = score
                agent.model.save()

            # Print progress and update plots.
            print('Game', agent.number_of_games, 'Score', score, 'Record', record)
            plot_scores.append(score)
            total_scores += score
            mean_scores = total_scores / agent.number_of_games
            plot_mean_scores.append(mean_scores)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    # Start the training process.
    train()