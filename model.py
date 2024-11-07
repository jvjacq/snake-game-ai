import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LinearNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Network layer initialisation.
        self.linear_one = nn.Linear(input_size, hidden_size)
        self.linear_two = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass with a ReLU function after the first layer.
        x = F.relu(self.linear_one(x))
        x = self.linear_two(x)
        return x
    
    def save(self, file_name='model.pth'):
        # Saving the model's learned parameters to a file.
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Trainer:
    def __init__(self, model, learning_rate, gamma):
        # Initialising the trainer with required values.
        self.learning_rate = learning_rate
        self.model = model
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

    def train_step(self, state, action, reward, next_state, game_over):
        # Convert inputs to PyTorch tensors.
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        # Add a batch dimension if necessary.
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        # Predict Q-values with the current state.
        prediction = self.model(state)
        target = prediction.clone()

        # Update the Q-values based on the reward and the predicted Q-values from the next state.
        for index in range(len(game_over)):
            q_new = reward[index]
            if not game_over[index]:
                q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
            target[index][torch.argmax(action).item()] = q_new

        # Perform backpropagation and optimization.
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()