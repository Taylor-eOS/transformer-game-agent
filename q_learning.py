import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

grid_size = 5  # Grid positions range from 0 to grid_size - 1
cell_size = 100  # Size of each cell in pixels
window_size = grid_size * cell_size

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, action_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.fc_out(x)
        return output

agent = DQNAgent(state_size=2, action_size=4)
optimizer = optim.Adam(agent.parameters(), lr=0.001)
criterion = nn.MSELoss()

epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
gamma = 0.9  # Discount factor

episodes = 1000
batch_size = 32
memory = []

pygame.init()
screen = pygame.display.set_mode((window_size, window_size))
clock = pygame.time.Clock()

actions = {
    0: np.array([0, -1]),  # Up
    1: np.array([0, 1]),   # Down
    2: np.array([-1, 0]),  # Left
    3: np.array([1, 0])    # Right
}

for episode in range(episodes):
    player_pos = np.array([random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)])
    food_pos = np.array([random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)])
    total_reward = 0
    for step in range(200):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        state = torch.tensor(player_pos, dtype=torch.float32)
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values = agent(state)
                action = torch.argmax(q_values).item()
        move = actions[action]
        next_pos = player_pos + move
        next_pos = np.clip(next_pos, 0, grid_size - 1)
        reward = -0.1  # Small negative reward to encourage shorter paths
        if np.array_equal(next_pos, food_pos):
            reward = 10.0  # Reward for reaching the food
            done = True
        else:
            done = False
        total_reward += reward
        next_state = torch.tensor(next_pos, dtype=torch.float32)
        memory.append((state, action, reward, next_state, done))
        if len(memory) > 10000:
            memory.pop(0)
        player_pos = next_pos
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states = torch.stack([item[0] for item in batch])
            actions_batch = torch.tensor([item[1] for item in batch])
            rewards = torch.tensor([item[2] for item in batch])
            next_states = torch.stack([item[3] for item in batch])
            dones = torch.tensor([item[4] for item in batch], dtype=torch.float32)
            q_values = agent(states)
            q_values = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = agent(next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)
            loss = criterion(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        screen.fill((255, 255, 255))
        for x in range(0, window_size, cell_size):
            pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, window_size))
        for y in range(0, window_size, cell_size):
            pygame.draw.line(screen, (200, 200, 200), (0, y), (window_size, y))
        agent_rect = pygame.Rect(player_pos[0] * cell_size, player_pos[1] * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (0, 0, 0), agent_rect)
        food_rect = pygame.Rect(food_pos[0] * cell_size, food_pos[1] * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (0, 255, 0), food_rect)
        pygame.display.flip()
        clock.tick(10)
        if done:
            print(f"Episode {episode + 1}, Step {step + 1}: Food reached, Total Reward: {total_reward}")
            break
    else:
        print(f"Episode {episode + 1}: Did not reach food, Total Reward: {total_reward}")
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

