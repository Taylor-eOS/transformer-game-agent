import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from utils import init_pygame, draw_grid, draw_agent_and_food
import pygame

grid_size = 10
cell_size = 60
window_size = grid_size * cell_size
episodes = 1200
entropy_coefficient = 0.01
visualize_after = 100
food_reached_count = 0

class TransformerAgent(nn.Module):
    def __init__(self, input_size, num_actions, embed_size=64, num_heads=4, num_layers=2):
        super(TransformerAgent, self).__init__()
        self.embedding = nn.Linear(input_size, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, num_actions)
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        logits = self.fc_out(x)
        return logits

agent = TransformerAgent(input_size=3, num_actions=4)
optimizer = optim.Adam(agent.parameters(), lr=0.001)
screen, clock = init_pygame(window_size)
actions = {
    0: np.array([0, -1]),
    1: np.array([0, 1]),
    2: np.array([-1, 0]),
    3: np.array([1, 0])}

def calculate_reward(old_pos, new_pos, food_pos):
    old_dist = np.linalg.norm(old_pos - food_pos)
    new_dist = np.linalg.norm(new_pos - food_pos)
    if np.array_equal(new_pos, food_pos):
        reward = 100
    else:
        distance_change = old_dist - new_dist
        if distance_change > 0:
            reward = distance_change * 10
        else:
            reward = distance_change * 5
    return reward

def calculate_distance_feature(player_pos, food_pos):
    distance = np.linalg.norm(player_pos - food_pos)
    max_distance = np.sqrt(2) * (grid_size - 1)
    scaled_distance = distance / max_distance
    return scaled_distance

for episode in range(episodes):
    player_pos = np.array([random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)])
    food_pos = np.array([random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)])
    total_reward = 0
    step = 0
    done = False
    visualize = food_reached_count >= visualize_after
    while not done:
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
        rel_food_pos = (food_pos - player_pos) / grid_size
        distance_feature = calculate_distance_feature(player_pos, food_pos)
        state = np.concatenate((rel_food_pos, [distance_feature]))
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        logits = agent(state_tensor)
        action_probs = torch.softmax(logits, dim=-1)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample().item()
        move = actions[action]
        next_pos = player_pos + move
        next_pos = np.clip(next_pos, 0, grid_size - 1)
        reward = calculate_reward(player_pos, next_pos, food_pos)
        total_reward += reward
        optimizer.zero_grad()
        log_prob = action_distribution.log_prob(torch.tensor(action))
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
        loss = -log_prob * reward - entropy_coefficient * entropy
        loss.backward()
        optimizer.step()
        player_pos = next_pos
        if visualize:
            draw_grid(screen, window_size, cell_size)
            draw_agent_and_food(screen, cell_size, player_pos, food_pos)
            pygame.display.flip()
            clock.tick(10)
        distance_to_food = np.linalg.norm(player_pos - food_pos)
        print(f"Episode {episode + 1}, Step {step + 1}: Distance to food: {round(distance_to_food,2)}")
        if np.array_equal(player_pos, food_pos):
            print(f"Food reached at episode {episode + 1}, step {step + 1}")
            food_reached_count += 1
            food_pos = np.array([random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)])
            done = True
        step += 1
    if not visualize:
        print(f"Episode {episode + 1}: Total Reward: {round(total_reward,2)}")
    else:
        print(f"Episode {episode + 1}: Total Reward: {round(total_reward,2)}")

