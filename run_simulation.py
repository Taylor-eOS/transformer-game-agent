import sys
import random
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import init_pygame, draw_grid, draw_agent_food_poison, calculate_distance_feature, place_poison_wall, teleport_agent, GRID_SIZE, CELL_SIZE, WINDOW_SIZE

ENTROPY_COEFFICIENT = 0.01
MAX_EPISODES = 2000
VISUALIZE_AFTER = 100
FOOD_REACHED_COUNT = 0
MAX_STEPS = 2000

class TransformerAgent(nn.Module):
    def __init__(self, input_size, num_actions, embed_size=32, num_heads=2, num_layers=1):
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
agent = TransformerAgent(input_size=6, num_actions=4)  # Adjusted input size to 6
optimizer = optim.Adam(agent.parameters(), lr=0.001)
screen = None
clock = None
actions = {0: np.array([0, -1]), 1: np.array([0, 1]), 2: np.array([-1, 0]), 3: np.array([1, 0])}

def calculate_reward(old_pos, new_pos, food_pos, poison_wall):
    if tuple(new_pos) in poison_wall:
        print('--------------------Stepped on poison wall--------------------')
        return -50
    old_dist = np.linalg.norm(old_pos - food_pos)
    new_dist = np.linalg.norm(new_pos - food_pos)
    if np.array_equal(new_pos, food_pos):
        return 50
    else:
        distance_change = old_dist - new_dist
        if distance_change > 0:
            return distance_change * 10
        else:
            return distance_change * 5

for episode in range(MAX_EPISODES):
    player_pos = np.array([random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)])
    food_pos = np.array([random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)])
    poison_wall = place_poison_wall(player_pos, food_pos)
    total_reward = 0
    step = 0
    done = False
    teleported = False
    last_teleport_step = -MAX_STEPS
    visualize = FOOD_REACHED_COUNT >= VISUALIZE_AFTER
    if visualize and screen is None:
        screen, clock = init_pygame(WINDOW_SIZE)
    while not done:
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        rel_food_pos = (food_pos - player_pos) / GRID_SIZE
        if poison_wall:
            closest_poison = min(poison_wall, key=lambda pos: np.linalg.norm(player_pos - pos))
            rel_poison_x = (closest_poison[0] - player_pos[0]) / GRID_SIZE
            rel_poison_y = (closest_poison[1] - player_pos[1]) / GRID_SIZE
            distance_to_poison = calculate_distance_feature(player_pos, closest_poison)
        else:
            rel_poison_x = 0
            rel_poison_y = 0
            distance_to_poison = 0
        distance_to_food = calculate_distance_feature(player_pos, food_pos)
        state = np.concatenate((rel_food_pos, [rel_poison_x, rel_poison_y, distance_to_poison, distance_to_food]))
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        logits = agent(state_tensor)
        action_probs = torch.softmax(logits, dim=-1)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample().item()
        next_pos = player_pos + actions[action]
        next_pos = np.clip(next_pos, 0, GRID_SIZE - 1)
        reward = calculate_reward(player_pos, next_pos, food_pos, poison_wall)
        total_reward += reward
        optimizer.zero_grad()
        log_prob = action_distribution.log_prob(torch.tensor(action))
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
        loss = -log_prob * reward - ENTROPY_COEFFICIENT * entropy
        loss.backward()
        optimizer.step()
        player_pos = next_pos
        if visualize:
            draw_grid(screen, WINDOW_SIZE, CELL_SIZE)
            draw_agent_food_poison(screen, CELL_SIZE, player_pos, food_pos, poison_wall)
            pygame.display.flip()
            clock.tick(10)
        distance_to_food = np.linalg.norm(player_pos - food_pos)
        print(f"Episode {episode + 1}, Step {step + 1}: Distance to food: {round(distance_to_food,2)}")
        step += 1
        if step > MAX_STEPS and step > last_teleport_step + MAX_STEPS // 2:
            player_pos = teleport_agent(player_pos, food_pos, poison_wall)
            print(f"Agent teleported at episode {episode + 1}, step {step}")
            teleported = True
            last_teleport_step = step
        if np.array_equal(player_pos, food_pos):
            FOOD_REACHED_COUNT += 1
            food_pos = np.array([random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)])
            poison_wall = place_poison_wall(player_pos, food_pos)
            done = True

