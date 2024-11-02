import sys
import random
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import init_pygame, draw_grid, draw_agent_food_poison, calculate_distance_feature, grid_size, cell_size, window_size, place_poison_wall, teleport_agent

entropy_coefficient = 0.01
max_episodes = 2000
visualize_after = 100
food_reached_count = 0
max_steps = 2000

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
agent = TransformerAgent(input_size=9, num_actions=4)
optimizer = optim.Adam(agent.parameters(), lr=0.001)
screen = None
clock = None
actions = {0: np.array([0, -1]), 1: np.array([0, 1]), 2: np.array([-1, 0]), 3: np.array([1, 0])}

def calculate_reward(old_pos, new_pos, food_pos, poison_wall):
    if tuple(new_pos) in poison_wall:
        print('Stepped on poison wall----------------------------------')
        return -50
    old_dist = np.linalg.norm(old_pos - food_pos)
    new_dist = np.linalg.norm(new_pos - food_pos)
    if np.array_equal(new_pos, food_pos):
        return 60
    else:
        distance_change = old_dist - new_dist
        if distance_change > 0:
            return distance_change * 10
        else:
            return distance_change * 5

for episode in range(max_episodes):
    player_pos = np.array([random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)])
    food_pos = np.array([random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)])
    poison_wall = place_poison_wall(player_pos, food_pos)
    total_reward = 0
    step = 0
    done = False
    teleported = False
    last_teleport_step = -max_steps
    visualize = food_reached_count >= visualize_after
    if visualize and screen is None:
        screen, clock = init_pygame(window_size)
    while not done:
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        rel_food_pos = (food_pos - player_pos) / grid_size
        rel_poison_wall = [((pos[0] - player_pos[0]) / grid_size, (pos[1] - player_pos[1]) / grid_size) for pos in poison_wall]
        flattened_poison = [coord for pos in rel_poison_wall for coord in pos]
        distance_feature = calculate_distance_feature(player_pos, food_pos)
        state = np.concatenate((rel_food_pos, flattened_poison, [distance_feature]))
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        logits = agent(state_tensor)
        action_probs = torch.softmax(logits, dim=-1)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample().item()
        move = actions[action]
        next_pos = player_pos + move
        next_pos = np.clip(next_pos, 0, grid_size - 1)
        reward = calculate_reward(player_pos, next_pos, food_pos, poison_wall)
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
            draw_agent_food_poison(screen, cell_size, player_pos, food_pos, poison_wall)
            pygame.display.flip()
            clock.tick(10)
        distance_to_food = np.linalg.norm(player_pos - food_pos)
        print(f"Episode {episode + 1}, Step {step + 1}: Distance to food: {round(distance_to_food,2)}")
        step += 1
        if step > max_steps and step > last_teleport_step + max_steps // 2:
            player_pos = teleport_agent(player_pos, food_pos, poison_wall)
            print(f"Agent teleported at episode {episode + 1}, step {step}")
            teleported = True
            last_teleport_step = step  # Update the step count for last teleportation
        if np.array_equal(player_pos, food_pos):
            #print(f"Food reached at episode {episode + 1}, step {step}")
            food_reached_count += 1
            food_pos = np.array([random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)])
            poison_wall = place_poison_wall(player_pos, food_pos)
            done = True

