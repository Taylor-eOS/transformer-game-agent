import sys
import random
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from utils import init_pygame, draw_grid, draw_agent_food_poison, calculate_distance_feature, place_poison_wall, teleport_agent, GRID_SIZE, CELL_SIZE, WINDOW_SIZE

ENTROPY_COEFFICIENT = 0.03
MAX_EPISODES = 5000
MAX_STEPS = 1000
VISUALIZE_AFTER = 1200
FOOD_REACHED_COUNT = 0
STATE_HISTORY = 32

class TransformerAgent(nn.Module):
    def __init__(self, input_size, num_actions, embed_size=64, num_heads=4, num_layers=2):
        super(TransformerAgent, self).__init__()
        self.embedding = nn.Linear(input_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, STATE_HISTORY, embed_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, num_actions)
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        logits = self.fc_out(x)
        return logits

def a_star(start, goal, obstacles, grid_size):
    open_set = set()
    open_set.add(tuple(start))
    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): np.linalg.norm(np.array(start) - np.array(goal))}
    while open_set:
        current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))
        if current == tuple(goal):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        open_set.remove(current)
        neighbors = get_neighbors(current, grid_size)
        for neighbor in neighbors:
            if neighbor in obstacles:
                continue
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal))
                open_set.add(neighbor)
    return []

def get_neighbors(pos, grid_size):
    x, y = pos
    neighbors = []
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size:
            neighbors.append((nx, ny))
    return neighbors

class NavigationController:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.current_path = []
        self.path_index = 0
    def set_target(self, start, target, obstacles):
        path = a_star(start, target, obstacles, self.grid_size)
        if path:
            self.current_path = path
            self.path_index = 0
        else:
            self.current_path = []
            self.path_index = 0
    def get_next_move(self, current_pos):
        if self.path_index < len(self.current_path):
            next_pos = self.current_path[self.path_index]
            self.path_index += 1
            move = np.array(next_pos) - np.array(current_pos)
            return move
        else:
            return np.array([0, 0])

def calculate_reward(old_pos, new_pos, food_pos, poison_wall, episode, last_pos):
    per_step_punishment = -1
    staying_punishment = -2 if np.array_equal(new_pos, old_pos) else 0  
    revisiting_punishment = -3 if np.array_equal(new_pos, last_pos) else 0  
    reward = per_step_punishment + staying_punishment + revisiting_punishment
    for pos in poison_wall:
        if tuple(new_pos) == pos:
            if episode < VISUALIZE_AFTER // 2:
                reward += 0
            else:
                print('------------------------------')
                reward += -30  
    old_dist = np.linalg.norm(old_pos - food_pos)
    new_dist = np.linalg.norm(new_pos - food_pos)
    if np.array_equal(new_pos, food_pos):
        reward += 110  
    else:
        distance_change = old_dist - new_dist
        if distance_change > 0:
            reward += distance_change * 0.4
        else:
            reward += distance_change * 0.2  
    return reward

agent = TransformerAgent(input_size=12, num_actions=4)
optimizer = optim.Adam(agent.parameters(), lr=0.001)
actions = {0: np.array([0, -1]), 1: np.array([0, 1]), 2: np.array([-1, 0]), 3: np.array([1, 0])}
screen = None
clock = None
nav_controller = NavigationController(GRID_SIZE)
current_target = None
path = []
state_buffer = deque(maxlen=STATE_HISTORY)
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
    nav_controller.current_path = []
    nav_controller.path_index = 0
    current_target = None
    initial_rel_food_x, initial_rel_food_y = (food_pos - player_pos) / GRID_SIZE
    initial_rel_poison = []
    initial_dist_poison = []
    for pos in poison_wall:
        rel_x = (pos[0] - player_pos[0]) / GRID_SIZE
        rel_y = (pos[1] - player_pos[1]) / GRID_SIZE
        distance = calculate_distance_feature(player_pos, pos)
        initial_rel_poison.extend([rel_x, rel_y])
        initial_dist_poison.append(distance)
    initial_distance_to_food = calculate_distance_feature(player_pos, food_pos)
    initial_state = np.array([initial_rel_food_x, initial_rel_food_y] + initial_rel_poison + initial_dist_poison + [initial_distance_to_food])
    for _ in range(STATE_HISTORY):
        state_buffer.append(initial_state)
    last_pos = player_pos.copy()
    while not done:
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        rel_food_x, rel_food_y = (food_pos - player_pos) / GRID_SIZE
        rel_poison = []
        dist_poison = []
        for pos in poison_wall:
            rel_x = (pos[0] - player_pos[0]) / GRID_SIZE
            rel_y = (pos[1] - player_pos[1]) / GRID_SIZE
            distance = calculate_distance_feature(player_pos, pos)
            rel_poison.extend([rel_x, rel_y])
            dist_poison.append(distance)
        distance_to_food = calculate_distance_feature(player_pos, food_pos)
        current_state = np.array([rel_food_x, rel_food_y] + rel_poison + dist_poison + [distance_to_food])
        state_buffer.append(current_state)
        state_sequence = np.array(state_buffer)
        state_tensor = torch.tensor(state_sequence, dtype=torch.float32).unsqueeze(0)
        logits = agent(state_tensor)
        action_probs = torch.softmax(logits, dim=-1)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample().item()
        move = actions[action]
        next_pos = player_pos + move
        next_pos = np.clip(next_pos, 0, GRID_SIZE - 1)
        reward = calculate_reward(player_pos, next_pos, food_pos, poison_wall, episode, last_pos)
        total_reward += reward
        last_pos = player_pos.copy()
        player_pos = next_pos
        optimizer.zero_grad()
        log_prob = action_distribution.log_prob(torch.tensor(action))
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
        loss = -log_prob * reward - ENTROPY_COEFFICIENT * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()
        if visualize:
            draw_grid(screen, WINDOW_SIZE, CELL_SIZE)
            draw_agent_food_poison(screen, CELL_SIZE, player_pos, food_pos, poison_wall)
            pygame.display.flip()
            clock.tick(10)
        distance_to_food = np.linalg.norm(player_pos - food_pos)
        print(f"Episode {episode + 1}/{VISUALIZE_AFTER}, Step {step + 1}: Distance to food: {round(distance_to_food,2)}")
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
        if step >= MAX_STEPS:
            done = True

