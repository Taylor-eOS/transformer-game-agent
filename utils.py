import random
import pygame
import numpy as np

grid_size = 10
cell_size = 60
window_size = grid_size * cell_size

def init_pygame(window_size):
    pygame.init()
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Agent Visualization")
    clock = pygame.time.Clock()
    return screen, clock

def draw_grid(screen, window_size, cell_size):
    screen.fill((255, 255, 255))
    for x in range(0, window_size, cell_size):
        pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, window_size))
    for y in range(0, window_size, cell_size):
        pygame.draw.line(screen, (200, 200, 200), (0, y), (window_size, y))

def draw_agent_food_poison(screen, cell_size, player_pos, food_pos, poison_wall):
    agent_rect = pygame.Rect(player_pos[0]*cell_size, player_pos[1]*cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, (0, 0, 0), agent_rect)
    food_rect = pygame.Rect(food_pos[0]*cell_size, food_pos[1]*cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, (0, 255, 0), food_rect)
    for pos in poison_wall:
        poison_rect = pygame.Rect(pos[0]*cell_size, pos[1]*cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (255, 0, 255), poison_rect)

def place_poison_wall(player_pos, food_pos):
    orientation = random.choice(['horizontal', 'vertical'])
    if orientation == 'horizontal':
        x = random.randint(0, grid_size - 3)
        y = random.randint(0, grid_size - 1)
        wall = [tuple((x+i, y)) for i in range(3)]
    else:
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 3)
        wall = [tuple((x, y+i)) for i in range(3)]
    while any(pos == tuple(player_pos) or pos == tuple(food_pos) for pos in wall):
        return place_poison_wall(player_pos, food_pos)
    return wall

def teleport_agent(player_pos, food_pos, poison_wall):
    while True:
        new_pos = np.array([random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)])
        if tuple(new_pos) != tuple(food_pos) and tuple(new_pos) not in poison_wall:
            return new_pos

def calculate_distance_feature(player_pos, food_pos):
    distance = np.linalg.norm(player_pos - food_pos)
    max_distance = np.sqrt(2) * (grid_size - 1)
    scaled_distance = distance / max_distance
    return scaled_distance

