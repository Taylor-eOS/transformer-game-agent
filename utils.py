import random
import pygame
import numpy as np

GRID_SIZE = 10
CELL_SIZE = 60
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

def init_pygame(WINDOW_SIZE):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Agent Visualization")
    clock = pygame.time.Clock()
    return screen, clock

def draw_grid(screen, WINDOW_SIZE, CELL_SIZE):
    screen.fill((255, 255, 255))
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (0, y), (WINDOW_SIZE, y))

def draw_agent_food_poison(screen, CELL_SIZE, player_pos, food_pos, poison_wall):
    agent_rect = pygame.Rect(player_pos[0]*CELL_SIZE, player_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, (0, 0, 0), agent_rect)
    food_rect = pygame.Rect(food_pos[0]*CELL_SIZE, food_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, (0, 255, 0), food_rect)
    for pos in poison_wall:
        poison_rect = pygame.Rect(pos[0]*CELL_SIZE, pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (255, 0, 255), poison_rect)

def place_poison_wall(player_pos, food_pos):
    orientation = random.choice(['horizontal', 'vertical'])
    if orientation == 'horizontal':
        x = random.randint(0, GRID_SIZE - 3)
        y = random.randint(0, GRID_SIZE - 1)
        wall = [tuple((x+i, y)) for i in range(3)]
    else:
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 3)
        wall = [tuple((x, y+i)) for i in range(3)]
    while any(pos == tuple(player_pos) or pos == tuple(food_pos) for pos in wall):
        return place_poison_wall(player_pos, food_pos)
    return wall

def teleport_agent(player_pos, food_pos, poison_wall):
    while True:
        new_pos = np.array([random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)])
        if tuple(new_pos) != tuple(food_pos) and tuple(new_pos) not in poison_wall:
            return new_pos

def calculate_distance_feature(player_pos, food_pos):
    distance = np.linalg.norm(player_pos - food_pos)
    max_distance = np.sqrt(2) * (GRID_SIZE - 1)
    scaled_distance = distance / max_distance
    return scaled_distance

