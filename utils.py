# utils.py
import pygame

def init_pygame(window_size):
    pygame.init()
    screen = pygame.display.set_mode((window_size, window_size))
    clock = pygame.time.Clock()
    return screen, clock

def draw_grid(screen, window_size, cell_size):
    screen.fill((255, 255, 255))
    for x in range(0, window_size, cell_size):
        pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, window_size))
    for y in range(0, window_size, cell_size):
        pygame.draw.line(screen, (200, 200, 200), (0, y), (window_size, y))

def draw_agent_and_food(screen, cell_size, player_pos, food_pos):
    agent_rect = pygame.Rect(player_pos[0] * cell_size, player_pos[1] * cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, (0, 0, 0), agent_rect)
    food_rect = pygame.Rect(food_pos[0] * cell_size, food_pos[1] * cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, (0, 255, 0), food_rect)

