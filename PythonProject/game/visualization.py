import pygame
import numpy as np
from pygame import gfxdraw
from .game_engine import SnakeGame
from typing import Tuple
from PythonProject.config import Config

class SnakeVisualization:
    def __init__(self, game: SnakeGame):
        self.game = game
        self.cell_size = Config.CELL_SIZE
        self.width = self.game.board_size[0] * self.cell_size
        self.height = self.game.board_size[1] * self.cell_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake AI')

        # Colores
        self.bg_color = (40, 40, 40)
        self.snake_color = (100, 200, 100)
        self.head_color = (200, 50, 50)
        self.food_color = (200, 50, 50)
        self.grid_color = (60, 60, 60)

    def draw(self):
        """Dibuja el estado actual del juego"""
        self.screen.fill(self.bg_color)

        # Dibujar cuadrícula
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, self.grid_color, (0, y), (self.width, y))

        # Dibujar serpiente
        for i, (x, y) in enumerate(self.game.snake):
            rect = pygame.Rect(
                x * self.cell_size,
                y * self.cell_size,
                self.cell_size, self.cell_size
            )

            if i == 0:  # Cabeza
                pygame.draw.rect(self.screen, self.head_color, rect)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)
            else:  # Cuerpo
                pygame.draw.rect(self.screen, self.snake_color, rect)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

        # Dibujar comida
        food_x, food_y = self.game.food
        food_rect = pygame.Rect(
            food_x * self.cell_size,
            food_y * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.food_color, food_rect)

        # Mostrar puntuación
        font = pygame.font.SysFont('Arial', 20)
        score_text = font.render(f'Score: {self.game.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (5, 5))

        pygame.display.flip()

    def close(self):
        pygame.quit()