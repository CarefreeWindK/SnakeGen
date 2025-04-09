import pygame
import numpy as np
from pygame import gfxdraw
from game.game_engine import SnakeGame
import config as Config
from agent.visualization import NeuralNetworkVisualizer
class SnakeVisualization:
    def __init__(self, game: SnakeGame, neural_network=None):
        self.game = game
        self.neural_network = neural_network
        self.cell_size = Config.Config.CELL_SIZE
        self.width = self.game.board_size[0] * self.cell_size
        self.height = self.game.board_size[1] * self.cell_size
        self.nn_width = 400  # Ancho para la visualización de la red

        pygame.init()
        self.screen = pygame.display.set_mode((self.width + self.nn_width, self.height))
        pygame.display.set_caption('Snake AI with Neural Network Visualization')
        self.font = pygame.font.SysFont('Arial', 20)  # <-- Esta línea es crucial
        if neural_network:
            self.nn_visualizer = NeuralNetworkVisualizer(
                    neural_network,
                    position=(self.width + 50, 50),
                    size=(self.nn_width - 100, self.height - 100)
                )
        else:
            self.nn_visualizer = None

        # Colores
        self.bg_color = (40, 40, 40)
        self.snake_color = (100, 200, 100)
        self.head_color = (200, 50, 50)
        self.food_color = (200, 50, 50)
        self.grid_color = (60, 60, 60)

    def draw(self, current_activation=None):
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

        score_text = self.font.render(f'Score: {self.game.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (5, 5))

        # Dibujar la red neuronal si existe
        if self.nn_visualizer:
            try:
                self.nn_visualizer.draw(self.screen, current_activation)
                self._draw_network_info()
            except Exception as e:
                print(f"Error al dibujar red neuronal: {e}")
        pygame.display.flip()

    def _draw_network_info(self):
        """Muestra información sobre la red neuronal"""
        info_texts = [
            f"Arquitectura: {self.neural_network.layer_sizes}",
            f"Neuronas entrada: {self.neural_network.layer_sizes[0]}",
            f"Neuronas ocultas: {self.neural_network.layer_sizes[1]}",
            f"Neuronas salida: {self.neural_network.layer_sizes[2]}"
        ]

        for i, text in enumerate(info_texts):
            text_surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (self.width + 20, 20 + i * 25))
    def close(self):
        pygame.quit()

    # class SnakeVisualization:
    #     def __init__(self, game, neural_network=None):
    #         self.game = game
    #         self.neural_network = neural_network
    #         self.cell_size = Config.Config.CELL_SIZE
    #         self.width = self.game.board_size[0] * self.cell_size
    #         self.height = self.game.board_size[1] * self.cell_size
    #         self.nn_width = 400  # Ancho para la visualización de la red
    #
    #         pygame.init()
    #         self.screen = pygame.display.set_mode((self.width + self.nn_width, self.height))
    #         pygame.display.set_caption('Snake AI with Neural Network Visualization')
    #
    #         # Inicializar visualizador de red neuronal si está disponible
    #         if neural_network:
    #             self.nn_visualizer = NeuralNetworkVisualizer(
    #                 neural_network,
    #                 position=(self.width + 50, 50),
    #                 size=(self.nn_width - 100, self.height - 100)
    #             )
    #         else:
    #             self.nn_visualizer = None
    #
    #         # Fuente para texto
    #         self.font = pygame.font.SysFont('Arial', 20)
    #
    #     def draw(self, current_activation=None):
    #         """Dibuja el juego y la red neuronal"""
    #         self.screen.fill((40, 40, 40))
    #
    #         # Dibujar el juego de Snake
    #         self._draw_game()
    #
    #         # Dibujar la red neuronal si existe
    #         if self.nn_visualizer:
    #             try:
    #                 self.nn_visualizer.draw(self.screen, current_activation)
    #                 self._draw_network_info()
    #             except Exception as e:
    #                 print(f"Error al dibujar red neuronal: {e}")
    #
    #         pygame.display.flip()
    #
    #     def _draw_game(self):
    #         """Dibuja el tablero de juego"""
    #         # [Tu implementación existente del dibujo del juego]
    #         # Dibujar serpiente, comida, tablero, etc.
    #
    #     def _draw_network_info(self):
    #         """Muestra información sobre la red neuronal"""
    #         info_texts = [
    #             f"Arquitectura: {self.neural_network.layer_sizes}",
    #             f"Neuronas entrada: {self.neural_network.layer_sizes[0]}",
    #             f"Neuronas ocultas: {self.neural_network.layer_sizes[1]}",
    #             f"Neuronas salida: {self.neural_network.layer_sizes[2]}"
    #         ]
    #
    #         for i, text in enumerate(info_texts):
    #             text_surface = self.font.render(text, True, (255, 255, 255))
    #             self.screen.blit(text_surface, (self.width + 20, 20 + i * 25))
    #
    #     def close(self):
    #         pygame.quit()