import numpy as np
import random
from typing import Tuple, List, Optional
import config as Config

class SnakeGame:
    def __init__(self, board_size: Tuple[int, int] = (20, 20)):
        self.board_size = board_size
        self.reset()

    def reset(self):
        """Reinicia el juego a su estado inicial"""
        width, height = self.board_size
        self.snake = [(width // 2, height // 2)]
        self.direction = (1, 0)  # Empieza moviéndose a la derecha
        self.food = self._generate_food()
        self.score = 0
        self.steps = 0
        self.reward = 0
        self.game_over = False
        self.valid_moves = {
            (1, 0): [(1, 0), (0, 1), (0, -1)],  # Derecha
            (-1, 0): [(-1, 0), (0, 1), (0, -1)],  # Izquierda
            (0, 1): [(0, 1), (1, 0), (-1, 0)],  # Abajo
            (0, -1): [(0, -1), (1, 0), (-1, 0)]  # Arriba
        }

    def _generate_food(self) -> Tuple[int, int]:
        """Genera comida en una posición aleatoria no ocupada por la serpiente"""
        width, height = self.board_size
        empty_cells = [(x, y) for x in range(width) for y in range(height)
                       if (x, y) not in self.snake]
        return random.choice(empty_cells) if empty_cells else (0, 0)

    def get_state(self) -> np.ndarray:
        """Devuelve el estado actual del juego como un array numpy de 24 elementos"""
        head_x, head_y = self.snake[0]
        width, height = self.board_size

        state = []

        # 1. Dirección a la comida (4 valores)
        food_dx = self.food[0] - head_x
        food_dy = self.food[1] - head_y
        state.extend([
            food_dx / width,
            food_dy / height,
            1 if food_dx > 0 else 0,  # Comida a la derecha
            1 if food_dy > 0 else 0  # Comida abajo
        ])

        # 2. Distancia a obstáculos en 8 direcciones (16 valores)
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1),
                      (-1, 0), (-1, -1), (0, -1), (1, -1)]

        for dx, dy in directions:
            distance = 0
            x, y = head_x, head_y
            wall_found = False
            body_found = False

            while 0 <= x < width and 0 <= y < height:
                distance += 1
                x += dx
                y += dy

                # Verificar si hay cuerpo de la serpiente en esta posición
                if not body_found and (x, y) in self.snake[1:]:
                    body_found = True

            # Normalizar distancia
            max_dist = max(width, height)
            state.extend([
                distance / max_dist,
                1 if wall_found else 0,
                1 if body_found else 0
            ])

        # 3. Dirección actual de la serpiente (4 valores one-hot)
        dir_vec = [0, 0, 0, 0]
        if self.direction == (1, 0):
            dir_vec[0] = 1  # Derecha
        elif self.direction == (-1, 0):
            dir_vec[1] = 1  # Izquierda
        elif self.direction == (0, 1):
            dir_vec[2] = 1  # Abajo
        elif self.direction == (0, -1):
            dir_vec[3] = 1  # Arriba

        state.extend(dir_vec)

        # Asegurar que tenemos exactamente 24 valores
        if len(state) < 24:
            state.extend([0] * (24 - len(state)))
        elif len(state) > 24:
            state = state[:24]

        return np.array(state, dtype=np.float32)

    def move(self, action: int) -> Tuple[bool, int]:
        """
        Realiza un movimiento en el juego.
        action: 0=derecha, 1=izquierda, 2=abajo, 3=arriba (relativo a la dirección actual)
        Devuelve: (game_over, reward)
        """
        if self.game_over:
            return True, self.reward

        self.steps += 1

        # Determinar nueva dirección
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        possible_moves = self.valid_moves[self.direction]

        if 0 <= action < len(possible_moves):
            new_direction = possible_moves[action]
        else:
            new_direction = self.direction  # Mantener dirección si acción no válida

        self.direction = new_direction

        # Mover la serpiente
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Verificar colisiones
        width, height = self.board_size
        if (new_head in self.snake or
                new_head[0] < 0 or new_head[0] >= width or
                new_head[1] < 0 or new_head[1] >= height or
                self.steps >= Config.Config.MAX_STEPS):
            self.game_over = True
            self.score-=0.05
            return True,self.reward   # Penalización por perder

        # Mover la serpiente
        self.snake.insert(0, new_head)

        # Verificar si comió comida
        if new_head == self.food:
            self.score += 1
            self.steps = 0
            self.reward += 1
            self.food = self._generate_food()
        else:
            self.snake.pop()  # Mantener longitud si no comió

        # Pequeña penalización por paso para incentivar eficiencia
        self.score -= 0.01

        return self.game_over, self.reward