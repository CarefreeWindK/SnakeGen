import pygame
import numpy as np
from typing import Optional


class NeuralNetworkVisualizer:
    def __init__(self, network, position, size):
        self.network = network
        self.position = position
        self.size = size
        self.node_radius = 15
        self.layer_spacing = size[0] / (len(network.weights) + 1)
        self.node_colors = [
            (100, 200, 100),  # Capa de entrada (verde)
            (200, 200, 100),  # Capas ocultas (amarillo)
            (200, 100, 100)  # Capa de salida (rojo)
        ]

    def draw(self, surface, current_activation: Optional[np.ndarray] = None):
        """Dibuja la red neuronal en la superficie de Pygame"""
        for layer_idx in range(len(self.network.weights) + 1):
            # Determinar número de nodos en esta capa
            if layer_idx == 0:
                num_nodes = self.network.layer_sizes[0]  # Capa de entrada
            else:
                num_nodes = self.network.layer_sizes[layer_idx]  # Capas ocultas/salida

            # Posicionamiento vertical de los nodos
            node_spacing = self.size[1] / (num_nodes + 1)

            for node_idx in range(num_nodes):
                # Posición del nodo
                x = self.position[0] + layer_idx * self.layer_spacing
                y = self.position[1] + (node_idx + 1) * node_spacing

                # Color base del nodo
                color_idx = min(layer_idx, len(self.node_colors) - 1)
                color = self.node_colors[color_idx]

                # Resaltar activación actual (solo para capa de entrada)
                if current_activation is not None and layer_idx == 0 and node_idx < len(current_activation):
                    intensity = min(255, max(0, int(255 * current_activation[node_idx])))
                    color = (intensity, intensity, 100)

                # Asegurar que el color sea válido
                color = tuple(min(255, max(0, c)) for c in color)

                # Dibujar nodo
                pygame.draw.circle(surface, color, (int(x), int(y)), self.node_radius)

                # Dibujar conexiones a la siguiente capa
                if layer_idx < len(self.network.weights):
                    self._draw_connections(surface, layer_idx, x, y, node_idx)

    def _draw_connections(self, surface, layer_idx, x, y, node_idx):
        """Dibuja las conexiones entre nodos"""
        next_num_nodes = self.network.layer_sizes[layer_idx + 1]
        next_node_spacing = self.size[1] / (next_num_nodes + 1)

        for next_node_idx in range(next_num_nodes):
            next_x = self.position[0] + (layer_idx + 1) * self.layer_spacing
            next_y = self.position[1] + (next_node_idx + 1) * next_node_spacing

            # Obtener peso (asegurarse de que los índices son válidos)
            if (next_node_idx < self.network.weights[layer_idx].shape[0] and
                    node_idx < self.network.weights[layer_idx].shape[1]):
                weight = self.network.weights[layer_idx][next_node_idx, node_idx]
            else:
                weight = 0

            # Determinar color y grosor de la línea basado en el peso
            line_color = (150, 150, 150) if weight < 0 else (200, 200, 200)
            line_width = max(1, min(3, int(abs(weight))))

            # Dibujar conexión
            pygame.draw.line(
                surface,
                line_color,
                (x + self.node_radius, y),
                (next_x - self.node_radius, next_y),
                line_width
            )