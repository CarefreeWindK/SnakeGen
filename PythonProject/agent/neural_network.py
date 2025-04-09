import numpy as np
from typing import List

from numpy import signedinteger
from numpy._typing import _32Bit, _64Bit

import config as Config

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int] = None, weights: List[np.ndarray] = None):
        """
        Inicializa la red neuronal.
        Si se proporcionan pesos, se usan esos. Si no, se inicializan aleatoriamente.
        """
        if layer_sizes is None:
            layer_sizes = [
                Config.Config.INPUT_NEURONS,
                Config.Config.HIDDEN_NEURONS,
                Config.Config.OUTPUT_NEURONS
            ]

        self.layer_sizes = layer_sizes

        if weights is not None:
            if len(weights) != len(layer_sizes) - 1:
                raise ValueError("El número de matrices de pesos debe coincidir con el número de capas.")
            self.weights = weights
        else:
            self.weights = []
            for i in range(len(layer_sizes) - 1):
                # Inicialización Xavier/Glorot
                limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
                weight_matrix = np.random.uniform(-limit, limit, (layer_sizes[i + 1], layer_sizes[i]))
                self.weights.append(weight_matrix)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante"""
        x = x.flatten()

        for i, weight in enumerate(self.weights):
            x = np.dot(weight, x)

            # No aplicar activación en la última capa (para usar softmax después)
            if i < len(self.weights) - 1:
                x = self.relu(x)

        return self.softmax(x)

    def relu(self, x: np.ndarray) -> np.ndarray:
        """Función de activación ReLU"""
        return np.maximum(0, x)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Función softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_weights_as_vector(self) -> np.ndarray:
        """Devuelve todos los pesos como un vector unidimensional"""
        return np.concatenate([w.flatten() for w in self.weights])

    def set_weights_from_vector(self, weights_vector: np.ndarray):
        """Establece los pesos a partir de un vector unidimensional"""
        ptr = 0
        for i in range(len(self.layer_sizes) - 1):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]

            total_weights = out_size * in_size
            weight_slice = weights_vector[ptr:ptr + total_weights]
            self.weights[i] = weight_slice.reshape((out_size, in_size))
            ptr += total_weights

    def predict(self, state: np.ndarray) -> int:
        """Muestra las activaciones mientras predice"""
        outputs = self.forward(state)

        if hasattr(self, 'last_activations'):
            print("Activaciones por capa:")
            for i, activation in enumerate(self.last_activations):
                print(f"Capa {i}: {activation}")

        return np.argmax(outputs)