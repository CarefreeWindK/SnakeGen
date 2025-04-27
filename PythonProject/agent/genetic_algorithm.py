import numpy as np
from typing import List, Tuple
from .neural_network import NeuralNetwork
from game.game_engine import SnakeGame
import random
import time
import config as Config

class GeneticAlgorithm:
    def __init__(self):
        self.population = []
        self.generation = 0
        self.best_score = 0
        self.best_score_gen = 0
        self.best_agent = None
        self.best_agent_gen = None
        self.scores_history = []

        # Inicializar población
        self._initialize_population()

    def _initialize_population(self):
        """Inicializa la población con redes neuronales aleatorias"""
        self.population = []
        for _ in range(Config.Config.POPULATION_SIZE):
            nn = NeuralNetwork()
            self.population.append(nn)

    def evaluate_agent(self, agent: NeuralNetwork, num_games: int = 1) -> float:
        """Evalúa un agente jugando múltiples juegos y devuelve su puntuación promedio"""
        total_score = 0
        total_steps = 0
        for _ in range(num_games):
            game = SnakeGame(Config.Config.BOARD_SIZE)
            state = game.get_state()
            game_over = False

            while not game_over:
                action = agent.predict(state)
                game_over, reward = game.move(action)
                state = game.get_state()
                total_steps += 1
            total_score += game.score
        # Fitness compuesto que considera:
        # 1. Puntuación (lo más importante)
        # 2. Eficiencia (puntos por paso)
        avg_score = total_score / num_games
        avg_steps = total_steps / num_games
        efficiency = avg_score / (avg_steps + 1)  # +1 para evitar división por cero
        fitness = (avg_score * 0.7) + (efficiency * 0.3)
        return max(fitness, 0.01)  # Nunca devolver cero
    def evaluate_population(self):
        """Evalúa toda la población y actualiza las puntuaciones"""
        scores = []

        for agent in self.population:
            score = self.evaluate_agent(agent, Config.Config.EVALUATION_GAMES)
            scores.append(score)

            # Actualizar mejor agente si es necesario
            if score > self.best_score_gen:
                self.best_score_gen = score
                self.best_agent_gen = agent

            if score > self.best_score:
                self.best_score = score
                self.best_agent = agent
                if Config.Config.SAVE_BEST_AGENT:
                    self._save_best_agent()


        self.scores_history.append({
            'generation': self.generation,
            'max': max(scores),
            'min': min(scores),
            'avg': sum(scores) / len(scores)
        })

        return scores

    def _save_best_agent(self):
        """Guarda los pesos del mejor agente en un archivo"""
        weights = self.best_agent.get_weights_as_vector()
        np.save(f'best_agent_gen_{self.generation}.npy', weights)

    def selection(self, scores: List[float], method: str = 'tournament') -> List[NeuralNetwork]:
        """Selecciona padres para la siguiente generación"""
        if method == 'roulette':
            return self._roulette_selection(scores)
        elif method == 'tournament':
            return self._tournament_selection(scores)
        elif method == 'elite':
            return self._elite_selection(scores)
        else:
            raise ValueError(f"Método de selección no válido: {method}")

    def _roulette_selection(self, scores: List[float]) -> List[NeuralNetwork]:
        """Selección por ruleta"""
        total = sum(scores)
        if total == 0:
            probabilities = [1 / len(scores)] * len(scores)
        else:
            probabilities = [score / total for score in scores]

        selected_indices = np.random.choice(
            len(self.population),
            size=len(self.population),
            p=probabilities,
            replace=True
        )

        return [self.population[i] for i in selected_indices]

    def _tournament_selection(self, scores: List[float]) -> List[NeuralNetwork]:
        """Selección por torneo"""
        selected = []

        for _ in range(len(self.population)):
            # Seleccionar k individuos aleatorios
            contestants = random.sample(
                list(zip(self.population, scores)),
                min(Config.Config.TOURNAMENT_SIZE, len(self.population))
            )

            # Seleccionar el mejor
            winner = max(contestants, key=lambda x: x[1])[0]
            selected.append(winner)

        return selected

    def _elite_selection(self, scores: List[float]) -> List[NeuralNetwork]:
        """Selección élite"""
        # Ordenar población por puntuación
        sorted_pop = sorted(
            zip(self.population, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Seleccionar los mejores
        elite_size = Config.Config.ELITISM_COUNT
        selected = [agent for agent, _ in sorted_pop[:elite_size]]

        # Completar con selección por ruleta
        # remaining = self._roulette_selection(scores)
        # selected.extend(remaining[:len(self.population) - elite_size])

        return selected

    def crossover(self, parent1: NeuralNetwork, parent2: NeuralNetwork) -> Tuple[NeuralNetwork, NeuralNetwork]:
        """Operador de cruce entre dos padres"""


        # Convertir pesos a vectores
        weights1 = parent1.get_weights_as_vector()
        weights2 = parent2.get_weights_as_vector()

        # Punto de cruce aleatorio
        crossover_point = random.randint(1, len(weights1) - 1)

        # Realizar cruce
        child1_weights = np.concatenate([weights1[:crossover_point], weights2[crossover_point:]])
        child2_weights = np.concatenate([weights2[:crossover_point], weights1[crossover_point:]])

        # Reconstruir los vectores como listas de matrices
        child1_weight_matrices = self._reconstruct_weights(child1_weights, parent1.layer_sizes)
        child2_weight_matrices = self._reconstruct_weights(child2_weights, parent2.layer_sizes)

        # Crear nuevos agentes con las estructuras de pesos reconstruidas
        child1 = NeuralNetwork(weights=child1_weight_matrices)
        child2 = NeuralNetwork(weights=child2_weight_matrices)

        return child1, child2

    def _reconstruct_weights(self, weight_vector: np.ndarray, layer_sizes: List[int]) -> List[np.ndarray]:
        """Reconstruye una lista de matrices de pesos a partir de un vector unidimensional"""
        weight_matrices = []
        ptr = 0

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]

            matrix_size = input_size * output_size

            # Extraer porciones del vector y convertirlas en una matriz
            matrix = weight_vector[ptr:ptr + matrix_size].reshape((output_size, input_size))
            weight_matrices.append(matrix)

            ptr += matrix_size

        return weight_matrices

    def mutate(self, agent: NeuralNetwork) -> NeuralNetwork:
        """Operador de mutación"""
        weights = agent.get_weights_as_vector()

        for i in range(len(weights)):
            if random.random() < Config.Config.MUTATION_RATE:
                # Mutación gaussiana
                mutation = np.random.normal(0, Config.Config.MUTATION_STRENGTH)
                weights[i] += mutation

        # Crear nuevo agente mutado
        mutated_agent = NeuralNetwork()
        mutated_agent.set_weights_from_vector(weights)

        return mutated_agent

    def create_next_generation(self, selected: List[NeuralNetwork]) -> List[NeuralNetwork]:
        """Crea la siguiente generación a partir de los padres seleccionados"""
        next_generation = []

        # Cruzar y mutar para crear el resto de la población
        while len(next_generation) < Config.Config.POPULATION_SIZE:
            # Seleccionar dos padres aleatorios

            parent1, parent2 = random.sample(selected, 2)
            if random.random() < 0.1:
                parent1 = selected[0]
                parent2 = self.best_agent
            # Cruzar
            child1, child2 = self.crossover(parent1, parent2)

            # Mutar
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            # Añadir a la nueva generación (asegurando no exceder el tamaño)
            if len(next_generation) < Config.Config.POPULATION_SIZE:
                next_generation.append(child1)
            if len(next_generation) < Config.Config.POPULATION_SIZE:
                next_generation.append(child2)

        return next_generation

    def run_generation(self):
        """Ejecuta una generación completa del algoritmo genético"""
        # Evaluar población actual
        scores = self.evaluate_population()

        # Seleccionar padres
        selected = self.selection(scores, Config.Config.SELECTION_METHOD)

        # Crear nueva generación
        self.population = self.create_next_generation(selected)
        self.generation += 1

        # Mostrar estadísticas
        print(f"Generación {self.generation}:")
        print(f"  Máxima puntuación: {max(scores):.2f}")
        print(f"  Mínima puntuación: {min(scores):.2f}")
        print(f"  Promedio: {sum(scores) / len(scores):.2f}")

        return max(scores)