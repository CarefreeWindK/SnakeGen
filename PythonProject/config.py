import numpy as np


class Config:
    # Configuración del juego
    BOARD_SIZE = (8, 8)
    CELL_SIZE = 20
    GAME_SPEED = 100  # ms entre actualizaciones
    MAX_STEPS = 50  # Máximo de pasos por partida

    # Configuración de la red neuronal
    INPUT_NEURONS = 24  # Características de entrada (visión en 8 direcciones, distancia a paredes, etc.)
    HIDDEN_NEURONS = 16
    OUTPUT_NEURONS = 4  # Arriba, abajo, izquierda, derecha

    # Configuración del algoritmo genético
    POPULATION_SIZE = 400
    GENERATIONS = 100
    MUTATION_RATE = 0.02
    MUTATION_STRENGTH = 0.5
    CROSSOVER_RATE = 0.5
    SELECTION_METHOD = 'roulette'  # 'tournament', 'roulette', or 'elite'
    TOURNAMENT_SIZE = 5
    ELITISM_COUNT = 4

    # Configuración de evaluación
    EVALUATION_GAMES = 3  # Número de juegos por evaluación
    SAVE_BEST_AGENT = True