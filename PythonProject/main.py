import os
import time

import numpy as np
import pygame
import matplotlib.pyplot as plt

from PythonProject.agent.neural_network import NeuralNetwork
from agent.genetic_algorithm import GeneticAlgorithm
from game.game_engine import SnakeGame
from game.visualization import SnakeVisualization
from PythonProject.config import Config

def train_agent():
    """Entrena el agente usando el algoritmo genético"""
    ga = GeneticAlgorithm()
    max_scores = []
    avg_scores = []

    try:
        for _ in range(Config.GENERATIONS):
            max_score = ga.run_generation()
            max_scores.append(max_score)
            avg_scores.append(ga.scores_history[-1]['avg'])

            # Visualizar progreso cada 10 generaciones
            if ga.generation % 10 == 0:
                visualize_training(ga)

        # Graficar resultados finales
        plt.figure(figsize=(12, 6))
        plt.plot(max_scores, label='Máxima puntuación')
        plt.plot(avg_scores, label='Puntuación promedio')
        plt.xlabel('Generación')
        plt.ylabel('Puntuación')
        plt.title('Progreso del Algoritmo Genético')
        plt.legend()
        plt.grid()
        plt.show()

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")

    return ga.best_agent


def visualize_training(ga: GeneticAlgorithm):
    """Visualiza el mejor agente de la generación actual"""
    if ga.best_agent is None:
        return

    game = SnakeGame(Config.BOARD_SIZE)
    vis = SnakeVisualization(game)

    running = True
    game_over = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game_over:
            state = game.get_state()
            action = ga.best_agent_gen.predict(state)
            game_over, _ = game.move(action)

            vis.draw()
            pygame.time.delay(Config.GAME_SPEED)
        else:
            print(f"Puntuación del mejor agente: {game.score}")
            running = False

    vis.close()
    ga.best_agent_gen = None
    ga.best_score_gen = 0


def load_best_agent(generation=None):
    """Carga el mejor agente de una generación específica o el más reciente"""
    try:
        if generation is None:
            # Buscar la generación más alta disponible
            files = [f for f in os.listdir() if f.startswith('best_agent_gen_') and f.endswith('.npy')]
            if not files:
                print("No se encontraron archivos de agentes guardados")
                return None

            # Extraer números de generación (versión robusta)
            gen_numbers = []
            for f in files:
                try:
                    # Eliminar prefijo y sufijo para obtener solo el número
                    num = f.replace('best_agent_gen_', '').replace('.npy', '')
                    gen_numbers.append(int(num))
                except ValueError:
                    continue  # Ignorar archivos con formato incorrecto

            if not gen_numbers:
                print("No se encontraron generaciones válidas")
                return None

            latest_gen = max(gen_numbers)
            generation = latest_gen

        filename = f'best_agent_gen_{generation}.npy'
        weights = np.load(filename)
        agent = NeuralNetwork()
        agent.set_weights_from_vector(weights)
        print(f"✔ Agente cargado de la generación {generation}")
        return agent

    except Exception as e:
        print(f"✖ Error al cargar el agente: {str(e)}")
        return None


def simulate_agent(agent, speed=1.0):
    """Ejecuta una simulación del agente sin visualización de red neuronal"""
    game = SnakeGame(Config.BOARD_SIZE)
    vis = SnakeVisualization(game)  # Versión sin el parámetro del agente

    running = True
    game_over = False

    print("\nControles:")
    print("- ESC: Salir")
    print("- R: Reiniciar simulación")
    print(f"- Velocidad actual: {speed}x")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:  # Reiniciar al presionar R
                    game = SnakeGame(Config.BOARD_SIZE)
                    game_over = False
                    print("\nSimulación reiniciada")

        if not game_over:
            state = game.get_state()
            action = agent.predict(state)
            game_over, _ = game.move(action)

            # Dibuja solo el juego (sin visualización de red)
            vis.draw()
            pygame.time.delay(int(Config.GAME_SPEED / speed))
        else:
            print(f"\nPuntuación final: {game.score}")
            print("Presiona R para reiniciar o ESC para salir")
            running = False
    vis.close()


if __name__ == "__main__":
    print("1. Entrenar nuevo agente")
    print("2. Simular generación más avanzada")
    print("3. Simular generación específica")

    choice = input("Selecciona opción: ")

    if choice == "1":
        best_agent = train_agent()
        simulate_agent(best_agent)
    elif choice == "2":
        agent = load_best_agent()
        if agent:
            simulate_agent(agent)
    elif choice == "3":
        gen = int(input("Número de generación: "))
        agent = load_best_agent(gen)
        if agent:
            simulate_agent(agent)