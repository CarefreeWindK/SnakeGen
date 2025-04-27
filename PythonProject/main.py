import time
import pygame
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    best_agent = train_agent()

    # Mostrar el mejor agente en acción
    if best_agent:
        print("\nMostrando el mejor agente en acción...")
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
                action = best_agent.predict(state)
                game_over, _ = game.move(action)

                vis.draw()
                pygame.time.delay(Config.GAME_SPEED)
            else:
                print(f"Puntuación final: {game.score}")
                running = False

        vis.close()