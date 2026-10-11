"""Optuna para SARSA y Q-learning + comparación baseline vs hiperparámetros tuneados.

Para cada environment y cada algoritmo:
  1. Optuna busca los mejores hiperparámetros (objetivo exacto: V^pi(start) de la
     política greedy, calculado con env.P, sin simular episodios de evaluación).
  2. Se entrena con los parámetros por defecto (baseline) y con los encontrados
     por Optuna, sobre semillas NUEVAS (distintas a las usadas al tunear).
  3. Se imprime una tabla comparativa y se dibujan las curvas de aprendizaje.

Uso:
    python optuna_tuning.py                         # grid_stoch y cliff
    python optuna_tuning.py --trials 20 --final-seeds 5
    python optuna_tuning.py --envs grid_det grid_stoch cliff
"""
import argparse

import numpy as np
import optuna
import matplotlib.pyplot as plt

from gridWorld import GridworldEnv
from cliff_walking import CliffWalkingEnv
from algorithms import train_sarsa, train_q_learning
from utils import policy_value, value_iteration, moving_average

ENVS = {
    "grid_det":   dict(make=lambda: GridworldEnv(is_slippery=False), episodes=1500, max_steps=100, window=100),
    "grid_stoch": dict(make=lambda: GridworldEnv(is_slippery=True),  episodes=1500, max_steps=100, window=100),
    "cliff":      dict(make=lambda: CliffWalkingEnv(),               episodes=500,  max_steps=200, window=20),
}
ALGOS = {"SARSA": train_sarsa, "Q-learning": train_q_learning}

EVAL_GAMMA = 0.99                      # gamma FIJO para evaluar (no el que se está tuneando)
TUNING_SEEDS = [0, 1]                  # semillas usadas dentro de Optuna
BASELINE = dict(alpha=0.1, gamma=0.99, epsilon_decay=0.995, epsilon_min=0.01)


def score(trainer, env, params, episodes, max_steps):
    """Objetivo de Optuna: media sobre semillas de V^pi(start) de la política greedy."""
    vals = []
    for seed in TUNING_SEEDS:
        Q, _ = trainer(env, episodes=episodes, epsilon=1.0, max_steps=max_steps,
                       seed=seed, **params)
        vals.append(policy_value(env, Q, EVAL_GAMMA)[env.start_state])
    return float(np.mean(vals))


def make_objective(trainer, env, episodes, max_steps):
    def objective(trial):
        params = dict(
            alpha=trial.suggest_float("alpha", 0.01, 0.9, log=True),
            gamma=trial.suggest_float("gamma", 0.8, 0.999),
            epsilon_decay=trial.suggest_float("epsilon_decay", 0.95, 0.9995),
            epsilon_min=trial.suggest_float("epsilon_min", 0.001, 0.2, log=True),
        )
        return score(trainer, env, params, episodes, max_steps)
    return objective


def run_config(trainer, env, params, episodes, max_steps, seeds):
    """Entrena con unos parámetros en varias semillas.

    Devuelve (curvas de recompensa [seeds x episodes], V(start) greedy por semilla).
    """
    curves, values = [], []
    for seed in seeds:
        Q, rewards = trainer(env, episodes=episodes, epsilon=1.0, max_steps=max_steps,
                             seed=seed, **params)
        curves.append(rewards)
        values.append(policy_value(env, Q, EVAL_GAMMA)[env.start_state])
    return np.array(curves), np.array(values)


def fmt(x):
    return f"{np.mean(x):8.3f} ± {np.std(x):.3f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", choices=ENVS, default=["grid_stoch", "cliff"])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--final-seeds", type=int, default=10)
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    final_seeds = list(range(100, 100 + args.final_seeds))   # distintas a TUNING_SEEDS

    for env_name in args.envs:
        cfg = ENVS[env_name]
        env = cfg["make"]()
        episodes, max_steps, window = cfg["episodes"], cfg["max_steps"], cfg["window"]
        tail = window                                          # últimos episodios para la media final

        V_star, _ = value_iteration(env, EVAL_GAMMA)
        print("=" * 78)
        print(f"ENVIRONMENT: {env_name}  |  V*(start) óptimo = {V_star[env.start_state]:.3f}")
        print("=" * 78)

        fig, axes = plt.subplots(1, len(ALGOS), figsize=(12, 4.5), sharey=True)
        fig.suptitle(f"{env_name}: baseline vs Optuna (media de {args.final_seeds} semillas)")

        for ax, (name, trainer) in zip(axes, ALGOS.items()):
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=0))
            study.optimize(make_objective(trainer, env, episodes, max_steps),
                           n_trials=args.trials)
            best = study.best_params

            print(f"\n{name}: mejores parámetros encontrados")
            for k, v in best.items():
                print(f"    {k:14s} = {v:.4f}  (baseline: {BASELINE[k]})")

            # Ejecución con los parámetros por defecto y con los de Optuna
            runs = {
                "baseline": run_config(trainer, env, BASELINE, episodes, max_steps, final_seeds),
                "optuna":   run_config(trainer, env, best,     episodes, max_steps, final_seeds),
            }

            print(f"\n    {'config':9s} | {'V(start) greedy':20s} | recompensa últimos {tail} eps")
            for label, (curves, values) in runs.items():
                print(f"    {label:9s} | {fmt(values):20s} | {fmt(curves[:, -tail:].mean(axis=1))}")

                ma = moving_average(curves.mean(axis=0), window)
                ax.plot(np.arange(len(ma)) + window - 1, ma, label=label)

            ax.set_title(name)
            ax.set_xlabel("Episodio")
            ax.grid(alpha=0.3)
            ax.legend()
        axes[0].set_ylabel("Recompensa por episodio")
        fig.tight_layout()
        print()

    plt.show()
