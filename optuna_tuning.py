"""Optimización de hiperparámetros con Optuna para SARSA y Q-learning.

Coste bajo por diseño:
  - Objetivo EXACTO y sin ruido: V^pi(start) de la política greedy aprendida,
    calculado resolviendo un sistema lineal con env.P (no se simulan episodios
    de evaluación).
  - Pocos episodios por entrenamiento y 2 semillas por trial.

Uso:
    python optuna_tuning.py --env grid_stoch --trials 30
    python optuna_tuning.py --env cliff --trials 20
"""
import argparse
import json

import numpy as np
import optuna

from gridWorld import GridworldEnv
from cliff_walking import CliffWalkingEnv
from algorithms import train_sarsa, train_q_learning
from utils import policy_value, value_iteration

ENVS = {
    #  nombre:      (fábrica del entorno,                      episodios, max_steps)
    "grid_det":   (lambda: GridworldEnv(is_slippery=False),   1500, 100),
    "grid_stoch": (lambda: GridworldEnv(is_slippery=True),    1500, 100),
    "cliff":      (lambda: CliffWalkingEnv(),                  500, 200),
}
ALGOS = {"SARSA": train_sarsa, "Q-learning": train_q_learning}
EVAL_GAMMA = 0.99                     # gamma FIJO para evaluar (no el que se está tuneando)
SEEDS = [0, 1]
BASELINE = dict(alpha=0.1, gamma=0.99, epsilon_decay=0.995, epsilon_min=0.01)


def score(trainer, env, params, episodes, max_steps):
    """Media sobre semillas de V^pi(start) de la política greedy aprendida."""
    vals = []
    for seed in SEEDS:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVS, default="grid_stoch")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    make_env, episodes, max_steps = ENVS[args.env]
    env = make_env()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    V_star, _ = value_iteration(env, EVAL_GAMMA)
    print(f"Entorno: {args.env} | V*(start) óptimo = {V_star[env.start_state]:.3f}\n")

    report = {}
    for name, trainer in ALGOS.items():
        base = score(trainer, env, BASELINE, episodes, max_steps)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(make_objective(trainer, env, episodes, max_steps),
                       n_trials=args.trials)
        print(f"{name}: baseline = {base:.3f} | mejor = {study.best_value:.3f}")
        for k, v in study.best_params.items():
            print(f"    {k} = {v:.4f}")
        print()
        report[name] = dict(baseline=base, best_value=study.best_value,
                            best_params=study.best_params)

    with open(f"optuna_{args.env}.json", "w") as f:
        json.dump(report, f, indent=2)
