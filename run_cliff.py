"""SARSA vs Q-learning en Cliff Walking (epsilon CONSTANTE para ver la diferencia on-policy / off-policy)."""
import numpy as np
import matplotlib.pyplot as plt

from cliff_walking import CliffWalkingEnv
from algorithms import train_sarsa, train_q_learning
from utils import moving_average, policy_value, value_iteration

EPISODES, MAX_STEPS = 500, 200
ALPHA, GAMMA = 0.5, 1.0
EPSILON = 0.1          # constante: epsilon_decay=1.0 y epsilon_min=epsilon
N_SEEDS = 10
SYMBOLS = {0: "<", 1: "v", 2: ">", 3: "^"}


def print_policy(Q, env):
    for r in range(env.n_rows):
        line = ""
        for c in range(env.n_cols):
            s = r * env.n_cols + c
            if s in env.cliff_states:
                line += " C "
            elif s == env.goal_state:
                line += " G "
            else:
                line += f" {SYMBOLS[int(np.argmax(Q[s]))]} "
        print(line)
    print()


def run(trainer, env, seed):
    return trainer(env, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA,
                   epsilon=EPSILON, epsilon_decay=1.0, epsilon_min=EPSILON,
                   max_steps=MAX_STEPS, seed=seed)


if __name__ == "__main__":
    env = CliffWalkingEnv()
    _, Q_opt = value_iteration(env, gamma=0.99)

    results = {}
    for name, trainer in [("SARSA", train_sarsa), ("Q-learning", train_q_learning)]:
        curves, Qs = [], []
        for seed in range(N_SEEDS):
            Q, rewards = run(trainer, env, seed)
            curves.append(rewards)
            Qs.append(Q)
        results[name] = (np.mean(curves, axis=0), Qs[0])
        print(f"{name}: recompensa media (últimos 100 episodios, {N_SEEDS} semillas) = "
              f"{np.mean(np.array(curves)[:, -100:]):.1f}")
        print(f"Política greedy (semilla 0):")
        print_policy(Qs[0], env)
        print(f"V(start) greedy exacto = {policy_value(env, Qs[0], 0.99)[env.start_state]:.2f}\n")

    print(f"V*(start) óptimo = {np.max(Q_opt[env.start_state]):.2f}")

    plt.figure(figsize=(9, 5))
    for name, (curve, _) in results.items():
        plt.plot(moving_average(curve, 10), label=name)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa por episodio (media de semillas, ventana 10)")
    plt.title(f"Cliff Walking, epsilon={EPSILON} constante")
    plt.ylim(-150, 0)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("cliff_comparison.png", dpi=120)
    plt.show()
