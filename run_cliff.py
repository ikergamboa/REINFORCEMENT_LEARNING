"""SARSA vs Q-learning en Cliff Walking, determinista y estocástico.
 
Epsilon CONSTANTE (epsilon_decay=1.0) para que se vea la diferencia on-policy / off-policy.
El modo estocástico usa slip_prob=0.2 (igual que el Gridworld slippery: 0.8 / 0.1 / 0.1).
"""
import numpy as np
import matplotlib.pyplot as plt
 
from cliff_walking import CliffWalkingEnv
from algorithms import train_sarsa, train_q_learning
from utils import moving_average, policy_value, value_iteration
 
EPISODES, MAX_STEPS = 500, 200
ALPHA, GAMMA = 0.5, 1.0
EPSILON = 0.1          # constante: epsilon_decay=1.0 y epsilon_min=epsilon
N_SEEDS = 10
MODES = [("deterministic", 0.0), ("stochastic", 0.2)]
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
    for mode, slip in MODES:
        env = CliffWalkingEnv(slip_prob=slip)
        _, Q_opt = value_iteration(env, gamma=0.99)
        print("=" * 60)
        print(f"CLIFF WALKING - {mode.upper()} (slip_prob={slip})")
        print("=" * 60)
 
        results = {}
        for name, trainer in [("SARSA", train_sarsa), ("Q-learning", train_q_learning)]:
            curves, values, Q0 = [], [], None
            for seed in range(N_SEEDS):
                Q, rewards = run(trainer, env, seed)
                curves.append(rewards)
                values.append(policy_value(env, Q, 0.99)[env.start_state])
                if seed == 0:
                    Q0 = Q
            results[name] = np.mean(curves, axis=0)
            last = np.array(curves)[:, -100:].mean(axis=1)
            print(f"\n{name} ({N_SEEDS} semillas)")
            print(f"  Recompensa media (últimos 100 episodios): {last.mean():.1f} ± {last.std():.1f}")
            print(f"  V(start) de la política greedy:           {np.mean(values):.2f} ± {np.std(values):.2f}")
            print("  Política greedy (semilla 0):")
            print_policy(Q0, env)
 
        print(f"V*(start) óptimo = {np.max(Q_opt[env.start_state]):.2f}\n")
 
        plt.figure(figsize=(9, 5))
        for name, curve in results.items():
            plt.plot(moving_average(curve, 10), label=name)
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa por episodio (media de semillas, ventana 10)")
        plt.title(f"Cliff Walking {mode}, epsilon={EPSILON} constante")
        plt.ylim(-150 if slip == 0 else -300, 0)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
 
    plt.show()
 