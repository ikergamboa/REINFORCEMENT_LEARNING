import numpy as np
import matplotlib.pyplot as plt

from gridWorld import GridworldEnv
from algorithms import train_sarsa, train_q_learning


def moving_average(values, window=100):
    """
    Calculate moving average for smoother plots.
    """

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid"
    )


def print_policy(Q, env):
    """
    Print the policy learned by the agent.
    """

    symbols = {
        0: "<",   # LEFT
        1: "v",   # DOWN
        2: ">",   # RIGHT
        3: "^"    # UP
    }

    print()

    for row in range(env.n_rows):

        line = ""

        for col in range(env.n_cols):

            state = row * env.n_cols + col

            if state == env.wall_state:

                line += " # "

            elif state == env.start_state:

                line += " S "

            elif state in env.terminal_states:

                if env.terminal_states[state] > 0:
                    line += " G "
                else:
                    line += " H "

            else:

                best_action = np.argmax(Q[state])

                line += f" {symbols[best_action]} "

        print(line)

    print()


def print_q_table(Q):
    """
    Print Q-table.
    """

    print("\nQ-table:")
    print("State | LEFT | DOWN | RIGHT | UP")
    print("----------------------------------")

    for state in range(Q.shape[0]):

        print(
            f"{state:5d} | "
            f"{Q[state, 0]:5.2f} | "
            f"{Q[state, 1]:5.2f} | "
            f"{Q[state, 2]:5.2f} | "
            f"{Q[state, 3]:5.2f}"
        )


def evaluate_policy(Q, env, episodes=100):
    """
    Evaluate a learned policy without exploration.

    The agent always selects the action with
    the highest Q-value.
    """

    rewards = []

    for episode in range(episodes):

        state, info = env.reset()

        total_reward = 0

        for step in range(100):

            # Greedy action
            action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            state = next_state

            if terminated or truncated:
                break

        rewards.append(total_reward)

    return np.mean(rewards)


# ============================================================
# PARAMETERS
# ============================================================

EPISODES = 5000

ALPHA = 0.1
GAMMA = 0.99

EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

SEED = 42


# ============================================================
# 1. DETERMINISTIC ENVIRONMENT
# ============================================================

print("=" * 60)
print("DETERMINISTIC ENVIRONMENT")
print("=" * 60)

env = GridworldEnv(
    render_mode=None,
    is_slippery=False
)

print("\nTraining SARSA...")

Q_sarsa_det, rewards_sarsa_det = train_sarsa(
    env,
    episodes=EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    epsilon_min=EPSILON_MIN,
    seed=SEED
)

print("SARSA finished.")

print("\nTraining Q-learning...")

Q_qlearning_det, rewards_qlearning_det = train_q_learning(
    env,
    episodes=EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    epsilon_min=EPSILON_MIN,
    seed=SEED
)

print("Q-learning finished.")


# Print learned policies

print("\nSARSA policy:")
print_policy(Q_sarsa_det, env)

print("Q-learning policy:")
print_policy(Q_qlearning_det, env)


# ============================================================
# 2. STOCHASTIC ENVIRONMENT
# ============================================================

print("=" * 60)
print("STOCHASTIC ENVIRONMENT")
print("=" * 60)

env_stochastic = GridworldEnv(
    render_mode=None,
    is_slippery=True
)

print("\nTraining SARSA...")

Q_sarsa_stoch, rewards_sarsa_stoch = train_sarsa(
    env_stochastic,
    episodes=EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    epsilon_min=EPSILON_MIN,
    seed=SEED
)

print("SARSA finished.")

print("\nTraining Q-learning...")

Q_qlearning_stoch, rewards_qlearning_stoch = train_q_learning(
    env_stochastic,
    episodes=EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    epsilon_min=EPSILON_MIN,
    seed=SEED
)

print("Q-learning finished.")


# Print learned policies

print("\nSARSA policy:")
print_policy(Q_sarsa_stoch, env_stochastic)

print("Q-learning policy:")
print_policy(Q_qlearning_stoch, env_stochastic)


# ============================================================
# 3. EVALUATION
# ============================================================

print("=" * 60)
print("EVALUATION")
print("=" * 60)

score_sarsa_det = evaluate_policy(
    Q_sarsa_det,
    GridworldEnv(is_slippery=False)
)

score_qlearning_det = evaluate_policy(
    Q_qlearning_det,
    GridworldEnv(is_slippery=False)
)

score_sarsa_stoch = evaluate_policy(
    Q_sarsa_stoch,
    GridworldEnv(is_slippery=True)
)

score_qlearning_stoch = evaluate_policy(
    Q_qlearning_stoch,
    GridworldEnv(is_slippery=True)
)

print(
    f"SARSA deterministic:   {score_sarsa_det:.3f}"
)

print(
    f"Q-learning deterministic: {score_qlearning_det:.3f}"
)

print(
    f"SARSA stochastic:      {score_sarsa_stoch:.3f}"
)

print(
    f"Q-learning stochastic: {score_qlearning_stoch:.3f}"
)


# ============================================================
# 4. PLOT DETERMINISTIC
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    moving_average(rewards_sarsa_det),
    label="SARSA"
)

plt.plot(
    moving_average(rewards_qlearning_det),
    label="Q-learning"
)

plt.xlabel("Episode")
plt.ylabel("Average reward")
plt.title("SARSA vs Q-learning - Deterministic GridWorld")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()


# ============================================================
# 5. PLOT STOCHASTIC
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    moving_average(rewards_sarsa_stoch),
    label="SARSA"
)

plt.plot(
    moving_average(rewards_qlearning_stoch),
    label="Q-learning"
)

plt.xlabel("Episode")
plt.ylabel("Average reward")
plt.title("SARSA vs Q-learning - Stochastic GridWorld")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()
