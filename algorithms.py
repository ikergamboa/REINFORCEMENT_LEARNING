import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Epsilon-greedy action selection.

    With probability epsilon:
        choose a random action.

    With probability 1-epsilon:
        choose the action with the highest Q-value.
    """

    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])

    return np.argmax(Q[state])


def train_sarsa(
    env,
    episodes=5000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    max_steps=100,
    seed=None
):
    """
    Train an agent using the SARSA algorithm.

    Returns:
        Q:
            Learned Q-table.

        rewards:
            Total reward obtained in each episode.
    """

    if seed is not None:
        np.random.seed(seed)

    # Q[state, action]
    Q = np.zeros((env.n_states, env.n_actions))

    rewards = []

    for episode in range(episodes):

        # Reset environment
        state, info = env.reset()

        # SARSA chooses the first action before entering the loop
        action = epsilon_greedy(Q, state, epsilon)

        total_reward = 0

        for step in range(max_steps):

            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            total_reward += reward

            # If terminal, there is no next action.
            if done:

                target = reward

                Q[state, action] += alpha * (
                    target - Q[state, action]
                )

                break

            # Choose next action
            next_action = epsilon_greedy(
                Q,
                next_state,
                epsilon
            )

            # SARSA update
            target = (
                reward
                + gamma * Q[next_state, next_action]
            )

            Q[state, action] += alpha * (
                target - Q[state, action]
            )

            # Move to next state/action
            state = next_state
            action = next_action

        rewards.append(total_reward)

        # Reduce exploration
        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

    return Q, rewards


def train_q_learning(
    env,
    episodes=5000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    max_steps=100,
    seed=None
):
    """
    Train an agent using the Q-learning algorithm.

    Returns:
        Q:
            Learned Q-table.

        rewards:
            Total reward obtained in each episode.
    """

    if seed is not None:
        np.random.seed(seed)

    # Q[state, action]
    Q = np.zeros((env.n_states, env.n_actions))

    rewards = []

    for episode in range(episodes):

        state, info = env.reset()

        total_reward = 0

        for step in range(max_steps):

            # Choose action using epsilon-greedy
            action = epsilon_greedy(
                Q,
                state,
                epsilon
            )

            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            total_reward += reward

            # Q-learning target
            if done:

                target = reward

            else:

                target = (
                    reward
                    + gamma * np.max(Q[next_state])
                )

            # Q-learning update
            Q[state, action] += alpha * (
                target - Q[state, action]
            )

            state = next_state

            if done:
                break

        rewards.append(total_reward)

        # Reduce exploration
        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

    return Q, rewards
