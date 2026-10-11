"""Utilidades compartidas: evaluación exacta de políticas y value iteration.

Ambas funciones usan el modelo env.P, así que NO simulan episodios: son
exactas (sin ruido) y muy baratas en cómputo (entornos de 12-48 estados).
"""
import numpy as np


def moving_average(values, window=100):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def policy_value(env, Q, gamma=0.99):
    """Valor exacto V^pi de la política greedy respecto a Q.

    Resuelve el sistema lineal (I - gamma * P_pi) V = r_pi.
    Los estados sin transiciones (muro, acantilado) o terminales quedan en V = 0.
    """
    n = env.n_states
    A = np.eye(n)
    b = np.zeros(n)
    for s in range(n):
        a = int(np.argmax(Q[s]))
        for prob, s2, reward, done in env.P[s][a]:
            b[s] += prob * reward
            if not done:
                A[s, s2] -= gamma * prob
    return np.linalg.solve(A, b)


def value_iteration(env, gamma=0.99, tol=1e-10):
    """Devuelve (V*, Q*) del MDP usando env.P (referencia 'óptima')."""
    n, na = env.n_states, env.n_actions
    V = np.zeros(n)
    while True:
        Q = np.zeros((n, na))
        for s in range(n):
            for a in range(na):
                for prob, s2, reward, done in env.P[s][a]:
                    Q[s, a] += prob * (reward + (0.0 if done else gamma * V[s2]))
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q
        V = V_new
