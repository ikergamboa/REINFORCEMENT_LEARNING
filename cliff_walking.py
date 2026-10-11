import numpy as np


class CliffWalkingEnv:
    """Cliff Walking (Sutton & Barto, ej. 6.6), misma interfaz que GridworldEnv.

    Grid 4x12, estado s = fila * 12 + columna, acciones 0=LEFT 1=DOWN 2=RIGHT 3=UP.

        fila 0  .  .  .  .  .  .  .  .  .  .  .  .
        fila 1  .  .  .  .  .  .  .  .  .  .  .  .
        fila 2  .  .  .  .  .  .  .  .  .  .  .  .
        fila 3  S  C  C  C  C  C  C  C  C  C  C  G

    - Cada paso cuesta -1. Caer al acantilado (C) da -100 y te devuelve a S
      (el episodio NO termina). Llegar a G termina el episodio.
    - slip_prob > 0 añade ruido: con esa prob. total el agente se desvía
      90 grados (mitad a cada lado), igual que en GridworldEnv.
    - Incluye env.P con el mismo formato [(prob, s', reward, done), ...].
    """

    def __init__(self, slip_prob=0.0, step_cost=-1.0, cliff_penalty=-100.0):
        self.n_rows, self.n_cols = 4, 12
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4
        self.start_state = 36
        self.goal_state = 47
        self.cliff_states = set(range(37, 47))
        self.slip_prob = slip_prob
        self.step_cost = step_cost
        self.cliff_penalty = cliff_penalty
        self.P = self._build_P()
        # probabilidades acumuladas precalculadas -> step() rápido
        self._cum = {
            (s, a): np.cumsum([t[0] for t in self.P[s][a]])
            for s in range(self.n_states) for a in range(self.n_actions)
            if self.P[s][a]
        }
        self.s = self.start_state

    def _move(self, s, a):
        deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        r, c = divmod(s, self.n_cols)
        dr, dc = deltas[a]
        nr = min(max(r + dr, 0), self.n_rows - 1)
        nc = min(max(c + dc, 0), self.n_cols - 1)
        s2 = nr * self.n_cols + nc
        if s2 in self.cliff_states:
            return self.start_state, self.cliff_penalty, False
        return s2, self.step_cost, s2 == self.goal_state

    def _build_P(self):
        slips = {0: (1, 3), 1: (0, 2), 2: (1, 3), 3: (0, 2)}
        p = self.slip_prob
        P = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        for s in range(self.n_states):
            if s in self.cliff_states:
                continue  # nunca se ocupan
            for a in range(self.n_actions):
                if s == self.goal_state:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                if p > 0:
                    outcomes = [(a, 1 - p), (slips[a][0], p / 2), (slips[a][1], p / 2)]
                else:
                    outcomes = [(a, 1.0)]
                merged = {}
                for act, prob in outcomes:
                    key = self._move(s, act)
                    merged[key] = merged.get(key, 0.0) + prob
                P[s][a] = [(prob, s2, r, d) for (s2, r, d), prob in merged.items()]
        return P

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.s = self.start_state
        return self.s, {}

    def step(self, action):
        trans = self.P[self.s][action]
        if len(trans) == 1:
            idx = 0
        else:
            idx = int(np.searchsorted(self._cum[(self.s, action)], np.random.random(), side="right"))
            idx = min(idx, len(trans) - 1)
        _, next_state, reward, terminated = trans[idx]
        self.s = next_state
        return next_state, reward, terminated, False, {}

    def render(self):
        out = ""
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                s = r * self.n_cols + c
                if s == self.s:
                    out += " A "
                elif s in self.cliff_states:
                    out += " C "
                elif s == self.goal_state:
                    out += " G "
                else:
                    out += " . "
            out += "\n"
        return out
