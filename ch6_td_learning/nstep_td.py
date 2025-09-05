import math
from typing import Callable, Hashable, Dict, List, Tuple

def nstep_td_prediction(env,
                        policy: Callable[[Hashable], int],
                        n: int = 3,
                        gamma: float = 0.99,
                        alpha: float = 0.1,
                        episodes: int = 500) -> Dict[Hashable, float]:
    if n < 1:
        raise ValueError("n must be >= 1")

    V: Dict[Hashable, float] = {}

    for _ in range(episodes):
        s0 = env.reset()
        V.setdefault(s0, 0.0)

        states: List[Hashable] = [s0]
        rewards: List[float] = [0.0]  # dummy R_0
        T = math.inf
        t = 0

        while True:
            if t < T:
                a = policy(states[-1])
                s_next, r, done, _ = env.step(a)
                states.append(s_next); rewards.append(float(r))
                V.setdefault(s_next, 0.0)
                if done:
                    T = t + 1  # first terminal index in 1-based reward indexing

            tau = t - n + 1
            if tau >= 0:
                # compute G_tau^(n)
                G = 0.0
                # rewards indices: tau+1 .. min(tau+n, T)
                last = min(tau + n, int(T)) if T != math.inf else tau + n
                for k in range(tau + 1, last + 1):
                    G += (gamma ** (k - tau - 1)) * rewards[k]
                # bootstrap if we haven’t hit terminal within n steps
                if tau + n < T:
                    G += (gamma ** n) * V[states[tau + n]]
                V[states[tau]] += alpha * (G - V[states[tau]])

            t += 1
            if tau >= (T - 1):
                break

    return V
