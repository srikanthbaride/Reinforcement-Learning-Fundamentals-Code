from typing import Callable, Hashable, Dict, List, Tuple

def td0_prediction(env,
                   policy: Callable[[Hashable], int],
                   gamma: float = 0.99,
                   alpha: float = 0.1,
                   episodes: int = 500,
                   sweep_per_episode: bool = True) -> Dict[Hashable, float]:
    """
    TD(0) state-value prediction for a fixed policy π.

    If sweep_per_episode=True (default), we store one full episode and then
    perform TD(0) updates in reverse order (T-1 down to 0). This matches the
    textbook’s single-episode hand calculation and the unit tests.

    If False, updates are done strictly online/forward during the episode.
    """
    V: Dict[Hashable, float] = {}

    for _ in range(episodes):
        s = env.reset()
        V.setdefault(s, 0.0)

        if not sweep_per_episode:
            done = False
            while not done:
                a = policy(s)
                s_next, r, done, _ = env.step(a)
                V.setdefault(s_next, 0.0)
                target = r if done else r + gamma * V[s_next]
                V[s] += alpha * (target - V[s])
                s = s_next
            continue

        # --- backward episode sweep mode ---
        traj: List[Tuple[Hashable, float, Hashable, bool]] = []  # (s, r, s_next, done)
        done = False
        while not done:
            a = policy(s)
            s_next, r, done, _ = env.step(a)
            V.setdefault(s_next, 0.0)
            traj.append((s, r, s_next, done))
            s = s_next

        # process in reverse so later-state updates are available to earlier ones
        for (s, r, s_next, done) in reversed(traj):
            target = r if done else r + gamma * V[s_next]
            V[s] += alpha * (target - V[s])

    return V
