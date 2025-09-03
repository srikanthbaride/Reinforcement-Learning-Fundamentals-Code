# --- add/replace this helper at the top of the file ---
def step(env, s, a):
    """
    Sample one step given (state tuple) s and action a, using env.P.
    Works whether env.P is keyed by integer state indices or by state tuples,
    and whether transitions store sp as index or as tuple.
    Returns: (next_state_tuple, reward)
    """
    # Normalize action to plain int (handles np.int64)
    a = int(a)

    # Convert input s (tuple) to index
    s_idx = env.s2i[s] if isinstance(s, tuple) else int(s)

    # Resolve transition list from env.P (supports idx- or tuple-keyed P)
    P = env.P
    if s_idx in P:
        trans = P[s_idx][a]
    else:
        # fall back to tuple-keyed P
        s_tuple = env.i2s[s_idx]
        trans = P[s_tuple][a]

    # Assume deterministic for this GridWorld; if stochastic, sample by probs
    tr = trans[0]
    sp = tr.sp

    # If sp is index, convert to tuple for episode generation
    if isinstance(sp, int):
        sp = env.i2s[sp]

    return sp, tr.r
