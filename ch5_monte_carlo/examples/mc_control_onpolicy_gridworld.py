# --- add/replace this helper at the top of the file ---
def step(env, s, a):
    """Sample one step given state and action; robust to P formats."""
    a = int(a)
    s_idx = env.s2i[s] if isinstance(s, tuple) else int(s)

    P = env.P
    if s_idx in P:
        trans = P[s_idx][a]
    else:
        trans = P[env.i2s[s_idx]][a]

    tr = trans[0]
    sp = tr.sp
    if isinstance(sp, int):
        sp = env.i2s[sp]
    return sp, tr.r
