from ch6_td_learning import Chain3Env, td0_prediction, nstep_td_prediction


def test_td0_single_episode_matches_hand_calc():
    env = Chain3Env()
    policy = lambda s: 0
    V = td0_prediction(env, policy, gamma=0.9, alpha=0.5, episodes=1)
    # Expected after one episode with +1 on C->T:
    # V(C)=0.5, V(B)=0.225, V(A)=0.10125
    assert abs(V["C"] - 0.5) < 1e-10
    assert abs(V["B"] - 0.225) < 1e-3
    assert abs(V["A"] - 0.10125) < 1e-10


def test_td0_converges_close_to_true_values():
    env = Chain3Env()
    policy = lambda s: 0
    gamma = 0.9
    V = td0_prediction(env, policy, gamma=gamma, alpha=0.1, episodes=3000)
    # True values for this env: v(A)=gamma^2, v(B)=gamma, v(C)=1.0
    assert abs(V["C"] - 1.0) < 0.03
    assert abs(V["B"] - gamma) < 0.04
    assert abs(V["A"] - gamma**2) < 0.05


def test_nstep_td_converges_close_to_true_values():
    env = Chain3Env()
    policy = lambda s: 0
    gamma = 0.9
    V = nstep_td_prediction(env, policy, n=2, gamma=gamma, alpha=0.1, episodes=3000)
    # True values for this env: v(A)=gamma^2, v(B)=gamma, v(C)=1.0
    assert abs(V["C"] - 1.0) < 0.03
    assert abs(V["B"] - gamma) < 0.05
    assert abs(V["A"] - gamma**2) < 0.06
