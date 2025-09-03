from ch3_multi_armed_bandits.epsilon_greedy import run as run_eps
def test_determinism():
    means=[0.6,0.4];T=30
    a=run_eps(means,0.1,T,123);b=run_eps(means,0.1,T,123)
    assert (a["choices"]==b["choices"]).all()
