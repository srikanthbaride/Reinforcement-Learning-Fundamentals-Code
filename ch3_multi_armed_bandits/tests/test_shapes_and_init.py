from ch3_multi_armed_bandits.epsilon_greedy import run as run_eps
from ch3_multi_armed_bandits.ucb import run as run_ucb
from ch3_multi_armed_bandits.thompson import run as run_ts

def test_shapes_and_init():
    means=[0.7,0.5,0.3];T=50
    out=run_eps(means,0.1,T,0)
    assert out["rewards"].shape==(T,)
    out=run_ucb(means,1.0,T,0)
    assert (out["N"]>=1).all()
    out=run_ts(means,T,0)
    assert len(out["alpha"])==3
