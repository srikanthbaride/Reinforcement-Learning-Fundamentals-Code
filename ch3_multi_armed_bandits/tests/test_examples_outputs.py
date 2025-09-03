import math
from ch3_multi_armed_bandits.examples import ex1_regret_basic,ex2_epsilon_update,ex3_ucb_score,ex4_thompson_update

def test_ex1(): out=ex1_regret_basic.compute(); assert math.isclose(out["regret"],20)
def test_ex2(): out=ex2_epsilon_update.compute(); assert math.isclose(out["Q_new"],0.6)
def test_ex3(): out=ex3_ucb_score.compute(); assert out["selected"]==3
def test_ex4(): out=ex4_thompson_update.compute(); assert out["alpha"]==[4,2] and out["beta"]==[3,5]
