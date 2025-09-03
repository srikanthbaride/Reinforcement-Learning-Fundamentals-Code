from ch2_rl_formulation.examples.numeric_checks import (
    discounted_return_example, state_value_example, gridworld_vq_under_fixed_policy
)

def test_discounted_return_boxed():
    assert round(discounted_return_example(), 2) == 5.23

def test_state_value_boxed():
    assert round(state_value_example(), 2) == 4.58

def test_gridworld_values_match_distance_gamma1():
    V, _, _ = gridworld_vq_under_fixed_policy(gamma=1.0)
    expected = {(3,0):-6, (2,2):-3, (0,2):-1, (0,3):0}
    for (i,j), val in expected.items():
        assert int(round(V[i, j])) == val

