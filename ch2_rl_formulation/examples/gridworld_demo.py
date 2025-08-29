import numpy as np
from ch2_rl_formulation.gridworld import GridWorld4x4
from ch2_rl_formulation.policies import greedy_toward_goal_policy
from ch2_rl_formulation.evaluation import policy_evaluation, q_from_v

def main():
    env = GridWorld4x4(step_reward=-1, goal_reward=0, goal=(0,3))
    S, A = env.states(), env.actions()
    P, R = env.P_tensor(), env.R_tensor()
    pi = greedy_toward_goal_policy(env)
    V = policy_evaluation(S, A, P, R, pi, gamma=1.0, theta=1e-12)
    Vgrid = np.array(V).reshape(4,4)
    print("V_pi grid (goal top-right):\n", Vgrid)
    expected = np.array([
        [-4, -3, -2,  0],
        [-5, -4, -3, -1],
        [-6, -5, -4, -2],
        [-7, -6, -5, -3],
    ], dtype=float)
    assert np.allclose(Vgrid, expected, atol=1e-12)
    Q = q_from_v(S, A, P, R, V, gamma=1.0)
    s_bl = env.state_index(3,0)
    a = {name: env.action_index(name) for name in A}
    print("\nQ at bottom-left (row=3,col=0):")
    for name in A:
        print(f"{name:>5}: {Q[s_bl, a[name]]:6.2f}")
    assert abs(Q[s_bl, a["up"]]   - (-7)) < 1e-12
    assert abs(Q[s_bl, a["right"]]- (-7)) < 1e-12
    assert abs(Q[s_bl, a["left"]] - (-8)) < 1e-12
    assert abs(Q[s_bl, a["down"]] - (-8)) < 1e-12
    print("\nAll checks PASS.")

if __name__ == "__main__":
    main()
