import numpy as np
from ch2_rl_formulation.gridworld import GridWorld4x4
from ch2_rl_formulation.policies import greedy_toward_goal_policy
from ch2_rl_formulation.evaluation import policy_evaluation, q_from_v

def test_numeric():
    assert abs(1+0.9*2+0.9**2*3-5.23)<1e-6
    assert abs((-1+0.9*(-1)+0.9**2*(-1)+0.9**3*10)-4.58)<1e-6
    assert abs(2+0.9*4-5.6)<1e-12
    assert abs(1+0.9*3-3.7)<1e-12
    assert abs(max(2+0.9*5,1+0.9*8)-8.2)<1e-12
    assert abs(2+0.9*6-7.4)<1e-12
    v4=sum((0.9**k)*(-1) for k in range(4))+0.9**4*10
    assert abs(v4-3.122)<1e-3

def test_grid():
    env=GridWorld4x4(step_reward=-1, goal_reward=0, goal=(0,3))
    S,A=env.states(),env.actions()
    P,R=env.P_tensor(),env.R_tensor()
    pi=greedy_toward_goal_policy(env)
    V=policy_evaluation(S,A,P,R,pi,gamma=1.0,theta=1e-12)
    Vgrid=np.array(V).reshape(4,4)
    expected=np.array([[-4,-3,-2,0],[-5,-4,-3,-1],[-6,-5,-4,-2],[-7,-6,-5,-3]],dtype=float)
    assert np.allclose(Vgrid,expected,atol=1e-12)
    Q=q_from_v(S,A,P,R,V,gamma=1.0)
    s_bl=env.state_index(3,0)
    a_up=env.action_index("up"); a_right=env.action_index("right")
    a_left=env.action_index("left"); a_down=env.action_index("down")
    assert abs(Q[s_bl,a_up]-(-7))<1e-12
    assert abs(Q[s_bl,a_right]-(-7))<1e-12
    assert abs(Q[s_bl,a_left]-(-8))<1e-12
    assert abs(Q[s_bl,a_down]-(-8))<1e-12
