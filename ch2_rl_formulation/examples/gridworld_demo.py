import numpy as np
from ch2_rl_formulation.gridworld import GridWorld4x4
from ch2_rl_formulation.policies import greedy_toward_goal_policy
from ch2_rl_formulation.evaluation import policy_evaluation,q_from_v

def main():
    env=GridWorld4x4(); S,A=env.states(),env.actions()
    P,R=env.P_tensor(),env.R_tensor()
    pi=greedy_toward_goal_policy(env)
    V=policy_evaluation(S,A,P,R,pi)
    print("V_pi grid:",np.array(V).reshape(4,4))
    Q=q_from_v(S,A,P,R,V)
    s_bl=env.state_index(3,0)
    print("Q bottom-left:",[Q[s_bl,env.action_index(a)] for a in A])
if __name__=='__main__': main()
