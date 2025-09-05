from ch6_td_learning.gridworld import Chain3Env
from ch6_td_learning.nstep_td import nstep_td_prediction

def main():
    env = Chain3Env()
    policy = lambda s: 0   # dummy policy
    V = nstep_td_prediction(env, policy, n=2, gamma=0.9, alpha=0.5, episodes=10)
    print("2-step TD estimates after 10 episodes (gamma=0.9, alpha=0.5):")
    for s in ("A","B","C","T"):
        print(f"  V({s}) = {V.get(s, 0.0):.6f}")

if __name__ == "__main__":
    main()
