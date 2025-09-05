from ch6_td_learning.gridworld import Chain3Env
from ch6_td_learning.td0 import td0_prediction

def main():
    env = Chain3Env()
    policy = lambda s: 0   # dummy policy, env ignores action
    V = td0_prediction(env, policy, gamma=0.9, alpha=0.5, episodes=10)
    print("TD(0) estimates after 10 episodes (gamma=0.9, alpha=0.5):")
    for s in ("A","B","C","T"):
        print(f"  V({s}) = {V.get(s, 0.0):.6f}")

if __name__ == "__main__":
    main()
