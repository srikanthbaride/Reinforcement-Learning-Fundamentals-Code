from utils.gridworld import GridWorld, Pos

if __name__ == "__main__":
    env = GridWorld()
    s = Pos(3, 0)
    print("Start:", s, "Goal:", env.goal)
    for a in env.A:
        ns, r = env.step(s, a)
        print(f"a={a} -> {ns}, r={r}")
