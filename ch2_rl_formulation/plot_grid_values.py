import numpy as np
import matplotlib.pyplot as plt
from utils.gridworld import GridWorld
from value_iteration import value_iteration

if __name__ == "__main__":
    env = GridWorld()
    V, pi = value_iteration(env)

    fig = plt.figure()
    plt.title("GridWorld Optimal Values (Chapter 2)")
    im = plt.imshow(V, interpolation='nearest')
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()
