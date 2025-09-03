from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_value_grid(V: np.ndarray, title: str = "State-Value Grid", fname: str | None = None):
    Vg = np.array(V).reshape(4, 4)
    fig, ax = plt.subplots()
    im = ax.imshow(Vg)
    ax.set_title(title)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{Vg[i,j]:.1f}", ha="center", va="center")
    plt.colorbar(im, ax=ax)
    if fname: plt.savefig(fname, bbox_inches="tight")
    else: plt.show()

def plot_greedy_policy(pi: np.ndarray, title: str = "Greedy Policy (0:R,1:L,2:D,3:U)", fname: str | None = None):
    P = np.array(pi).reshape(4,4)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5)
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.grid(True)
    dirs = {0:(1,0),1:(-1,0),2:(0,1),3:(0,-1)}
    for i in range(4):
        for j in range(4):
            dx, dy = dirs.get(int(P[i,j]), (0,0))
            ax.arrow(j, i, dx*0.3, dy*0.3, head_width=0.1, length_includes_head=True)
    if fname: plt.savefig(fname, bbox_inches="tight")
    else: plt.show()

