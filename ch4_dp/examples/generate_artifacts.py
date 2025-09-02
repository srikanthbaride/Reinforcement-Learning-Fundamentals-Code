from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rldp.gridworld import make_gridworld, unravel_index, ACTIONS, arrows_from_policy
from rldp.dp import policy_evaluation, policy_iteration, value_iteration

def save_grid_csv(V, n, out_csv):
    M = np.zeros((n, n))
    for s in range(n*n):
        i, j = unravel_index(s, n)
        M[i, j] = V[s]
    df = pd.DataFrame(M)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

def save_policy_csv(pi, n, out_csv):
    arr = arrows_from_policy(pi).reshape(n, n)
    df = pd.DataFrame(arr)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

def plot_values(V, n, out_png, title=None):
    M = np.zeros((n, n))
    for s in range(n*n):
        i, j = unravel_index(s, n)
        M[i, j] = V[s]
    fig = plt.figure()
    plt.imshow(M, interpolation='nearest')
    plt.colorbar()
    if title:
        plt.title(title)
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{M[i,j]:.0f}", ha='center', va='center')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches='tight', dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='4x4', choices=['4x4','6x6'])
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--theta', type=float, default=1e-6)
    ap.add_argument('--outdir', default='artifacts/ch4_4x4')
    args = ap.parse_args()

    n = 4 if args.env == '4x4' else 6
    states, actions, P, R, meta = make_gridworld(n=n)
    # Policy Iteration
    pi_pi, V_pi = policy_iteration(states, actions, P, R, gamma=args.gamma, theta=args.theta)
    # Value Iteration
    pi_vi, V_vi = value_iteration(states, actions, P, R, gamma=args.gamma, theta=args.theta)

    os.makedirs(args.outdir, exist_ok=True)

    # Save values (final)
    save_grid_csv(V_pi, n, os.path.join(args.outdir, f'pi_values_{args.env}.csv'))
    save_grid_csv(V_vi, n, os.path.join(args.outdir, f'vi_values_{args.env}.csv'))
    plot_values(V_pi, n, os.path.join(args.outdir, f'pi_values_{args.env}.png'), 'Policy Iteration Values')
    plot_values(V_vi, n, os.path.join(args.outdir, f'vi_values_{args.env}.png'), 'Value Iteration Values')

    # Save policies
    save_policy_csv(pi_pi, n, os.path.join(args.outdir, f'pi_policy_{args.env}.csv'))
    save_policy_csv(pi_vi, n, os.path.join(args.outdir, f'vi_policy_{args.env}.csv'))

    print('Artifacts written to:', args.outdir)

if __name__ == '__main__':
    main()
