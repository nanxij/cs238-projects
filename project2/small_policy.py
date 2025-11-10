import numpy as np
import pandas as pd
import os
from collections import defaultdict

# constants ----------------
GAMMA = 0.95
NUM_STATES = 100
NUM_ACTIONS = 4
MAX_ITERS = 10000
TOL = 1e-8

# helper--------------
def convert_indices(x, y):
    return int(x+(y-1)*10)

# load data -----------
def load_samples(filename):
    df = pd.read_csv(filename)
    cols = list(df.columns)
    samples = df[['s','a','r','sp']].copy()
    # Ensure types are ints
    samples['s'] = samples['s'].astype(int)
    samples['sp'] = samples['sp'].astype(int)
    samples['a'] = samples['a'].astype(int)
    return samples

# estimate model ----------------------------
def estimate_model(samples, num_states, num_actions):
    # counts and total rewards
    transition_counts = defaultdict(lambda: defaultdict(int))
    reward_sums = defaultdict(float) 
    count_sa = defaultdict(int)

    for _, row in samples.iterrows():
        s = int(row['s'])
        a = int(row['a'])
        r = float(row['r'])
        sp = int(row['sp'])
        transition_counts[(s,a)][sp] += 1
        reward_sums[(s,a)] += r
        count_sa[(s,a)] += 1

    # build P and R
    P = np.zeros((num_states+1, num_actions+1,num_states+1))
    R = np.zeros((num_states+1, num_actions+1))
    seen_sa = set(count_sa.keys())

    for s in range(1, num_states+1):
        for a in range(1, num_actions+1):
            i = (s,a)
            if i in count_sa:
                total = count_sa[i]
                R[s,a] = reward_sums[i] / total
                for sp, c in transition_counts[i].items():
                    if 1 <= sp <= num_states:
                        P[s,a,sp] = c / total
                    else:
                        pass
            else:
                # fallback for unseen (s,a): self-loop with zero reward
                P[s,a,s] = 1.0
                R[s,a] = 0.0

    return P, R

# value iteration given P, R ---------------------
def value_iteration(P, R, gamma=GAMMA, tol=TOL, max_iters=MAX_ITERS):
    S = P.shape[0]-1
    A = P.shape[1]-1
    V = np.zeros(S+1)  # 1-based
    for it in range(max_iters):
        V_prev = V.copy()
        # Bellman optimality update
        Q = np.zeros((S+1, A+1))
        for a in range(1, A+1):
            Q[:, a] = R[:, a] + gamma * (P[:, a, :] @ V)
        V = np.max(Q[:, 1:], axis=1)
        V[0] = 0.0
        diff = np.max(np.abs(V - V_prev))
        if diff < tol:
            # print(it)
            break
    # deterministic greedy policy
    policy = np.zeros(S+1, dtype=int)
    for s in range(1, S+1):
        qsa = np.zeros(A+1)
        for a in range(1, A+1):
            qsa[a] = R[s,a] + gamma * (P[s,a,:] @ V)
        # ties
        best_a = int(np.argmax(qsa[1:])+1)
        policy[s] = best_a
    return V, policy

# utility to convert back to (x,y) for saving -------
def state_to_xy(state):
    y = (state-1) // 10 + 1
    x = state-(y-1) * 10
    return x, y

# main --------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='small.csv', help='input CSV file with samples')
    parser.add_argument('--gamma', type=float, default=0.95)
    args = parser.parse_args()

    samples =load_samples(args.csv)
    NUM_STATES = 100
    NUM_ACTIONS = 4

    P, R = estimate_model(samples, NUM_STATES, NUM_ACTIONS)
    V, policy = value_iteration(P, R, gamma=args.gamma, tol=TOL, max_iters=MAX_ITERS)

    # output namw
    out_name = os.path.splitext(args.csv)[0] + ".policy"
    with open(out_name, "w") as f:
        for s in range(1,NUM_STATES+1):
            f.write(f"{policy[s]}\n")
