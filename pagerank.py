import numpy as np


def main():
    N = 3
    # for the simplicity of this project, we would create a transition matrix for each sites
    A = np.array([
        [0, 1, 0],
        [0.5, 0, 0.5],
        [1, 0, 0]
    ])
    # construct the transition matrix
    column_sums = A.sum(axis=0, keepdims=True)
    column_sums[column_sums == 0] = 1
    transition_matrix = A / column_sums

    # initial state
    R = np.ones(N) / N
    for _ in range(100):
        R_next = transition_matrix @ R
        if np.allclose(R, R_next, atol=1e-6):
            break
        R = R_next
    # Normalize R to represent probabilities
    R = R / R.sum()
    with open('PageRank.txt', 'a') as files:
        files.write(f'739261,{R[0]}\n')
        files.write(f'4436465,{R[1]}\n')
        files.write(f'9229752,{R[2]}\n')
    print("PageRank:", R)

if __name__ == "__main__":
    main()