import numpy as np

def compute_covariance(N, centered_indexing=True):
    matrix = np.zeros(shape=(N**2, N**2))
    for n_l in range(N):
        for m_l in range(N):
            for n_r in range(N):
                for m_r in range(N):
                    if centered_indexing:
                        if (n_l + m_l) % N == 0 or (n_l == 0 or m_l == 0):
                            if n_l == 0 and m_l == 0:
                                idx_l = 0
                            elif n_l == 0:
                                idx_l = 3 * (m_l - 1) + 1
                            elif m_l == 0:
                                idx_l = 3 * (N - n_l - 1) + 2
                            else:
                                idx_l = 3 * (n_l - 1) + 3
                        elif n_l + m_l > N:
                            idx_l = 3 * N - 2 + (n_l - 1) * (N - 2) + m_l - 2
                        else: # RED UT NÄR N%3 = 0
                            idx_l = 3 * N - 2 + (n_l - 1) * (N - 2) + m_l - 1
                        if (n_r + m_r) % N == 0 or (n_r == 0 or m_r == 0):
                            if n_r == 0 and m_r == 0:
                                idx_r = 0
                            elif n_r == 0:
                                idx_r = 3 * (m_r - 1) + 1
                            elif m_r == 0:
                                idx_r = 3 * (N - n_r - 1) + 2
                            else:
                                idx_r = 3 * (n_r - 1) + 3
                        elif n_r + m_r > N:
                            idx_r = 3 * N - 2 + (n_r - 1) * (N - 2) + m_r - 2
                        else:
                            idx_r = 3 * N - 2 + (n_r - 1) * (N - 2) + m_r - 1
                    else:
                        idx_l = n_l*N+m_l
                        idx_r = n_r*N+m_r
            
                    matrix[idx_l, idx_r] = compute_element(n_l, m_l, n_r, m_r, N)
    return matrix

def compute_element(n_l, m_l, n_r, m_r, N):
    res = 0

    if n_l == n_r and m_l == m_r:
        res += 1
    if n_l == (n_r + m_r) % N and n_r == (n_l + m_l) % N:
        res += 1
    if m_l == (n_r + m_r) % N and m_r == (n_l + m_l) % N:
        res += 1
    
    if (n_l + m_r) % N == 0 and (n_r + m_l) % N == 0:
        res += 1
    if (n_l + n_r + m_r) % N == 0 and (n_l + m_l + m_r) % N == 0:
        res += 1
    if (m_l + n_r + m_r) % N == 0 and (n_l + m_l + n_r) % N == 0:
        res += 1
    
    tmp_l = 0
    if (n_l + m_l) % N == 0:
        tmp_l += 1
    if n_l == 0:
        tmp_l += 1
    if m_l == 0:
        tmp_l += 1
    
    tmp_r = 0
    if (n_r + m_r) % N == 0:
        tmp_r += 1
    if n_r == 0:
        tmp_r += 1
    if m_r == 0:
        tmp_r += 1
    
    res += tmp_l * tmp_r

    return res

if __name__ == "__main__":
    N = 3
    O_dim = (N - 1) * (N - 2)
    covariance = compute_covariance(N, centered_indexing=True)
    assert(np.allclose(covariance[:-O_dim, -O_dim:], 0))

    print("matrix")
    print(covariance)
    print("diagonal")
    print(np.diag(covariance))
    print("rank")
    print(np.linalg.matrix_rank(covariance))

    # print(covariance[:-O_dim, :-O_dim])
    
    # print(covariance[-O_dim:, -O_dim:])

    # tmp_rank = 694
    # for N in range(64, 100):
    #     O_dim = (N - 1) * (N - 2)
    #     covariance = compute_covariance(N, centered_indexing=True)
    #     assert(np.allclose(covariance[:-O_dim, -O_dim:], 0))

    #     rank = int(np.linalg.matrix_rank(covariance))
    #     print(N, N**2, f"({2*N+1})", rank, f"({rank-tmp_rank})")
    #     tmp_rank = rank

        # k = N ** 2 - O_dim
        # tmp = []
        # while k < N ** 2:
        #     if np.diag(covariance)[k] != 1:
        #         tmp.append((k, int(np.diag(covariance)[k])))
        #     k += 1
        # print(N, tmp)
