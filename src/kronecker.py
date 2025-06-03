import numpy as np

def covar_matrix(N, simplified=True):
    matrix = np.zeros(shape=(N**2, N**2))
    for n_l in range(N):
        for m_l in range(N):
            for n_r in range(N):
                for m_r in range(N):
                    if simplified:
                        matrix[n_l*N+m_l, n_r*N+m_r] = simplified_formula(n_l, m_l, n_r, m_r, N)
                    else: 
                        matrix[n_l*N+m_l, n_r*N+m_r] = complicated_formula(n_l, m_l, n_r, m_r, N)
    return matrix

def simplified_formula(n_l, m_l, n_r, m_r, N):
    res = 0

    if n_l == n_r and m_l == m_r:
        res += 1
    if (n_l + m_l) % N == 0 and (n_r + m_r) % N == 0:
        res += 1
    if (n_l + m_r) % N == 0 and (n_r + m_l) % N == 0:
        res += 1

    if (n_l + m_l) % N == 0:
        if n_r == 0:
            res += 1
        if m_r == 0:
            res += 1
    if (n_r + m_r) % N == 0:
        if n_l == 0:
            res += 1
        if m_l == 0:
            res += 1
    
    if n_l == 0 and n_r == 0:
        res += 1
    if n_l == 0 and m_r == 0:
        res += 1
    if m_l == 0 and m_r == 0:
        res += 1
    if m_l == 0 and n_r == 0:
        res += 1
    
    if (n_l - n_r - m_r) % N == 0 and (n_l + m_l - n_r) % N == 0:
        res += 1
    if (n_l + n_r + m_r) % N == 0 and (n_l + m_l + m_r) % N == 0:
        res += 1
    if (m_l + n_r + m_r) % N == 0 and (n_l + m_l + n_r) % N == 0:
        res += 1
    if (m_l - n_r - m_r) % N == 0 and (n_l + m_l - m_r) % N == 0:
        res += 1

    return N * res

def complicated_formula(n_l, m_l, n_r, m_r, N):
    res = 0
    
    for i in range(N):
        for j in range(N):
            if i == j:
                if (i - n_l) % N == (j - n_r) % N and (i + m_l) % N == (j + m_r) % N:
                    res += 1
                if (i - n_l) % N == (i + m_l) % N and (j - n_r) % N == (j + m_r) % N:
                    res += 1
                if (i - n_l) % N == (i + m_r) % N and (j - n_r) % N == (i + m_l) % N:
                    res += 1
                
            if (i - n_l) % N == (i + m_l) % N:
                if i == (j + m_r) % N and j == (j - n_r) % N:
                    res += 1
                if i == (j - n_r) % N and j == (j + m_r) % N:
                    res += 1
            
            if (j - n_r) % N == (j + m_r) % N:
                if i == (i + m_l) % N and j == (i - n_l) % N:
                    res += 1
                if i == (i - n_l) % N and j == (i + m_l) % N:
                    res += 1
            
            if (i - n_l) % N == (j - n_r) % N:
                if i == (i + m_l) % N and j == (j + m_r) % N:
                    res += 1
                if i == (j + m_r) % N and j == (i + m_l) % N:
                    res += 1
            
            if (i - n_l) % N == (j + m_r) % N:
                if i == (i + m_l) % N and j == (j - n_r) % N:
                    res += 1
                if i == (j - n_r) % N and j == (i + m_l) % N:
                    res += 1
            
            if (i + m_l) % N == (j - n_r) % N:
                if i == (i - n_l) % N and j == (j + m_r) % N:
                    res += 1
                if i == (j + m_r) % N and j == (i - n_l) % N:
                    res += 1

            if (i + m_l) % N == (j + m_r) % N:
                if i == (i - n_l) % N and j == (j - n_r) % N:
                    res += 1
                if i == (j - n_r) % N and j == (i - n_l) % N:
                    res += 1

    return res

    

if __name__ == "__main__":
    N = 2
    matrix_simplified = covar_matrix(N, simplified=True)
    matrix_complicated = covar_matrix(N, simplified=False)

    print(f"Computing covariance matrix for {N = } with dimensions {N**2}x{N**2}")

    print(f"Simplified and complicated formulas evaluate to same covariance matrix: {np.allclose(matrix_simplified, matrix_complicated)}")

    print(f"Rank of covariance matrix: {np.linalg.matrix_rank(matrix_simplified)}")

    print(matrix_simplified)

    # print(f"List of eigenvalues for the covariance matrix: {[round(k) for k in np.linalg.eigvalsh(matrix_simplified)]}")

    # for M in range(1, 30):
    #     print(M, np.linalg.matrix_rank(covar_matrix(M)))
