import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

TAU = 2 * np.pi

def wrap_to_pi(x):
    """Wrap to (-pi, pi]."""
    y = (x + np.pi) % (2*np.pi) - np.pi
    y = np.where(np.isclose(y, -np.pi), np.pi, y)
    return y

# -----------------------
# MRA data generation
# -----------------------
def generate_mra_data(x, sigma, m, seed=0, shifts=None):
    """
    Observations: y_j[n] = x[(n - s_j) mod J] + sigma * noise
    using integer circular shifts s_j.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    J = x.size

    if shifts is None:
        shifts = rng.integers(0, J, size=m, endpoint=False)
    else:
        shifts = np.asarray(shifts, dtype=int)
        assert shifts.shape == (m,)

    Y = np.zeros((m, J), dtype=float)
    for j in range(m):
        # np.roll(x, s) gives y[n] = x[n - s] (mod J)
        Y[j] = np.roll(x, shifts[j]) + sigma * rng.standard_normal(J)

    theta_abs = wrap_to_pi(TAU * shifts / J)          # "absolute" angles (wrt underlying x)
    theta_rel = wrap_to_pi(theta_abs - theta_abs[0])  # gauge: theta_1 = 0 (doc convention)
    theta_rel[0] = 0.0

    return x, Y, shifts, theta_abs, theta_rel

# -----------------------
# Build Eq(3) alpha values
# -----------------------
def alphas_from_dfts_ratio_phase(Xhat, k_list=None, eps=1e-12):
    """
    For each k and pair (j1,j2), define alpha[j1,j2,k] = arg( Xhat[j2,k] / Xhat[j1,k] ).
    Compute phase stably as arg( Xhat[j2,k] * conj(Xhat[j1,k]) ).
    Then in the noiseless model Xhat[j,k] = e^{-ik theta_j} Xhat_true[k], we get:
        alpha[j1,j2,k] = k (theta_{j1} - theta_{j2})  (mod 2pi),
    matching Eq. (3).  (Indices j1,j2 correspond to rows in Eq. (3).)
    """
    m, J = Xhat.shape
    if k_list is None:
        k_list = list(range(1, J))  # k = 1..J-1
    K = len(k_list)

    alpha = np.zeros((K, m, m), dtype=np.float64)
    weight = np.zeros((K, m, m), dtype=np.float64)

    for kk, k in enumerate(k_list):
        Xk = Xhat[:, k]
        # C[j1,j2] = Xhat[j2] * conj(Xhat[j1])  => phase = arg(Xhat[j2]/Xhat[j1])
        C = Xk[None, :] * np.conj(Xk[:, None])
        mag = np.abs(C)
        valid = mag > eps
        alpha[kk][valid] = np.angle(C[valid])
        weight[kk][valid] = mag[valid]

    return alpha, weight, k_list

def build_star_measurements(alpha, weight, k_list):
    """
    Use only pairs (0, j) for j=1..m-1 (star graph), consistent with anchoring theta_1=0.
    This matches the 'set theta_1=0 and forget the first column' idea in the doc. 
    """
    K, m, _ = alpha.shape
    pairs = [(0, j) for j in range(1, m)]

    ii, jj, kk_vals, aa, ww = [], [], [], [], []
    for kk, k in enumerate(k_list):
        for (j1, j2) in pairs:
            w = weight[kk, j1, j2]
            if w <= 0:
                continue
            ii.append(j1); jj.append(j2)
            kk_vals.append(float(k))
            aa.append(float(alpha[kk, j1, j2]))
            ww.append(float(w))

    return {
        "i": np.array(ii, dtype=np.int32),
        "j": np.array(jj, dtype=np.int32),
        "k": np.array(kk_vals, dtype=np.float64),
        "a": np.array(aa, dtype=np.float64),
        "w": np.array(ww, dtype=np.float64),
        "m": m
    }

# -----------------------
# Iterative fixed-wrap solver for Eq(3)
# -----------------------
def precompute_reduced_laplacian(meas):
    """
    Build reduced Laplacian L for LS step with theta[0]=0.
    Normal equations from minimizing Σ w (k(θ_i-θ_j) - (a+2πn))^2.
    """
    i = meas["i"]; j = meas["j"]; k = meas["k"]; w = meas["w"]; m = meas["m"]
    wk2 = w * (k**2)

    diag = np.zeros(m-1, dtype=np.float64)
    rows, cols, data = [], [], []

    for t in range(len(i)):
        ui, uj = int(i[t]), int(j[t])
        val = wk2[t]

        if ui == 0 and uj == 0:
            continue
        if ui == 0:
            rj = uj - 1
            diag[rj] += val
        elif uj == 0:
            ri = ui - 1
            diag[ri] += val
        else:
            ri, rj = ui - 1, uj - 1
            diag[ri] += val
            diag[rj] += val
            rows += [ri, rj]
            cols += [rj, ri]
            data += [-val, -val]

    rows += list(range(m-1))
    cols += list(range(m-1))
    data += list(diag)

    return sp.coo_matrix((data, (rows, cols)), shape=(m-1, m-1)).tocsr()

def assemble_rhs_given_wraps(meas, n_wrap):
    """
    RHS for reduced system L θ = g given wraps n:
      k(θ_i - θ_j) = a + 2π n
    """
    i = meas["i"]; j = meas["j"]; k = meas["k"]; a = meas["a"]; w = meas["w"]; m = meas["m"]
    g_full = np.zeros(m, dtype=np.float64)

    b = a + TAU * n_wrap
    t = w * k * b
    np.add.at(g_full, i,  t)
    np.add.at(g_full, j, -t)

    return g_full[1:]  # remove anchor variable θ_0

def iterative_eq3_fixed_wraps(meas, theta0, max_iter=50, tol=1e-12,
                             cg_rtol=1e-12, cg_maxiter=5000, damping=1.0,
                             verbose=True):
    """
    Alternate:
      n <- round((k(θ_i-θ_j)-a)/2π)
      θ <- argmin Σ w (k(θ_i-θ_j) - (a+2πn))^2  with θ_0=0
    IMPORTANT: theta0 must be in the gauge θ_0=0 (i.e., relative to the first obs).
    """
    m = meas["m"]
    theta = np.array(theta0, dtype=np.float64).copy()

    # --- CRITICAL FIX: enforce document gauge θ_1 = 0 (index 0 in code) ---
    theta = wrap_to_pi(theta - theta[0])
    theta[0] = 0.0

    L = precompute_reduced_laplacian(meas)
    last_n = None

    for it in range(max_iter):
        i = meas["i"]; j = meas["j"]; k = meas["k"]; a = meas["a"]

        pred = k * (theta[i] - theta[j])
        n_new = np.rint((pred - a) / TAU).astype(np.int64)

        g = assemble_rhs_given_wraps(meas, n_new)
        theta_red, cg_info = spla.cg(L, g, rtol=cg_rtol, maxiter=cg_maxiter)

        theta_ls = np.zeros(m, dtype=np.float64)
        theta_ls[0] = 0.0
        theta_ls[1:] = theta_red

        theta = (1.0 - damping) * theta + damping * theta_ls
        theta = wrap_to_pi(theta)
        theta[0] = 0.0

        res = wrap_to_pi(k * (theta[i] - theta[j]) - a)
        rms = float(np.sqrt(np.mean(res**2)))

        stable = (last_n is not None and np.array_equal(n_new, last_n))
        if verbose:
            print(f"[iter {it:02d}] wrapped-RMS={rms:.3e} wraps_stable={stable} cg_info={cg_info}")

        if stable and rms < tol:
            return theta, n_new, {"status": "converged", "iters": it+1, "wrapped_rms": rms}

        last_n = n_new

    return theta, last_n, {"status": "max_iter", "iters": max_iter, "wrapped_rms": rms}

# -----------------------
# Demo: generate + solve
# -----------------------
if __name__ == "__main__":
    # ---- USER CHOICES ----
    sigma = 0.01          # noise std (free choice)
    m = 20               # number of samples/observations (free choice)

    # Option A: provide your own signal array (J inferred)
    x_true = np.zeros(15); 
    x_true[:len(x_true)//2] = 1; 
    x = x_true
    J = len(x)

    x, Y, shifts, theta_abs, theta_true = generate_mra_data(x, sigma=sigma, m=m, seed=3)

    Xhat = np.fft.fft(Y, axis=1)
    alpha, weight, k_list = alphas_from_dfts_ratio_phase(Xhat, k_list=list(range(1, J)))
    meas = build_star_measurements(alpha, weight, k_list)

    # Test: initialize with TRUE thetas (document gauge: theta_1=0)
    theta0 = np.zeros_like(theta_true)
    # print(theta_true)
    # theta0 = theta_true.copy()

    theta_est, n_wrap, info = iterative_eq3_fixed_wraps(meas, theta0, verbose=True)
    print("info:", info)

    # Evaluate (both are in gauge theta[0]=0)
    err = wrap_to_pi(theta_est - theta_true)
    print("wrapped RMSE:", np.sqrt(np.mean(err**2)))
    print("max abs err:", np.max(np.abs(err)))
    for idx in range(m):
        print(f"{theta_est[idx], theta_true[idx]}")
