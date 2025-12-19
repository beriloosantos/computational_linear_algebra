import os
import time
import requests
import tarfile
import io
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.sparse import issparse, eye, csr_matrix, diags
from scipy.sparse.linalg import splu, eigsh, eigs, svds, norm as spnorm
from scipy.io import mmread

# Matrix Information
MATRICES_INFO = [
    {
        "id": "a",
        "name": "bcsstk16",
        "url": "https://sparse.tamu.edu/MM/HB/bcsstk16.tar.gz",
        "title": "(a) Symmetric Positive Definite\n(bcsstk16)",
        "color": "#1f77b4",
        "sigma": 10.0,
        "description": "Stiffness matrix from structural engineering (finite element problem)."
    },
    {
        "id": "b",
        "name": "fpga_dcop_17",
        "url": "https://sparse.tamu.edu/MM/Sandia/fpga_dcop_17.tar.gz",
        "title": "(b) Nonsymmetric\n(fpga_dcop_17)",
        "color": "#ff7f0e",
        "sigma": 0.5,
        "description": "Directed graph matrix from FPGA circuit optimization (transport model)."
    },
    {
        "id": "c",
        "name": "FEM_3D_thermal2",
        "url": "https://sparse.tamu.edu/MM/Botonakis/FEM_3D_thermal2.tar.gz",
        "title": "(c) Large and Sparse\n(FEM_3D_thermal2)",
        "color": "#2ca02c",
        "sigma": 0.01,
        "description": "3D thermal finite element problem (n > 2000, highly sparse)."
    }
]

def download_and_load_matrix(name, url):
    """
    Downloads a matrix from SuiteSparse and loads it into a CSR sparse format.
    Returns a scipy sparse matrix in CSR format.
    """
    filename_mtx = f"{name}.mtx"

    if os.path.exists(filename_mtx):
        print(f"Loading existing {filename_mtx}...")
        return mmread(filename_mtx).tocsr()

    print(f"Downloading {name} from {url}...")
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Download failed for {name}. Status: {response.status_code}")

    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        mtx_member = next((m for m in tar.getmembers() if m.name.endswith('.mtx')), None)
        if mtx_member is None:
            raise Exception(f".mtx file not found in archive for {name}")
        f = tar.extractfile(mtx_member)
        with open(filename_mtx, 'wb') as out:
            out.write(f.read())

    return mmread(filename_mtx).tocsr()

# ============================================================================
# 2. EIGENVALUE SOLVERS
# ============================================================================

def normalize_vector(v):
    """Normalizes a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def rayleigh_quotient(A, v):
    """Computes the Rayleigh quotient for a given matrix and vector."""
    v = v.reshape(-1, 1)
    numerator = v.T @ (A @ v)
    denominator = v.T @ v
    return float(numerator[0, 0] / denominator[0, 0])

def get_wilkinson_shift(A_sub):
    """Computes the Wilkinson shift for a 2x2 trailing submatrix."""
    if A_sub.shape[0] < 2:
        return A_sub[0, 0]
    m = A_sub.shape[0]
    a_mm = A_sub[m-1, m-1]
    a_mm1 = A_sub[m-1, m-2]
    a_m1m1 = A_sub[m-2, m-2]
    delta = (a_m1m1 - a_mm) / 2.0
    sign_delta = 1.0 if delta >= 0 else -1.0
    denom = abs(delta) + np.sqrt(delta**2 + a_mm1**2)
    if denom == 0:
        return a_mm
    mu = a_mm - (sign_delta * a_mm1**2) / denom
    return mu

def power_method(A, true_lambda=None, max_iter=10000, tol=1e-8):
    """
    Power method for dominant eigenvalue/eigenvector.
    Returns: (lambda, v, iterations, elapsed_time, residual_history, error_history)
    """
    start_time = time.perf_counter()
    n = A.shape[0]
    v = normalize_vector(np.random.randn(n))
    lambda_old = 0.0

    res_history = []
    err_history = []

    for k in range(1, max_iter + 1):
        w = A @ v
        v_new = normalize_vector(w)
        lambda_new = rayleigh_quotient(A, v_new)

        # Compute residual: ||A*v - λ*v||
        residual = np.linalg.norm(A @ v_new - lambda_new * v_new)
        res_history.append(residual)

        if true_lambda is not None:
            err = abs(lambda_new - true_lambda) / abs(true_lambda)
            err_history.append(err)

        if abs(lambda_new - lambda_old) < tol:
            elapsed = time.perf_counter() - start_time
            return float(lambda_new), v_new, k, elapsed, res_history, err_history

        v = v_new
        lambda_old = lambda_new

    elapsed = time.perf_counter() - start_time
    return float(lambda_new), v, max_iter, elapsed, res_history, err_history

def inverse_power_method(A, sigma, true_lambda=None, max_iter=10000, tol=1e-8):
    """
    Inverse power method with shift σ.
    Returns: (lambda, v, iterations, elapsed_time, residual_history, error_history)
    """
    start_time = time.perf_counter()
    n = A.shape[0]
    v = normalize_vector(np.random.randn(n))
    lambda_old = 0.0

    # Prepare linear system solver: (A - σI) x = v
    I = eye(n, format='csc') if issparse(A) else np.eye(n)
    M = A - sigma * I

    if issparse(M):
        try:
            solve = splu(M.tocsc()).solve
        except RuntimeError:
            # Fallback to dense solve if sparse LU fails
            solve = lambda b: la.solve(M.toarray(), b)
    else:
        lu, piv = la.lu_factor(M)
        solve = lambda b: la.lu_solve((lu, piv), b)

    res_history = []
    err_history = []

    for k in range(1, max_iter + 1):
        y = solve(v)
        v_new = normalize_vector(y)
        lambda_new = rayleigh_quotient(A, v_new)

        residual = np.linalg.norm(A @ v_new - lambda_new * v_new)
        res_history.append(residual)

        if true_lambda is not None:
            err = abs(lambda_new - true_lambda) / abs(true_lambda)
            err_history.append(err)

        if abs(lambda_new - lambda_old) < tol:
            elapsed = time.perf_counter() - start_time
            return lambda_new, v_new, k, elapsed, res_history, err_history

        v = v_new
        lambda_old = lambda_new

    elapsed = time.perf_counter() - start_time
    return lambda_new, v, max_iter, elapsed, res_history, err_history

def qr_iteration(H, max_iter=20000, tol=1e-8, use_shift=True, return_eigenvectors=False):
    """
    QR iteration with Wilkinson shift.
    Returns eigenvalues (and eigenvectors if requested) of upper Hessenberg matrix H.
    """
    H = H.copy().astype(float)
    n = H.shape[0]
    eigenvalues = []
    eigenvectors = None
    if return_eigenvectors:
        eigenvectors = np.eye(n)

    m = n
    iter_count = 0

    while m > 1 and iter_count < max_iter:
        iter_count += 1

        # Check for convergence of the bottom subdiagonal element
        if abs(H[m-1, m-2]) < tol:
            eigenvalues.append(H[m-1, m-1])
            m -= 1
            continue

        # Apply shift
        if use_shift:
            sigma = get_wilkinson_shift(H[:m, :m])
        else:
            sigma = 0.0

        # QR decomposition: H - sigma*I = Q*R
        H_shifted = H[:m, :m] - sigma * np.eye(m)
        Q, R = np.linalg.qr(H_shifted)

        # Update: H = R*Q + sigma*I
        H[:m, :m] = R @ Q + sigma * np.eye(m)

        # Accumulate eigenvectors if requested
        if return_eigenvectors:
            eigenvectors[:, :m] = eigenvectors[:, :m] @ Q

    if m == 1:
        eigenvalues.append(H[0, 0])

    eigenvalues = np.array(eigenvalues)

    if return_eigenvectors:
        return eigenvalues, eigenvectors, iter_count
    else:
        return eigenvalues, iter_count

def lanczos_method(A, k_subspace, true_lambda=None, return_eigenvectors=False, qr_iter_func=qr_iteration):
    """
    Lanczos method for symmetric matrices.
    Returns approximate eigenvalues of A and convergence history.
    """
    start_time = time.perf_counter()
    n = A.shape[0]
    
    # Initialize Lanczos vectors and coefficients
    alphas = []  # diagonal entries
    betas = []   # off-diagonal entries (beta_0 = 0)
    
    v0 = normalize_vector(np.random.randn(n))
    v_prev = np.zeros(n)
    v = v0.copy()
    beta = 0.0
    
    # Store basis vectors for eigenvector recovery
    Q = np.zeros((n, k_subspace))
    if k_subspace > 0:
        Q[:, 0] = v0
    
    err_history = []
    res_history = []

    for j in range(k_subspace):
        # Lanczos iteration
        w = A @ v
        alpha = np.dot(w, v)
        alphas.append(alpha)
        
        w = w - alpha * v - beta * v_prev
        
        beta_new = np.linalg.norm(w)
        
        # Store beta (for next iteration's recurrence)
        if j < k_subspace - 1:
            betas.append(beta_new)
        
        # Store current basis vector (for eigenvector recovery)
        if j < k_subspace:
            Q[:, j] = v
        
        # Build tridiagonal matrix T_j (size j+1 x j+1)
        if j == 0:
            T = np.array([[alphas[0]]])
        else:
            # Use only the first (len(alphas)-1) betas for off-diagonals
            betas_for_T = betas[:len(alphas)-1]
            T = np.diag(alphas) + np.diag(betas_for_T, k=1) + np.diag(betas_for_T, k=-1)

        # Monitor convergence
        if true_lambda is not None and T.shape[0] > 0:
            # Compute eigenvalues of T
            if return_eigenvectors:
                ritz_vals, ritz_vecs_T, _ = qr_iter_func(T, use_shift=True, return_eigenvectors=True)
            else:
                ritz_vals, _ = qr_iter_func(T, use_shift=True, return_eigenvectors=False)
            
            current_max = np.max(np.abs(ritz_vals))
            err = abs(current_max - true_lambda) / abs(true_lambda)
            err_history.append(err)
            
            # Calculate residual for dominant Ritz pair
            if return_eigenvectors and len(betas) > 0:
                idx_max = np.argmax(np.abs(ritz_vals))
                y = ritz_vecs_T[:, idx_max]
                # Lanczos residual formula: |beta_j * y[-1]|
                res = abs(beta_new * y[-1])
                res_history.append(res)
        
        # Check for breakdown
        if beta_new < 1e-12:
            break
        
        # Prepare for next iteration
        v_prev = v.copy()
        if j < k_subspace - 1:
            v = w / beta_new
            beta = beta_new
    
    # Build final tridiagonal matrix
    if len(alphas) == 1:
        T_final = np.array([[alphas[0]]])
    else:
        # Use all betas except the last one for matrix construction
        betas_for_matrix = betas[:len(alphas)-1] if len(betas) >= len(alphas) else betas
        T_final = np.diag(alphas) + np.diag(betas_for_matrix, k=1) + np.diag(betas_for_matrix, k=-1)
    
    # Compute final eigenvalues and eigenvectors of T
    if return_eigenvectors:
        final_vals, final_vecs_T, _ = qr_iter_func(T_final, use_shift=True, return_eigenvectors=True)
        # Recover approximate eigenvectors of A: v_i = Q * y_i
        # Use only the basis vectors we actually computed
        basis_size = min(len(alphas), k_subspace)
        final_vecs = Q[:, :basis_size] @ final_vecs_T
    else:
        final_vals, _ = qr_iter_func(T_final, use_shift=True, return_eigenvectors=False)
        final_vecs = None
    
    elapsed = time.perf_counter() - start_time
    return final_vals, final_vecs, T_final, min(len(alphas), k_subspace), elapsed, res_history, err_history

def arnoldi_method(A, k_subspace, true_lambda=None, return_eigenvectors=False, qr_iter_func=qr_iteration):
    """
    Arnoldi method for general matrices.
    Returns approximate eigenvalues of A and convergence history.
    """
    start_time = time.perf_counter()
    n = A.shape[0]
    
    # Initialize Arnoldi basis and Hessenberg matrix
    Q = np.zeros((n, k_subspace + 1))
    H = np.zeros((k_subspace + 1, k_subspace))
    
    v0 = normalize_vector(np.random.randn(n))
    Q[:, 0] = v0
    
    err_history = []
    res_history = []

    for k in range(k_subspace):
        v = A @ Q[:, k]
        
        # Modified Gram-Schmidt
        for j in range(k + 1):
            H[j, k] = np.dot(Q[:, j], v)
            v = v - H[j, k] * Q[:, j]
        
        H[k+1, k] = np.linalg.norm(v)
        
        # Monitor convergence using current Hessenberg matrix
        H_current = H[:k+1, :k+1]
        if return_eigenvectors:
            ritz_vals, ritz_vecs_H, _ = qr_iter_func(H_current, use_shift=True, return_eigenvectors=True)
        else:
            ritz_vals, _ = qr_iter_func(H_current, use_shift=True, return_eigenvectors=False)

        if true_lambda is not None and len(ritz_vals) > 0:
            current_max = np.max(np.abs(ritz_vals))
            err = abs(current_max - true_lambda) / abs(true_lambda)
            err_history.append(err)
        
        # Calculate residual for dominant Ritz pair (for plotting)
        if len(ritz_vals) > 0:
            idx_max = np.argmax(np.abs(ritz_vals))
            if return_eigenvectors and hasattr(ritz_vecs_H, 'shape'):
                y = ritz_vecs_H[:, idx_max]
                # Arnoldi residual formula: |h_{j+1,j} * y[-1]|
                res = abs(H[k+1, k] * y[-1]) if H[k+1, k] > 0 else 0.0
                res_history.append(res)

        if H[k+1, k] < 1e-12:
            break
            
        Q[:, k+1] = v / H[k+1, k]

    # Final Hessenberg matrix
    H_final = H[:k_subspace, :k_subspace]
    
    # Compute eigenvalues and eigenvectors of H
    if return_eigenvectors:
        final_vals, final_vecs_H, _ = qr_iter_func(H_final, use_shift=True, return_eigenvectors=True)
        # Recover approximate eigenvectors of A
        final_vecs = Q[:, :k_subspace] @ final_vecs_H
    else:
        final_vals, _ = qr_iter_func(H_final, use_shift=True, return_eigenvectors=False)
        final_vecs = None

    elapsed = time.perf_counter() - start_time
    return final_vals, final_vecs, H_final, k+1, elapsed, res_history, err_history

def krylov_subspace_method(A, k_subspace, true_lambda=None, is_symmetric=True, 
                           num_eigenvalues=5, qr_iter_func=qr_iteration):
    """
    Wrapper function for Krylov subspace methods (Lanczos for symmetric, Arnoldi for general).
    Returns k largest eigenvalues and their eigenvectors.
    """
    if is_symmetric:
        # Use Lanczos for symmetric matrices
        eigenvalues, eigenvectors, T_matrix, iterations, elapsed, res_history, err_history = lanczos_method(
            A, k_subspace, true_lambda=true_lambda, return_eigenvectors=True, qr_iter_func=qr_iter_func
        )
        
        # Get largest magnitude eigenvalues
        idx_sorted = np.argsort(np.abs(eigenvalues))[::-1]
        top_eigenvalues = eigenvalues[idx_sorted[:num_eigenvalues]]
        top_eigenvectors = eigenvectors[:, idx_sorted[:num_eigenvalues]] if eigenvectors is not None else None
        
        method_name = "Lanczos"
        
    else:
        # Use Arnoldi for non-symmetric matrices
        eigenvalues, eigenvectors, H_matrix, iterations, elapsed, res_history, err_history = arnoldi_method(
            A, k_subspace, true_lambda=true_lambda, return_eigenvectors=True, qr_iter_func=qr_iter_func
        )
        
        # Get largest magnitude eigenvalues
        idx_sorted = np.argsort(np.abs(eigenvalues))[::-1]
        top_eigenvalues = eigenvalues[idx_sorted[:num_eigenvalues]]
        top_eigenvectors = eigenvectors[:, idx_sorted[:num_eigenvalues]] if eigenvectors is not None else None
        
        method_name = "Arnoldi"
        T_matrix = H_matrix  # For consistent return
    
    return top_eigenvalues, top_eigenvectors, T_matrix, iterations, elapsed, res_history, err_history, method_name

# ============================================================================
# 3. NUMERICAL ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_matrix(matrix_info, k_subspace=50, num_eigenvalues=5):
    """
    Comprehensive analysis of a single matrix using all eigenvalue methods.
    Returns a dictionary with all results and metrics.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {matrix_info['name']}")
    print(f"{'='*60}")
    
    try:
        # Load the matrix
        A = download_and_load_matrix(matrix_info['name'], matrix_info['url'])
        n = A.shape[0]
        nnz = A.nnz
        sparsity = (nnz / (n * n)) * 100 if n < 10000 else 100 * nnz / (n * n)
        
        print(f"Dimensions: {n} x {n}")
        print(f"Non-zero elements: {nnz} (Sparsity: {sparsity:.6f}%)")
        
        # Check symmetry
        is_sym = False
        if hasattr(A, 'get_shape'):
            try:
                diff = A - A.T
                is_sym = (abs(diff).sum() < 1e-10)
            except:
                is_sym = False
        print(f"Symmetric: {is_sym}")
        
        # Get reference eigenvalues using SciPy (ground truth)
        print("\nComputing reference eigenvalues with SciPy...")
        ref_max, ref_sigma = None, None
        
        try:
            if is_sym:
                # Symmetric matrices
                ref_max_vals, _ = eigsh(A, k=1, which='LM')
                ref_max = ref_max_vals[0]
                ref_sigma_vals, _ = eigsh(A, k=1, sigma=matrix_info['sigma'], which='LM')
                ref_sigma = ref_sigma_vals[0]
                
                # Get multiple eigenvalues for comparison
                ref_multiple_vals, _ = eigsh(A, k=min(num_eigenvalues, n-1), which='LM')
            else:
                # Non-symmetric matrices
                ref_max_vals, _ = eigs(A, k=1, which='LM', maxiter=50000, tol=1e-6)
                ref_max = ref_max_vals[0]
                ref_sigma_vals, _ = eigs(A, k=1, sigma=matrix_info['sigma'], which='LM', maxiter=50000, tol=1e-6)
                ref_sigma = ref_sigma_vals[0]
                
                # Get multiple eigenvalues
                ref_multiple_vals, _ = eigs(A, k=min(num_eigenvalues, n-1), which='LM')
                
            print(f"Reference λ_max: {ref_max:.6e}")
            print(f"Reference λ near σ={matrix_info['sigma']}: {ref_sigma:.6e}")
            
        except Exception as e:
            print(f"Warning: Could not compute all reference eigenvalues: {e}")
            ref_max = 1.0  # Placeholder for relative error calculation
        
        # Run all eigenvalue methods
        results = {}
        
        # 1. Power Method (Dominant eigenvalue)
        print("\n1. Running Power Method...")
        lambda_power, v_power, k_power, t_power, res_power, err_power = power_method(
            A, true_lambda=ref_max, max_iter=5000
        )
        results['power'] = {
            'lambda': lambda_power,
            'eigenvector': v_power,
            'iterations': k_power,
            'time': t_power,
            'residual_history': res_power,
            'error_history': err_power,
            'final_residual': res_power[-1] if res_power else None,
            'final_error': err_power[-1] if err_power else None
        }
        print(f"   λ: {lambda_power:.6e}, Iterations: {k_power}, Time: {t_power:.4f}s")
        
        # 2. Inverse Power Method (Eigenvalue near sigma)
        print("\n2. Running Inverse Power Method...")
        lambda_inv, v_inv, k_inv, t_inv, res_inv, err_inv = inverse_power_method(
            A, sigma=matrix_info['sigma'], true_lambda=ref_sigma, max_iter=5000
        )
        results['inverse'] = {
            'lambda': lambda_inv,
            'eigenvector': v_inv,
            'iterations': k_inv,
            'time': t_inv,
            'residual_history': res_inv,
            'error_history': err_inv,
            'final_residual': res_inv[-1] if res_inv else None,
            'final_error': err_inv[-1] if err_inv else None
        }
        print(f"   λ near σ={matrix_info['sigma']}: {lambda_inv:.6e}, Iterations: {k_inv}, Time: {t_inv:.4f}s")
        
        # 3. QR Iteration (Full spectrum of a reduced matrix)
        print("\n3. Running QR Iteration on reduced matrix...")
        # First reduce matrix using Krylov subspace
        if is_sym:
            _, _, T_matrix, _, _, _, _ = lanczos_method(A, min(100, n//2), return_eigenvectors=False)
        else:
            _, _, T_matrix, _, _, _, _ = arnoldi_method(A, min(100, n//2), return_eigenvectors=False)
        
        # QR without shift
        qr_vals_no_shift, qr_iters_no_shift = qr_iteration(T_matrix, use_shift=False)
        # QR with Wilkinson shift
        qr_vals_shift, qr_iters_shift = qr_iteration(T_matrix, use_shift=True)
        
        results['qr'] = {
            'no_shift': {
                'eigenvalues': qr_vals_no_shift,
                'iterations': qr_iters_no_shift,
                'dominant': np.max(np.abs(qr_vals_no_shift))
            },
            'with_shift': {
                'eigenvalues': qr_vals_shift,
                'iterations': qr_iters_shift,
                'dominant': np.max(np.abs(qr_vals_shift))
            },
            'speedup': qr_iters_no_shift / max(qr_iters_shift, 1)
        }
        print(f"   QR no shift: {qr_iters_no_shift} iterations")
        print(f"   QR with shift: {qr_iters_shift} iterations")
        print(f"   Speedup: {results['qr']['speedup']:.2f}x")
        
        # 4. Krylov Subspace Method (Multiple eigenvalues)
        print("\n4. Running Krylov Subspace Method...")
        krylov_vals, krylov_vecs, krylov_T, krylov_iters, krylov_time, krylov_res, krylov_err, method_name = krylov_subspace_method(
            A, k_subspace=k_subspace, true_lambda=ref_max, is_symmetric=is_sym, 
            num_eigenvalues=num_eigenvalues
        )
        
        results['krylov'] = {
            'eigenvalues': krylov_vals,
            'eigenvectors': krylov_vecs,
            'method': method_name,
            'iterations': krylov_iters,
            'time': krylov_time,
            'residual_history': krylov_res,
            'error_history': krylov_err,
            'T_matrix': krylov_T
        }
        print(f"   Method: {method_name}, Top {num_eigenvalues} eigenvalues:")
        for i, val in enumerate(krylov_vals[:5]):
            print(f"     λ_{i+1}: {val:.6e}")
        
        # Store matrix properties
        results['matrix_info'] = {
            'name': matrix_info['name'],
            'dimensions': n,
            'nnz': nnz,
            'sparsity': sparsity,
            'symmetric': is_sym,
            'sigma': matrix_info['sigma'],
            'reference': {
                'lambda_max': ref_max,
                'lambda_sigma': ref_sigma,
                'multiple': ref_multiple_vals if 'ref_multiple_vals' in locals() else None
            }
        }

        create_convergence_plots(results, matrix_info)
        
        return results
        
    except Exception as e:
        print(f"Error analyzing matrix {matrix_info['name']}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_convergence_plots(results, matrix_info):
    """
    Creates convergence plots for error and residual.
    """
    for method_key in ['power', 'inverse', 'krylov']:
        if method_key not in results:
            continue
        if 'error_history' in results[method_key]:
            err_data = results[method_key]['error_history']
            if err_data is not None and len(err_data) > 0:
                err_array = np.array(err_data, dtype=np.float64)
                mask_invalid = ~np.isfinite(err_array)
                if np.any(mask_invalid):
                    err_array[mask_invalid] = 1.0
                err_array = np.clip(err_array, 1e-16, 1e10)
                results[method_key]['error_history'] = err_array
            else:
                results[method_key]['error_history'] = []

        if 'residual_history' in results[method_key]:
            res_data = results[method_key]['residual_history']
            if res_data is not None and len(res_data) > 0:
                res_array = np.array(res_data, dtype=np.float64)
                mask_invalid = ~np.isfinite(res_array)
                if np.any(mask_invalid):
                    res_array[mask_invalid] = 1.0
                res_array = np.clip(res_array, 1e-16, 1e10)
                results[method_key]['residual_history'] = res_array
            else:
                results[method_key]['residual_history'] = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"Convergence Analysis: {matrix_info['name']}", fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    colors = {'power': 'blue', 'inverse': 'green', 'krylov': 'red'}
    
    for method_key, color in colors.items():
        if method_key in results and 'error_history' in results[method_key]:
            data = results[method_key]['error_history']
            if len(data) > 0 and isinstance(data, (list, np.ndarray)):
                x_indices = np.arange(len(data))
                valid_mask = np.isfinite(data)
                if np.any(valid_mask):
                    label = f"{method_key.capitalize()}"
                    if 'iterations' in results[method_key]:
                        label += f" (k={results[method_key]['iterations']})"
                    
                    if method_key == 'krylov' and 'method' in results[method_key]:
                        label = f"{results[method_key]['method']} {label}"
                    
                    ax1.semilogy(x_indices[valid_mask], np.array(data)[valid_mask], 
                                label=label, color=color, alpha=0.7, linewidth=2)
    
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Relative Error (log scale)")
    ax1.set_title("Error Convergence")
    ax1.grid(True, alpha=0.3)
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend()

    ax2 = axes[0, 1]
    
    for method_key, color in colors.items():
        if method_key in results and 'residual_history' in results[method_key]:
            data = results[method_key]['residual_history']
            if len(data) > 0 and isinstance(data, (list, np.ndarray)):
                x_indices = np.arange(len(data))
                valid_mask = np.isfinite(data)
                if np.any(valid_mask):
                    label = f"{method_key.capitalize()}"
                    if method_key == 'krylov' and 'method' in results[method_key]:
                        label = f"{results[method_key]['method']}"
                    
                    ax2.semilogy(x_indices[valid_mask], np.array(data)[valid_mask], 
                                label=label, color=color, alpha=0.7, linewidth=2)
    
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Residual ||Av - λv|| (log scale)")
    ax2.set_title("Residual Convergence")
    ax2.grid(True, alpha=0.3)
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend()

    ax3 = axes[1, 0]
    methods = []
    times = []
    method_colors = []
    
    for method_key, color in colors.items():
        if method_key in results and 'time' in results[method_key]:
            time_val = results[method_key]['time']
            if time_val is not None and np.isfinite(time_val):
                label = method_key.capitalize()
                if method_key == 'krylov' and 'method' in results[method_key]:
                    label = results[method_key]['method']
                methods.append(label)
                times.append(time_val)
                method_colors.append(color)
    
    if methods:
        bars = ax3.bar(methods, times, color=method_colors, alpha=0.7)
        ax3.set_xlabel("Method")
        ax3.set_ylabel("Time (seconds)")
        ax3.set_title("Computational Time Comparison")

        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(times),
                    f'{time_val:.4f}s', ha='center', va='bottom', fontsize=9)

    ax4 = axes[1, 1]
    
    if ('matrix_info' in results and 
        'reference' in results['matrix_info'] and 
        'lambda_max' in results['matrix_info']['reference'] and
        results['matrix_info']['reference']['lambda_max'] is not None):
        
        ref_val = results['matrix_info']['reference']['lambda_max']
        errors = []
        error_labels = []
        error_colors = []

        if 'power' in results and 'lambda' in results['power']:
            lambda_power = results['power']['lambda']
            if lambda_power is not None and np.isfinite(lambda_power) and ref_val != 0:
                err = abs(lambda_power - ref_val) / abs(ref_val)
                if np.isfinite(err):
                    errors.append(err)
                    error_labels.append('Power')
                    error_colors.append('blue')

        if ('reference' in results['matrix_info'] and 
            'lambda_sigma' in results['matrix_info']['reference'] and
            results['matrix_info']['reference']['lambda_sigma'] is not None):
            
            ref_sigma = results['matrix_info']['reference']['lambda_sigma']
            if 'inverse' in results and 'lambda' in results['inverse']:
                lambda_inv = results['inverse']['lambda']
                if (lambda_inv is not None and np.isfinite(lambda_inv) and 
                    ref_sigma != 0 and np.isfinite(ref_sigma)):
                    err = abs(lambda_inv - ref_sigma) / abs(ref_sigma)
                    if np.isfinite(err):
                        errors.append(err)
                        error_labels.append('Inverse')
                        error_colors.append('green')

        if ('krylov' in results and 'eigenvalues' in results['krylov'] and 
            len(results['krylov']['eigenvalues']) > 0):
            
            lambda_krylov = results['krylov']['eigenvalues'][0]
            if (lambda_krylov is not None and np.isfinite(lambda_krylov) and 
                ref_val != 0 and np.isfinite(ref_val)):
                
                err = abs(lambda_krylov - ref_val) / abs(ref_val)
                if np.isfinite(err):
                    errors.append(err)
                    label = results['krylov'].get('method', 'Krylov')
                    error_labels.append(label)
                    error_colors.append('red')

        if errors:
            bars_err = ax4.bar(error_labels, errors, color=error_colors, alpha=0.7)
            ax4.set_yscale('log')
            ax4.set_xlabel("Method")
            ax4.set_ylabel("Final Relative Error (log scale)")
            ax4.set_title("Final Accuracy Comparison")

            for bar, err_val in zip(bars_err, errors):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                        f'{err_val:.2e}', ha='center', va='bottom', fontsize=9, rotation=0)

    plot_filename = f"convergence_{matrix_info['name']}.pdf"
    try:
        plt.savefig(plot_filename, dpi=150)
        print(f"Saved convergence plot: {plot_filename}")
    except Exception as e:
        print(f"Warning: Could not save plot with normal settings: {e}")
        try:
            plt.savefig(plot_filename, dpi=100, bbox_inches=None)
            print(f"Saved convergence plot (fallback): {plot_filename}")
        except Exception as e2:
            print(f"Error: Could not save plot at all: {e2}")
    
    plt.close()

def generate_summary_table(all_results):
    """
    Generates a summary table of all results for the report.
    """
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)

    headers = ["Matrix", "Dim", "NNZ", "Method", "λ", "Iterations", "Time(s)", "Error", "Residual"]
    print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<12} {headers[4]:<15} {headers[5]:<12} {headers[6]:<10} {headers[7]:<12} {headers[8]:<12}")
    print("-"*110)

    for matrix_name, results in all_results.items():
        if results is None:
            continue

        matrix_info = results['matrix_info']

        # Power method results
        power = results['power']
        print(f"{matrix_info['name']:<15} {matrix_info['dimensions']:<10} {matrix_info['nnz']:<10} "
              f"{'Power':<12} {power['lambda']:<15.6e} {power['iterations']:<12} "
              f"{power['time']:<10.4f} {power['final_error']:<12.2e} {power['final_residual']:<12.2e}")

        # Inverse power method results
        inverse = results['inverse']
        # Safe formatting for inverse (handle None values)
        inverse_error_str = "N/A"
        if inverse['final_error'] is not None and np.isfinite(inverse['final_error']):
            inverse_error_str = f"{inverse['final_error']:.2e}"
        
        inverse_residual_str = "N/A"
        if inverse['final_residual'] is not None and np.isfinite(inverse['final_residual']):
            inverse_residual_str = f"{inverse['final_residual']:.2e}"
        
        print(f"{'':<15} {'':<10} {'':<10} {'Inverse':<12} {inverse['lambda']:<15.6e} {inverse['iterations']:<12} "
              f"{inverse['time']:<10.4f} {inverse_error_str:<12} {inverse_residual_str:<12}")

        # Krylov method results (show only dominant eigenvalue)
        krylov = results['krylov']
        if len(krylov['eigenvalues']) > 0:
            print(f"{'':<15} {'':<10} {'':<10} {krylov['method']:<12} {krylov['eigenvalues'][0]:<15.6e} {krylov['iterations']:<12} "
                  f"{krylov['time']:<10.4f} {'N/A':<12} {'N/A':<12}")

    print("="*110)

def visualize_matrices():
    """
    Creates visualization of all matrices' sparsity patterns.
    Hybrid approach with rasterized elements for PDF.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    for i, matrix_info in enumerate(MATRICES_INFO):
        try:
            A = download_and_load_matrix(matrix_info['name'], matrix_info['url'])
            
            ax = axes[i]
            
            # Try spy with rasterized=True for PDF output
            mappable = ax.spy(A, markersize=0.5, color=matrix_info['color'],
                            rasterized=True)  # Critical for PDF
            
            # Force rendering by getting the figure canvas
            fig.canvas.draw()

            # Add matrix information
            title = matrix_info['title']
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            info_text = f"n = {A.shape[0]:,}\nnnz = {A.nnz:,}\n"
            info_text += f"σ = {matrix_info['sigma']}"
            
            ax.text(0.02, 0.1, info_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='left',
                   fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            ax.set_xlabel("Column index")
            ax.set_ylabel("Row index")
            ax.grid(False)

        except Exception as e:
            print(f"Error visualizing {matrix_info['name']}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading\n{matrix_info['name']}",
                        ha='center', va='center', transform=axes[i].transAxes,
                        fontsize=12, color='red')
    
    plt.tight_layout()
    
    # Save with explicit renderer
    try:
        # Force render before saving
        fig.canvas.draw()
        
        # Save with PDF backend explicitly
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages("matrices_sparsity_alt.pdf") as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        print("Saved matrix visualization: matrices_sparsity_alt.pdf")
        
        # Also save as PNG
        fig.savefig("matrices_sparsity.png", dpi=300, bbox_inches='tight')
        print("Saved matrix visualization: matrices_sparsity.png")
        
    except Exception as e:
        print(f"Error saving figure: {e}")

    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("EIGENVALUE ANALYSIS OF SPARSE MATRICES")
    print("="*80)
    
    # Step 1: Visualize all matrices
    print("\nStep 1: Visualizing matrices...")
    visualize_matrices()
    sys.exit()
    # Step 2: Analyze each matrix comprehensively
    print("\nStep 2: Running eigenvalue analysis for each matrix...")
    all_results = {}
    
    for matrix_info in MATRICES_INFO:
        results = analyze_matrix(
            matrix_info, 
            k_subspace=50,          # Size of Krylov subspace
            num_eigenvalues=5       # Number of eigenvalues to compute with Krylov
        )
        all_results[matrix_info['name']] = results
        
        # Small pause between matrices
        time.sleep(1)
    
    # Step 3: Generate summary table
    print("\nStep 3: Generating summary table...")
    generate_summary_table(all_results)
    
    # Step 4: Save detailed results to file
    print("\nStep 4: Saving detailed results...")
    try:
        with open("eigenvalue_results_summary.txt", "w") as f:
            f.write("EIGENVALUE ANALYSIS RESULTS\n")
            f.write("="*60 + "\n\n")
            
            for matrix_name, results in all_results.items():

                matrix_info = results['matrix_info']
                f.write(f"\nMatrix: {matrix_info['name']}\n")
                f.write(f"Dimensions: {matrix_info['dimensions']} × {matrix_info['dimensions']}\n")
                f.write(f"Non-zero elements: {matrix_info['nnz']}\n")
                f.write(f"Sparsity: {matrix_info['sparsity']:.6f}%\n")
                f.write(f"Symmetric: {matrix_info['symmetric']}\n")
                
                if matrix_info['reference']['lambda_max'] is not None:
                    f.write(f"\nReference eigenvalues (SciPy):\n")
                    f.write(f"  Dominant λ: {matrix_info['reference']['lambda_max']:.6e}\n")
                    f.write(f"  λ near σ={matrix_info['sigma']}: {matrix_info['reference']['lambda_sigma']:.6e}\n")
                
                f.write(f"\nPower Method:\n")
                f.write(f"  λ: {results['power']['lambda']:.6e}\n")
                f.write(f"  Iterations: {results['power']['iterations']}\n")
                f.write(f"  Time: {results['power']['time']:.4f}s\n")
                f.write(f"  Final error: {results['power']['final_error']:.2e}\n")
                
                f.write(f"\nInverse Power Method (σ={matrix_info['sigma']}):\n")
                f.write(f"  λ: {results['inverse']['lambda']:.6e}\n")
                f.write(f"  Iterations: {results['inverse']['iterations']}\n")
                f.write(f"  Time: {results['inverse']['time']:.4f}s\n")
                f.write(f"  Final error: {results['inverse']['final_error']:.2e}\n")
                
                f.write(f"\nKrylov Subspace Method ({results['krylov']['method']}):\n")
                for j, val in enumerate(results['krylov']['eigenvalues'][:5]):
                    f.write(f"  λ_{j+1}: {val:.6e}\n")
                f.write(f"  Iterations: {results['krylov']['iterations']}\n")
                f.write(f"  Time: {results['krylov']['time']:.4f}s\n")
                
                f.write(f"\nQR Iteration Speed Comparison:\n")
                f.write(f"  Without shift: {results['qr']['no_shift']['iterations']} iterations\n")
                f.write(f"  With Wilkinson shift: {results['qr']['with_shift']['iterations']} iterations\n")
                f.write(f"  Speedup: {results['qr']['speedup']:.2f}x\n")
                
                f.write("\n" + "-"*60 + "\n")
        
        print("Saved detailed results to: eigenvalue_results_summary.txt")
        
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - matrices_sparsity.png: Sparsity patterns of all matrices")
    print("  - convergence_*.png: Convergence plots for each matrix")
    print("  - eigenvalue_results_summary.txt: Detailed numerical results")

    # print("\nNext steps:")
    # print("  1. Use these results for your PDF report (Part 4: Interpretation)")
    # print("  2. For mechanical vibrations (bcsstk16):")
    # print("     - Largest eigenvalue → highest natural frequency")
    # print("     - Corresponding eigenvector → vibration mode shape")
    # print("     - Smaller eigenvalues → lower frequency modes")
    # print("  3. For thermal analysis (FEM_3D_thermal2):")
    # print("     - Eigenvalues → decay rates in heat diffusion")
    # print("     - Largest λ → fastest decaying modes")