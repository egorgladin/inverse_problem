import numpy as np

from single_index_model import cosine_basis, get_estimate, generate_data, objective, algorithm
from utils import drawer, create_base_figure, build_gif


def test_einsum(M, d, n):
    print('Testing einsum:', end=' ')

    A = np.random.randn(M, d, n)
    res_einsum = np.einsum('ijk,ilk->ijl', A, A)

    res_true = np.empty((M, d, d))
    for i in range(M):
        res_true[i] = A[i] @ A[i].T

    print('Success' if np.allclose(res_einsum, res_true) else 'Fail')


def test_cosine_basis(M, d, n):
    print('Testing cosine_basis:', end=' ')
    u = np.random.randn(M, n)
    Psi = cosine_basis(u, d)
    for i in range(M):
        for j in range(d):
            for k in range(n):
                if np.abs(Psi[i, j, k] - np.cos(np.pi * j * u[i, k])) > 1e-9:
                    print("Fail")
                    return 0
    print("Success")


def visualize_2d():
    n = 50
    p = 2
    d = 4
    noise_theta = 1.
    I_d = np.eye(d)

    sample_size = 30
    n_steps = 10
    rho_0 = 0.1
    rho = 0.5
    rhos = [rho_0 * (0.5 ** i) for i in range(n_steps)]

    X, z, theta0, a0, true_theta = generate_data(n, p, d, noise_theta, None, return_true=True)
    theta0[0] = 0.25
    theta0[1] = np.sqrt(1 - theta0[0]**2)

    obj_func = lambda theta: objective(X, theta, d, 0, None, z, I_d)[0]
    create_base_figure(obj_func, true_theta, rho_0, rho, sample_size)  # X, d, z, I_d,

    target = lambda mean: objective(X, mean[None, :], d, 0, None, z, I_d)[0][0]
    aux_func = lambda m_, s_, w_, k_: drawer(m_, s_, w_, k_, true_theta, target)
    algorithm(n_steps, theta0, a0, X, z, rhos, rhos, sample_size, aux_func=aux_func)

    build_gif(noise_theta, n_steps)


def test_r():
    import os
    os.environ["R_HOME"] = r"D:\Software\R-4.2.1"
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    # from rpy2.robjects.vectors import StrVector

    # utils_ = rpackages.importr('utils')
    # utils_.chooseCRANmirror(ind=1)
    #
    # packageNames = ('MAVE',)
    # packnames_to_install = [x for x in packageNames if not rpackages.isinstalled(x)]
    #
    # # Running R in Python example installing packages:
    # if len(packnames_to_install) > 0:
    #     utils_.install_packages(StrVector(packnames_to_install))

    mave = rpackages.importr('MAVE')

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()

    np.random.seed(0)
    X = np.random.randn(100, 10)
    theta = np.zeros(10)
    theta[-2:] = 1
    y = ((X @ theta)**2)[:, None]

    nr, nc = X.shape
    Xr = robjects.r.matrix(X, nrow=nr, ncol=nc)
    robjects.r.assign("X", Xr)

    nr, nc = y.shape
    yr = robjects.r.matrix(y, nrow=nr, ncol=nc)
    robjects.r.assign("y", yr)

    res = robjects.r("""
        dr.mave <- mave(y~X, max.dim = 1)
        dir <- coef(dr.mave, dim = 1)
        dir
        """)  # , method = 'MEANOPG'
    print(res)


if __name__ == '__main__':
    # M = 10
    # n = 12
    # d = 11
    #
    # test_einsum(M, d, n)
    # test_cosine_basis(M, d, n)
    # visualize_2d()

    test_r()
