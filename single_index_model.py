import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import decomposition
import pickle
from sliced import SlicedInverseRegression, SlicedAverageVarianceEstimation
import rpy2

from utils import plot_results


def cosine_basis(u, d):
    # u has shape (sample_size, n)
    K = np.arange(d, dtype='float')
    arguments = np.pi * u[:, None, :] * K[None, :, None]  # shape (sample_size, d, n)
    return np.cos(arguments)


def get_a_tilde(Psi, G_squared, z, a, rho_a):
    # Psi has shape (sample_size, d, n)
    # G_squared has shape (d, d)
    PsiPsi_T = np.einsum('ijk,ilk->ijl', Psi, Psi)  # shape (sample_size, d, d)
    try:
        D_G_sq_inv = np.linalg.inv(PsiPsi_T + G_squared)  # shape (sample_size, d, d)
    except np.linalg.LinAlgError:
        D_G_sq_inv = np.linalg.pinv(PsiPsi_T + G_squared)

    # Psi @ z has shape (sample_size, d)
    Psi_z = ((Psi @ z + a / rho_a) if rho_a > 0 else Psi @ z)[:, :, None]  # shape (sample_size, d, 1)
    a_tilde = D_G_sq_inv @ Psi_z  # shape (sample_size, d, 1)
    return np.squeeze(a_tilde, axis=2)


def get_estimate(Psi, a):
    # Psi has shape (sample_size, d, n)
    # a has shape (sample_size, d)
    estimate = np.swapaxes(Psi, 1, 2) @ a[:, :, None]  # shape (sample_size, n, 1)
    return np.squeeze(estimate, axis=2)


def objective(X, thetas, d, rho_a, a, z, I_d):
    # thetas has shape (sample_size, p)
    # a has shape (d,)
    # thetas @ X has shape (sample_size, n)
    Psi = cosine_basis(thetas @ X, d)  # shape (sample_size, d, n)
    a_tilde = get_a_tilde(Psi, I_d / rho_a, z, a, rho_a) if rho_a > 0\
        else get_a_tilde(Psi, np.zeros((d, d)), z, None, 0)  # shape (sample_size, d)
    estimate = get_estimate(Psi, a_tilde)  # shape (sample_size, n)
    ls = 0.5 * ((estimate - z) ** 2).sum(axis=1)  # shape (sample_size,)
    return ls, a_tilde


def algorithm(n_steps, theta, a, X, z, rhos_theta, rhos_a, sample_size, seed=0, aux_func=None):
    # X has shape (p, n)
    p = len(theta)
    d = len(a)
    I_d = np.eye(d)
    pca = decomposition.PCA(n_components=1)

    traj = [(theta, a)]
    for k in range(n_steps):  # tqdm(range(n_steps)):
        rho_theta, rho_a = rhos_theta[k], rhos_a[k]
        np.random.seed(seed + k)
        thetas = np.random.normal(loc=theta, scale=np.sqrt(rho_theta), size=(sample_size, p))

        ls, a_tilde = objective(X, thetas, d, rho_a, a, z, I_d)
        weights = scipy.special.softmax(-ls, axis=0)  # shape (sample_size,)

        mat_for_pca = (weights * thetas.T) @ thetas
        pca.fit(mat_for_pca)
        theta = pca.components_[0]
        if np.linalg.norm(theta - traj[0][0]) > np.linalg.norm(theta + traj[0][0]):  # fixme
            theta *= -1.

        Psi = cosine_basis(theta[None, :] @ X, d)  # shape (1, d, n)
        a = np.squeeze(get_a_tilde(Psi, I_d / rho_a, z, a, rho_a), axis=0)  # shape (d,)
        traj.append((theta, a))

        if aux_func:  # aux_func can be used e.g. for plotting
            aux_func(traj[-2][0], thetas, weights, k)

    return traj


def generate_data(n, p, d, noise_theta, noise_a, noise_m=0., plot=True, return_true=False):
    np.random.seed(0)
    X = np.random.randn(p, n)

    np.random.seed(10)
    true_theta = np.random.randn(p)
    true_theta /= np.linalg.norm(true_theta)

    theta_X = true_theta[None, :] @ X
    Psi = cosine_basis(theta_X, d)  # shape (1, d, n)

    np.random.seed(2)
    true_a = np.random.randn(d)
    z = get_estimate(Psi, true_a[None, :])  # shape (1, n)
    np.random.seed(5)
    z = np.squeeze(z, axis=0) + noise_m * np.random.randn(n)

    if plot:
        grid = np.linspace(theta_X.min() - 0.1, theta_X.max() + 0.1, num=500)
        basis = cosine_basis(grid[None, :], d)
        f = get_estimate(basis, true_a[None, :])  # shape (1, n)

        plt.figure(figsize=(12, 5))
        plt.plot(grid, np.squeeze(f, axis=0), label=r"$m(u)=\Psi(u)^\top a^*$")
        plt.scatter(np.squeeze(theta_X, axis=0), z, c='k', label=r"$X^\top \theta^*$")
        plt.legend()
        plt.savefig(f"plots/target_{n}_{p}_{d}{f'_noise_{noise_m}' if noise_m != 0 else ''}.png", bbox_inches='tight')

    np.random.seed(3)
    theta0 = true_theta + noise_theta * np.random.randn(p)
    theta0 /= np.linalg.norm(theta0)
    print(f"Distance from solution: {np.linalg.norm(theta0 - true_theta):.3f}")

    np.random.seed(4)
    a0 = np.zeros_like(true_a)  # true_a + noise_a * np.random.randn(d)

    return (X, z, theta0, a0, true_theta) if return_true else (X, z, theta0, a0)


def grid_search():
    n = 200
    p = 10
    d = 10
    noise_theta = 2.
    noise_a = 0.1

    X, z, theta0, a0, true_theta = generate_data(n, p, d, noise_theta, noise_a, plot=False, return_true=True)

    n_steps = 30
    rhos_0 = [0.5, 1]  # [0.01, 0.04, 0.1]  # 0.004,
    rhos = [0.6, 0.8]  # [0.6, 0.7, 0.8, 0.9]
    sample_sizes = [1600]  # [200, 400, 800, 1600]
    I_d = np.eye(d)

    RUN_ALG = True
    if RUN_ALG:
        n_takes = 50
        obj_values = []
        errors = []
        counter = 0
        total = len(rhos_0) * len(rhos) * len(sample_sizes) * n_takes
        for rho in rhos:
            for sample_size in sample_sizes:
                for rho_0 in rhos_0:
                    rhos_theta = [rho_0 * (rho ** i) for i in range(n_steps)]
                    rhos_a = [rho_0 * (rho ** i) for i in range(n_steps)]

                    obj_vals_takes = []
                    errs_takes = []
                    for take in tqdm(range(n_takes)):
                        traj = algorithm(n_steps, theta0, a0, X, z, rhos_theta, rhos_a, sample_size, seed=take*n_steps)
                        obj_vals = [objective(X, theta[None, :], d, 0, a, z, I_d)[0].item() for theta, a in traj]
                        errs = [np.linalg.norm(theta - true_theta) for theta, _ in traj]
                        obj_vals_takes.append(obj_vals)
                        errs_takes.append(errs)

                        counter += 1
                        if counter % 400 == 0:
                            print(f"\nFinished {counter}/{total} runs")

                    # obj_vals_avg = np.row_stack(obj_vals_takes).mean(axis=0)
                    # errs_avg = np.row_stack(errs_takes).mean(axis=0)
                    # obj_values.append(obj_vals_avg)
                    # errors.append(errs_avg
                    obj_values.append(obj_vals_takes)
                    errors.append(errs_takes)

        with open('results3.pickle', 'wb') as handle:
            pickle.dump((obj_values, errors), handle)

    else:
        with open('results3.pickle', 'rb') as handle:
            tup = pickle.load(handle)
            obj_values, errors = tup

    plot_results(rhos, sample_sizes, rhos_0, obj_values, plot_many=True, suffix='3')
    plot_results(rhos, sample_sizes, rhos_0, errors, is_objective=False, plot_many=True, suffix='3')


if __name__ == '__main__':
    n = 200
    p = 10
    d = 10
    noise_theta = 0.1
    noise_a = None
    noise_m = 0.3

    X, z, theta0, a0, true_theta = generate_data(n, p, d, noise_theta, noise_a, noise_m=noise_m, plot=False, return_true=True)

    n_steps = 5  # fixme
    rho_0 = 0.01
    rho = 0.7
    sample_size = 200
    I_d = np.eye(d)
    n_takes = 3  # fixme

    rhos_theta = [rho_0 * (rho ** i) for i in range(n_steps)]
    rhos_a = [rho_0 * (rho ** i) for i in range(n_steps)]

    obj_vals_takes = []
    errs_takes = []
    for take in tqdm(range(n_takes)):
        traj = algorithm(n_steps, theta0, a0, X, z, rhos_theta, rhos_a, sample_size, seed=take * n_steps)
        obj_vals = [objective(X, theta[None, :], d, 0, a, z, I_d)[0].item() for theta, a in traj]
        errs = [np.linalg.norm(theta - true_theta) for theta, _ in traj]
        obj_vals_takes.append(obj_vals)
        errs_takes.append(errs)

    best_n_slices = None
    best_obj_val = 1e9
    for n_slices in range(10, 101, 10):
        # sir = SlicedInverseRegression(n_directions=1, n_slices=n_slices)
        sir = SlicedAverageVarianceEstimation(n_directions=1, n_slices=n_slices)
        # X_sir = sir.fit_transform(X.T, z)  # shape (n, 1)
        sir.fit(X.T, z)
        theta_sir = sir.directions_  # [0, :]
        sir_val = objective(X, theta_sir, d, 0, None, z, I_d)[0].item()
        if sir_val < best_obj_val:
            best_n_slices = n_slices
            best_obj_val = sir_val
        print(f"n_slices={n_slices}, sir_val={round(sir_val, 1)}")

    plot_results([rho], [sample_size], [rho_0], [obj_vals_takes], plot_many=True, suffix='noise', hline=best_obj_val, hline_name='SIR')
    plot_results([rho], [sample_size], [rho_0], [errs_takes], is_objective=False, plot_many=True, suffix='noise')

