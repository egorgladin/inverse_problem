import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import decomposition


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
        else get_a_tilde(Psi, np.zeros((d, d)), z, a, 0)  # shape (sample_size, d)
    estimate = get_estimate(Psi, a_tilde)  # shape (sample_size, n)
    ls = 0.5 * ((estimate - z) ** 2).sum(axis=1)  # shape (sample_size,)
    return ls, a_tilde


def algorithm(n_steps, theta, a, X, z, rhos_theta, rhos_a, sample_size, aux_func=None):
    # X has shape (p, n)
    p = len(theta)
    d = len(a)
    I_d = np.eye(d)
    pca = decomposition.PCA(n_components=1)

    traj = [(theta, a)]
    for k in tqdm(range(n_steps)):
        rho_theta, rho_a = rhos_theta[k], rhos_a[k]
        np.random.seed(100 + k)
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


def generate_data(n, p, d, noise_theta, noise_a, plot=True, return_true=False):
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
    z = np.squeeze(z, axis=0)

    if plot:
        grid = np.linspace(theta_X.min() - 0.1, theta_X.max() + 0.1, num=500)
        basis = cosine_basis(grid[None, :], d)
        f = get_estimate(basis, true_a[None, :])  # shape (1, n)

        plt.figure(figsize=(12, 5))
        plt.plot(grid, np.squeeze(f, axis=0), label=r"$m(u)=\Psi(u)^\top a^*$")
        plt.scatter(np.squeeze(theta_X, axis=0), z, c='k', label=r"$X^\top \theta^*$")
        plt.legend()
        plt.savefig("plots/target.png", bbox_inches='tight')

    np.random.seed(3)
    theta0 = true_theta + noise_theta * np.random.randn(p)
    theta0 /= np.linalg.norm(theta0)

    np.random.seed(4)
    a0 = np.zeros_like(true_a)  # true_a + noise_a * np.random.randn(d)

    return (X, z, theta0, a0, true_theta) if return_true else (X, z, theta0, a0)


def plot_results(params_x, params_y, params_inner, results):
    n_rows, n_cols = len(params_y), len(params_x)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 12))

    for j, par_x in enumerate(params_x):
        for i, par_y in enumerate(params_y):
            ax = axs[i, j]
            for par_in in params_inner:
                obj_values = results.pop(0)
                ax.plot(obj_values, label=r'$\rho_0=$' + f'{par_in}')
                ax.set_title(r'$\rho=$' + f'{par_x}; {par_y} samples')
            ax.legend()
            ax.set_yscale('log')

    for ax in axs.flat:
        ax.set(xlabel='iterations', ylabel=r'objective $l(x)$')
    plt.tight_layout()
    plt.savefig(f"plots/exper.png", bbox_inches='tight')


if __name__ == '__main__':
    n = 200
    p = 10
    d = 10
    noise_theta = 0.1
    noise_a = 0.1

    X, z, theta0, a0 = generate_data(n, p, d, noise_theta, noise_a, plot=False)

    n_steps = 100
    rhos_0 = [0.1, 0.5, 1.]
    rhos = [0.7, 0.9, 0.95]
    sample_sizes = [100, 400, 1000]
    I_d = np.eye(d)

    results = []
    for rho in rhos:
        for sample_size in sample_sizes:
            for rho_0 in rhos_0:
                rhos_theta = [rho_0 * (rho ** i) for i in range(n_steps)]
                rhos_a = [rho_0 * (rho ** i) for i in range(n_steps)]

                traj = algorithm(n_steps, theta0, a0, X, z, rhos_theta, rhos_a, sample_size)
                obj_vals = [objective(X, theta[None, :], d, 0, a, z, I_d)[0] for theta, a in traj]
                results.append(obj_vals)

    plot_results(rhos, sample_sizes, rhos_0, results)
