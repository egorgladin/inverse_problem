import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import pickle
import imageio.v2 as imageio


def plot_results(params_x, params_y, params_inner, results, is_objective=True, plot_many=False,
                 alpha=0.3, suffix='', hline=None, hline_name=None):
    n_rows, n_cols = len(params_y), len(params_x)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)

    basic_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    basic_colors = basic_colors[:len(params_inner)]

    names = []
    dims = int(len(params_x) > 1) + int(len(params_y) > 1)
    for j, par_x in enumerate(params_x):
        for i, par_y in enumerate(params_y):
            # ax = axs[i, j] if (len(params_x) > 1 and len(params_y) > 1) else axs[i + j]
            ax = axs[i, j] if dims == 2 else (axs[i + j] if dims == 1 else axs)
            for color, par_in in zip(basic_colors, params_inner):
                obj_values = results.pop(0)
                lbl = r'$\rho_0=$' + f'{par_in}'
                if plot_many:
                    for take, o_v in enumerate(obj_values):
                        names.append(lbl)
                        ax.plot(o_v, label=('_' + lbl) if take else lbl, alpha=alpha, c=color)
                else:
                    ax.plot(obj_values, label=lbl)

                if hline is not None:
                    n_iter = len(obj_values[0]) if plot_many else len(obj_values)
                    ax.hlines(y=hline, xmin=0, xmax=n_iter, label=hline_name)  # , linewidth=2, color='r'
                ax.set_title(r'$\rho=$' + f'{par_x}; {par_y} samples')

            ax.legend()
            ax.set_yscale('log')
            ax.tick_params(axis='y', which='minor')
            ax.grid(True, which="both")
            # ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

    for ax in (axs.flat if dims > 1 else [axs]):
        ylabel = r'objective $\ell(x)$' if is_objective else r'$||\theta_k-\theta^*||$'
        ax.set(xlabel='iterations', ylabel=ylabel)


    plt.tight_layout()
    file_name = f"plots/" + ("obj" if is_objective else "err") + "_exper" + suffix + ".png"
    plt.savefig(file_name, bbox_inches='tight')


def drawer(mean, sample, weights, k, theta_true, target_func):
    with open('ax_base.pkl', 'rb') as fl:
        axes = pickle.load(fl)
    ax = axes[0]
    fig = ax.figure

    ax.plot(mean[0], mean[1], 'kD', markersize=5)
    ax.annotate(fr'$\theta_{k}$', xy=mean, xytext=(mean[0] + 0.02, mean[1] - 0.04), color='k', fontsize='x-large')

    err = np.linalg.norm(mean - theta_true)
    obj_val = target_func(mean)
    ax.plot([], [], ' ', label=fr'$||\theta_{k}-\theta^*|| = {round(err, 4)}, \ell(\theta_{k}) = {round(obj_val, 4)}$')
    ax.legend(loc='lower left')

    fig.savefig(f'plots/visualize_2d/{k}_0.png', bbox_inches='tight')

    sc = ax.scatter(sample[:, 0], sample[:, 1], c=weights, s=12, vmin=0., vmax=1., cmap='cool', zorder=2)
    fig.colorbar(sc, cax=axes[1], ticks=[0.2 * i for i in range(6)])

    fig.savefig(f'plots/visualize_2d/{k}_1.png', bbox_inches='tight')
    plt.close(fig)


def math_axes(ax, bounds):
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim(*bounds)
    ax.set_ylim(*bounds)


def grid_with_obj_values(bounds, objective):  # X, d, z, I_d):
    grid_size = 100
    x = np.linspace(*bounds, grid_size)
    y = np.linspace(*bounds, grid_size)
    x1, x2 = np.meshgrid(x, y)
    thetas = np.stack((x1, x2), axis=2)  # shape (50, 50, 2)

    vals = np.empty((grid_size, grid_size))
    for i in range(grid_size):
        theta = thetas[i, :, :]
        vals[i] = objective(theta)
        # vals[i] = objective(X, theta, d, 0, None, z, I_d)[0]

    return vals


def create_base_figure(objective, true_theta, rho_0, rho, sample_size):  # X, d, z, I_d
    # Initialize figure
    fig_base, axes_base = plt.subplots(1, 3, gridspec_kw={'width_ratios': [5, 1, 1]}, figsize=(10, 8))
    ax_base = axes_base[0]
    bounds = (0, 1.1)
    math_axes(ax_base, bounds)

    # Create levels
    vals = grid_with_obj_values(bounds, objective)  # X, d, z, I_d)

    pos = ax_base.imshow(vals, interpolation='bilinear', origin='lower',
                    cmap=cm.inferno, alpha=0.6, extent=(*bounds, *bounds))
    fig_base.colorbar(pos, fraction=0.046, pad=0.04, cax=axes_base[2])

    ax_base.plot(true_theta[0], true_theta[1], 'r*', markersize=11, label=r'$\theta^*$')
    ax_base.legend()

    circle = plt.Circle((0, 0), 1, color='k', fill=False)  # , clip_on=False
    ax_base.add_patch(circle)

    axes_base[1].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    axes_base[1].yaxis.tick_right()
    axes_base[1].set_yticks([0.2 * i for i in range(6)])
    axes_base[0].set_title(fr'Init. variance $\rho_0 = {rho_0}$, var. decay $\rho = {rho}$, sample size $M={sample_size}$')

    with open('ax_base.pkl', 'wb') as fs:
        pickle.dump(axes_base, fs)
    plt.close(fig_base)


def build_gif(noise_theta, n_steps):
    # build gif
    gif_name = f'{noise_theta}.gif'
    filenames = []
    for k in range(n_steps):
        filenames.append(f'plots/visualize_2d/{k}_0.png')
        filenames.append(f'plots/visualize_2d/{k}_1.png')

    with imageio.get_writer(gif_name, mode='I', duration='2') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

