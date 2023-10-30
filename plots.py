# -*- coding: utf-8 -*-
#
#    Copyright (C) 2023 Enis Chenchene (enis.chenchene@uni-graz.at)
#                       Hui Huang (hui.huang@uni-graz.at)
#                       Jinniao Qiu (jinniao.qiu@ucalgary.ca)
#
#    This file is part of the example code repository for the paper:
#
#      E. Chenchene, H. Huang, J. Qui.
#      A consensus-based algorithm for non-convex multi-player games,
#      2023. DOI: XX.XXXXX/arXiv.XXXX.YYYYY.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
This file contains several useful functions to plot the results of all the
experiments in Section 3 of:

E. Chenchene, H. Huang, J. Qui.
A consensus-based algorithm for non-convex multi-player games,
2023. DOI: XX.XXXXX/arXiv.XXXX.YYYYY.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import structures as st

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})
rc('text', usetex=True)


def plot_compare_choices_Nla(Times, Vs_alpha, Vs_lambda, Vs_sample):

    maxit, num_of_runs = np.shape(Vs_alpha)

    _fig, axs = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    ecm = mpl.colors.LinearSegmentedColormap.from_list('ecm', ['black', 'red'])
    sm = plt.cm.ScalarMappable(cmap=ecm, norm=plt.Normalize(vmin=0, vmax=1))

    for i in range(num_of_runs):
        axs[0].semilogy(Times, Vs_alpha[:, i],
                        color=ecm((i + 1) / num_of_runs), alpha=0.1)
        axs[1].semilogy(Times, Vs_lambda[:, i],
                        color=ecm((i + 1) / num_of_runs), alpha=0.1)
        axs[2].semilogy(Times, Vs_sample[:, i],
                        color=ecm((i + 1) / num_of_runs), alpha=0.1)

    axs[0].grid(which='both')
    axs[0].set_xlabel(r"Time $(t)$")
    axs[0].set_ylabel("$V(t)$")

    axs[1].grid(which='both')
    axs[1].set_xlabel(r"Time $(t)$")

    axs[2].grid(which='both')
    axs[2].set_xlabel(r"Time $(t)$")

    cbar1 = plt.colorbar(sm, ax=axs[0], orientation="horizontal", pad=0.2,
                         ticks=np.linspace(0, 1, 5))
    cbar1.ax.set_xticklabels(['-6', '-2.75', '0.5', '3.75', '7'])

    cbar2 = plt.colorbar(sm, ax=axs[1], orientation="horizontal", pad=0.2,
                         ticks=np.linspace(0, 1, 5))
    cbar2.ax.set_xticklabels(['1.7', '2.2', '2.7', '3.2', '3.7'])

    cbar3 = plt.colorbar(sm, ax=axs[2], orientation="horizontal", pad=0.2,
                         ticks=np.linspace(0, 1, 5))
    cbar3.ax.set_xticklabels(['4', '100', '200', '300', '400'])

    plt.savefig('results/dependence_on_alpha_lambda_N.pdf',
                bbox_inches='tight')
    plt.show()


def plot_experiment_2(Lambdas, Sigmas, Vs_M, V_0):

    fig1, ax2 = plt.subplots(layout='constrained', figsize=(7, 5))

    CS = ax2.contourf(Lambdas, Sigmas, Vs_M.T, levels=np.linspace(-10, 6, 17))
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Drift parameter $(\lambda)$')
    ax2.set_ylabel('Diffusion parameter $(\sigma)$')

    ax2.contour(CS, levels=[np.log(V_0)], colors='k')

    plt.plot(Sigmas ** 2/2, Sigmas, color='r')
    plt.xlim(Lambdas[0], Lambdas[-1])
    plt.ylim(Sigmas[0], Sigmas[-1])
    plt.text(14, 10, '$2 \lambda - \sigma^2=0$', color='red', rotation=47)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=-10,
                                                                  vmax=6))
    plt.colorbar(sm, ax=ax2)
    plt.savefig('results/dependence_sigma_and_lambda_2.pdf',
                bbox_inches='tight')
    plt.show()

    return


def plot_illustration(Xs, Xs_alpha, x_opt, a, b, size_box):

    num_snapshots, N, M = np.shape(Xs)
    times = [0, 0.001, 0.007, 0.5]

    _fig, axs = plt.subplots(num_snapshots, M, figsize=(int(4 * M) + 1, 10),
                             sharex=True, sharey=True)

    for k in range(num_snapshots):

        X = Xs[k, :, :]
        x_alpha = Xs_alpha[k, :]

        for m in range(M):

            space_range = np.linspace(x_opt[m] - size_box, x_opt[m] + size_box,
                                      100000)

            # vector of complements
            x_alpha_min_m = np.delete(x_alpha, m)
            x_opt_min_m = np.delete(x_opt, m)

            E_m_alpha = st.E_one_dim(m, space_range, x_alpha_min_m, a, b,
                                     x_opt)
            E_m_opt = st.E_one_dim(m, space_range, x_opt_min_m, a, b, x_opt)
            E_m_meas = st.E_one_dim(m, X[:, m], x_alpha_min_m, a, b, x_opt)

            E_x_alpha = st.E_one_dim(m, x_alpha[m], x_alpha_min_m, a, b, x_opt)
            E_x_opt = st.E_one_dim(m, x_opt[m], x_opt_min_m, a, b, x_opt)

            axs[k, m].plot(space_range, E_m_alpha, c="b", label="alpha")
            axs[k, m].plot(space_range, E_m_opt, c="k", label="opt")
            axs[k, m].plot(X[:, m], E_m_meas, marker='o', markersize=5,
                           color='b', linestyle='None')
            axs[k, m].plot(x_alpha[m], E_x_alpha, markersize=10, marker='*',
                           color='k')
            axs[k, m].plot(x_alpha[m], E_x_alpha, marker='*', color='b')
            axs[k, m].plot(x_opt[m], E_x_opt, markersize=10, marker='*',
                           color='k')
            axs[k, m].plot(x_opt[m], E_x_opt, marker='*', color='r')

            axs[k, m].set_xlim(space_range[0], space_range[-1])
            axs[k, m].set_ylim(min(E_m_alpha) - 1e1, max(E_m_alpha) + 1e1)

            if m == 0:
                axs[k, m].set_ylabel(f'$t={times[k]}$')

            if k == 3:
                axs[k, m].set_xlabel(f'player $m={int(m+1)}$')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("results/illustration_1d.pdf", bbox_inches='tight')
    plt.show()

    return


def plot_iso_vs_ani(Times, Vs_i_s, Res_i_s, Vs_a_s, Res_a_s, Vs_i_l, Res_i_l,
                    Vs_a_l, Res_a_l):

    maxit = len(Times)
    mev = maxit // 10

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 25})

    # plotting results with lambda small
    _fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharey=True, sharex=True)

    axs[0, 0].semilogy(Times, np.mean(Vs_i_s, axis=1), '-*', color='red',
                       linewidth=2, markersize=6, markevery=mev)
    axs[0, 0].fill_between(Times, Vs_i_s.min(1),
                           Vs_i_s.max(1), color='red', alpha=0.3)
    axs[0, 0].set_ylabel('$V(t)$')
    axs[0, 0].title.set_text('Isotropic')
    axs[0, 0].grid()

    axs[0, 1].semilogy(Times, np.mean(Vs_a_s, axis=1), '-*', color='red',
                       linewidth=2, markersize=6, markevery=mev)
    axs[0, 1].fill_between(Times, Vs_a_s.min(1),
                           Vs_a_s.max(1), color='red', alpha=0.3)
    axs[0, 1].title.set_text('Anisotropic')
    axs[0, 1].grid()

    axs[1, 0].semilogy(Times, np.mean(Res_i_s, axis=1), '-*', color='blue',
                       linewidth=2, markersize=6, markevery=mev)
    axs[1, 0].fill_between(Times, Res_i_s.min(1),
                           Res_i_s.max(1), color='blue', alpha=0.3)
    axs[1, 0].set_ylabel('Residual')
    axs[1, 0].set_xlabel('Time $(t)$')
    axs[1, 0].grid()

    axs[1, 1].semilogy(Times, np.mean(Res_a_s, axis=1), '-*', color='blue',
                       linewidth=2, markersize=6, markevery=mev)
    axs[1, 1].fill_between(Times, Res_a_s.min(1),
                           Res_a_s.max(1), color='blue', alpha=0.3)
    axs[1, 1].set_xlabel('Time $(t)$')
    axs[1, 1].grid()

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('results/iso_vs_aniso_small.pdf', bbox_inches='tight')
    plt.show()

    # plotting results with lambda large
    _fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharey=True, sharex=True)

    axs[0, 0].semilogy(Times, np.mean(Vs_i_l, axis=1), '-*', color='red',
                       linewidth=2, markersize=6, markevery=mev)
    axs[0, 0].fill_between(Times, Vs_i_l.min(1),
                           Vs_i_l.max(1), color='red', alpha=0.3)
    axs[0, 0].set_ylabel('$V(t)$')
    axs[0, 0].title.set_text('Isotropic')
    axs[0, 0].grid()

    axs[0, 1].semilogy(Times, np.mean(Vs_a_l, axis=1), '-*', color='red',
                       linewidth=2, markersize=6, markevery=mev)
    axs[0, 1].fill_between(Times, Vs_a_l.min(1),
                           Vs_a_l.max(1), color='red', alpha=0.3)
    axs[0, 1].title.set_text('Anisotropic')
    axs[0, 1].grid()

    axs[1, 0].semilogy(Times, np.mean(Res_i_l, axis=1), '-*', color='blue',
                       linewidth=2, markersize=6, markevery=mev)
    axs[1, 0].fill_between(Times, Res_i_l.min(1),
                           Res_i_l.max(1), color='blue', alpha=0.3)
    axs[1, 0].set_ylabel('Residual')
    axs[1, 0].set_xlabel('Time $(t)$')
    axs[1, 0].grid()

    axs[1, 1].semilogy(Times, np.mean(Res_a_l, axis=1), '-*', color='blue',
                       linewidth=2, markersize=6, markevery=mev)
    axs[1, 1].fill_between(Times, Res_a_l.min(1),
                           Res_a_l.max(1), color='blue', alpha=0.3)
    axs[1, 1].set_xlabel('Time $(t)$')
    axs[1, 1].grid()

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('results/iso_vs_aniso_large.pdf', bbox_inches='tight')
    plt.show()


def plot_dependence_on_d(Ds, Ns, Alphas, Out_dN, Out_da):

    plt.figure(figsize=(6, 5))

    plt.contourf(Ds, Ns, Out_dN.T, levels=np.linspace(-10, 6, 9))
    plt.xlabel('Dimension $(d)$')
    plt.ylabel('Sample size $(N)$')
    plt.colorbar()
    plt.savefig('results/d_vs_N.pdf', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6, 5))

    plt.contourf(Ds, Alphas, Out_da.T, levels=np.linspace(-10, 6, 9))
    plt.xlabel('Dimension $(d)$')
    plt.ylabel('Parameter $a$')
    plt.yscale('log')
    plt.colorbar()
    plt.savefig('results/d_vs_alpha.pdf', bbox_inches='tight')

    plt.show()
