# -*- coding: utf-8 -*-
#
#    Copyright (C) 2023 Enis Chenchene (enis.chenchene@uni-graz.at)
#                       Hui Huang (hui.huang@uni-graz.at)
#                       Jinniao Qiu (jinniao.qiu@ucalgary.ca)
#
#    This file is part of the example code repository for the paper:
#
#      E. Chenchene, H. Huang, J. Qui.
#      A consensus-based algorithm for non-convex multiplayer games,
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
This file contains an implementation of the experiments conducted
in Section 3 of:

E. Chenchene, H. Huang, J. Qui.
A consensus-based algorithm for non-convex multiplayer games,
2023. DOI: XX.XXXXX/arXiv.XXXX.YYYYY.

"""

import numpy as np
import structures as st
import cbo_one_dimensional as opt_one_d
import cbo as opt
import plots as show
from tqdm import trange


def illustration_1():
    '''
    This function generates Figure 1.
    '''
    # model parameters
    N = 40
    M = 4
    a, b, x_opt = st.init_pbl_one_dim(M)

    # algorithm parameters
    dt = 1e-4
    sig = 1e-1
    lam = (1e3 + sig ** 2) / 2
    alpha = 1e5

    # starting point
    mu = np.array([-2, 2, -1, 3])
    # print(mu)
    X0 = np.random.normal(x_opt + mu, 10, (N, M))

    # algorithm
    opt_one_d.cbo_mpg_one_dim(N, M, a, b, x_opt, X0, dt, sig, lam, alpha,
                              maxit=500, Verbose=True, Plot=True)


def parameters_test_1(Small=False):
    '''
    This function generates Figure 2(A).
    '''

    if Small:
        num_of_runs = 10
    else:
        num_of_runs = 500

    maxit = 500

    # model parameters
    N = 40
    M = 4
    a, b, x_opt = st.init_pbl_one_dim(M)

    # algorithm parameters
    dt = 1e-4
    sig = 1e-1

    # starting point
    mu = np.array([-2, 1, 0, 3])
    X0 = np.random.normal(x_opt + mu, 5, (N, M))

    # placeholders for plots
    Times = np.array([k * dt for k in range(maxit)])
    Vs_alpha = np.zeros((maxit, num_of_runs))
    Vs_lambda = np.zeros((maxit, num_of_runs))
    Vs_sample = np.zeros((maxit, num_of_runs))

    # testing alpha
    lam = (1e3 + sig ** 2) / 2
    Alphas = np.logspace(-6, 7, num_of_runs)

    for i in trange(num_of_runs, desc=f'{"Dependence on alpha":<25}'):

        alpha = Alphas[i]
        Vs = opt_one_d.cbo_mpg_one_dim(N, M, a, b, x_opt, X0, dt, sig, lam,
                                       alpha, maxit, Plot=False, Verbose=False)
        Vs_alpha[:, i] = Vs

    # testing lambda
    Ubs = np.logspace(2, 4, num_of_runs)
    alpha = 1e6

    for i in trange(num_of_runs, desc=f'{"Dependence on lambda":<25}'):

        lam = (Ubs[i] + sig ** 2) / 2
        Vs = opt_one_d.cbo_mpg_one_dim(N, M, a, b, x_opt, X0, dt, sig, lam,
                                       alpha, maxit, Plot=False, Verbose=False)
        Vs_lambda[:, i] = Vs

    # testing dependence on N
    lam = (1e3 + sig ** 2) / 2
    Ns = np.linspace(4, 4000, num_of_runs)

    for i in trange(num_of_runs, desc=f'{"Dependence on N":<25}'):

        N = int(Ns[i])
        X0 = np.random.normal(x_opt + mu, 5, (N, M))
        Vs = opt_one_d.cbo_mpg_one_dim(N, M, a, b, x_opt, X0, dt, sig, lam,
                                       alpha, maxit, Plot=False, Verbose=False)
        Vs_sample[:, i] = Vs

    show.plot_compare_choices_Nla(Times, Vs_alpha, Vs_lambda, Vs_sample)


def parameters_test_2(Small=False):
    '''
    This function generates Figure 2(B).
    '''

    if Small:
        num_of_runs = 10
    else:
        num_of_runs = 500

    maxit = 500

    # model parameters
    M = 4
    a, b, x_opt = st.init_pbl_one_dim(M)

    # algorithm parameters
    dt = 1e-5
    alpha = 1e6
    N = 100

    Lambdas = np.logspace(-1, 6, num_of_runs)
    Sigmas = np.logspace(-1, 2, num_of_runs)

    # starting point
    mu = np.array([-2, 1, 0, 3])
    X0 = np.random.normal(x_opt + mu, 5, (N, M))
    V_0 = np.sum(np.square(X0 - x_opt[np.newaxis, :])) / N

    # Auxiliary objects
    Vs_M = np.zeros((num_of_runs, num_of_runs))

    for i in trange(num_of_runs, desc=f'{"Generating image 2(B)":<25}'):
        for j in range(num_of_runs):

            lam = Lambdas[i]
            sig = Sigmas[j]

            Vs = opt_one_d.cbo_mpg_one_dim(N, M, a, b, x_opt, X0, dt, sig, lam,
                                           alpha, maxit)
            Vs_M[i, j] = min(max(np.log(Vs[-1]), -10), 6)

    show.plot_experiment_2(Lambdas, Sigmas, Vs_M, V_0)

    return Lambdas, Sigmas, Vs_M, V_0


def test_ani_vs_iso(Small=False):
    '''
    Generates Figure 3.
    '''

    if Small:
        num_of_runs = 2
    else:
        num_of_runs = 20

    # model parameters
    d = 5
    M = 4
    N = 10000
    a, b, c, L, phi, d_phi, x_opt = st.init_pbl(d, M)

    # algorithm parameters
    dt = 1e-3
    sig = 1
    alpha = 1e10
    maxit = 1000

    # placeholders
    Vs_i_s = np.zeros((maxit, num_of_runs))
    Res_i_s = np.zeros((maxit, num_of_runs))
    Vs_a_s = np.zeros((maxit, num_of_runs))
    Res_a_s = np.zeros((maxit, num_of_runs))

    Vs_i_l = np.zeros((maxit, num_of_runs))
    Res_i_l = np.zeros((maxit, num_of_runs))
    Vs_a_l = np.zeros((maxit, num_of_runs))
    Res_a_l = np.zeros((maxit, num_of_runs))

    lam_s = (1e1 + sig ** 2) / 2
    lam_l = (1e2 + sig ** 2) / 2

    pbar = trange(num_of_runs, leave=True)
    for i in pbar:

        # starting point
        mu = 2 * (np.random.rand(M) - 0.5)
        X0 = x_opt[:, np.newaxis, :] + np.random.normal(mu, 10, (d, N, M))

        pbar.set_description(f"{'Processing an. small (1/4)':<25}")
        vs_a_s, res_a_s = opt.cbo_mpg(dt, sig, lam_s, alpha, X0, d, N, M, a, b,
                                      c, L, x_opt, maxit, case='anisotropic')

        pbar.set_description(f"{'Processing is. small (2/4)':<25}")
        vs_i_s, res_i_s = opt.cbo_mpg(dt, sig, lam_s, alpha, X0, d, N, M, a, b,
                                      c, L, x_opt, maxit, case='isotropic')

        pbar.set_description(f"{'Processing an. large (3/4)':<25}")
        vs_a_l, res_a_l = opt.cbo_mpg(dt, sig, lam_l, alpha, X0, d, N, M, a, b,
                                      c, L, x_opt, maxit, case='anisotropic')

        pbar.set_description(f"{'Processing is. large (4/4)':<25}")
        vs_i_l, res_i_l = opt.cbo_mpg(dt, sig, lam_l, alpha, X0, d, N, M, a, b,
                                      c, L, x_opt, maxit, case='isotropic')

        Vs_i_s[:, i] = vs_i_s
        Res_i_s[:, i] = res_i_s
        Vs_a_s[:, i] = vs_a_s
        Res_a_s[:, i] = res_a_s

        Vs_i_l[:, i] = vs_i_l
        Res_i_l[:, i] = res_i_l
        Vs_a_l[:, i] = vs_a_l
        Res_a_l[:, i] = res_a_l

    Times = np.array([k * dt for k in range(maxit)])
    show.plot_iso_vs_ani(Times, Vs_i_s, Res_i_s, Vs_a_s, Res_a_s, Vs_i_l,
                         Res_i_l, Vs_a_l, Res_a_l)


def testing_dependence_on_d():
    '''
    This function generates Figure 4.
    '''

    # model parameters
    M = 4

    # fixed algorithm parameters
    dt = 1e-4
    sig = 1e-1
    lam = (1e4 + sig ** 2) / 2
    maxit = 15

    # variables algorithm parameters
    Ds = np.linspace(2, 20, 20)
    Ns = np.linspace(2, 500, 499)
    Alphas = np.logspace(1, 10, 100)

    # placeholders
    Out_dN = np.zeros((20, 499))
    Out_da = np.zeros((20, 100))

    # testing dimension vs sample size N
    alpha = 1e10

    pbar_dN = trange(20, desc=f'{"Generating image 4(A)":<10}', leave=True)
    for i in pbar_dN:

        # initialize problem on d dimensions
        d = int(Ds[i])
        a, b, c, L, phi, d_phi, x_opt = st.init_pbl(d, M)

        for (j, N) in enumerate(Ns):

            pbar_dN.set_description('Generating image 4(A)'
                                    + f"{' ({}/{})'.format(j, 499) :>10}")
            N = int(N)
            mu = 1e-1 * (np.random.rand(M) - 0.5)
            X0 = x_opt[:, np.newaxis, :] + np.random.normal(mu, 3, (d, N, M))

            Vs, Res = opt.cbo_mpg(dt, sig, lam, alpha, X0, d, N, M, a, b, c, L,
                                  x_opt, maxit, case='anisotropic')

            Out_dN[i, j] = max(min(np.log(Vs[-1]), 6), -10)

    # testing dimension vs alpha
    N = 1000
    Ds = np.linspace(2, 20, 20)

    pbar_da = trange(20, desc=f'{"Generating image 4(B)":<10}', leave=True)
    for i in pbar_da:

        # initialize problem on d dimensions
        d = int(Ds[i])
        a, b, c, L, phi, d_phi, x_opt = st.init_pbl(d, M)

        mu = 1e-1 * (np.random.rand(M) - 0.5)
        X0 = x_opt[:, np.newaxis, :] + np.random.normal(mu, 3, (d, N, M))

        for (j, alpha) in enumerate(Alphas):

            pbar_da.set_description('Generating image 4(B)'
                                    + f"{' ({}/{})'.format(j, 100) :>10}")
            Vs, Res = opt.cbo_mpg(dt, sig, lam, alpha, X0, d, N, M, a, b, c, L,
                                  x_opt, maxit, case='anisotropic')

            Out_da[i, j] = max(min(np.log(Vs[-1]), 6), -10)

    show.plot_dependence_on_d(Ds, Ns, Alphas, Out_dN, Out_da)
