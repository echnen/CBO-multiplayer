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
This file contains an implementation of Algorithm 1 for d > 1 detailed in
Section 3 of:

E. Chenchene, H. Huang, J. Qui.
A consensus-based algorithm for non-convex multi-player games,
2023. DOI: XX.XXXXX/arXiv.XXXX.YYYYY.

"""

import numpy as np
import plots as show
import structures as st


def compute_consensus(alpha, X, d, N, M, L, a, b, c):

    x_alpha = np.zeros((d, M))
    Mn = np.mean(X, axis=1)

    for i in range(M):

        Mn_hat = np.delete(Mn, obj=i, axis=1)
        x_i = X[:, :, i]

        E_i = np.sum(x_i * (c[:, i][:, np.newaxis]
                            - st.phi(L @ (x_i + np.sum(Mn_hat,
                                                       axis=1)[:, np.newaxis]),
                                     a, b)), axis=0)

        E_i_min = min(E_i)
        W_i = np.exp(- alpha * (E_i - E_i_min))
        tot_mass = np.sum(W_i)
        x_alpha[:, i] = np.sum(W_i[np.newaxis, :] * x_i, axis=1) / tot_mass

    return x_alpha


def cbo_mpg(dt, sig, lam, alpha, X0, d, N, M, a, b, c, L, x_opt, maxit,
            case, Plot=False, Verbose=False):

    # setting parameters
    sdt = np.sqrt(dt)

    # initialization
    X = np.copy(X0)

    # useful objects
    Vs = np.zeros(maxit)
    Res = np.zeros(maxit)

    k = 0

    while k < maxit:

        # compute current consensus point
        x_alpha = compute_consensus(alpha, X, d, N, M, L, a, b, c)

        # compute variance in the sense of the paper
        V_k = np.sum(np.square(X - x_opt[:, np.newaxis, :])) / N
        Vs[k] = V_k

        # compute residual in the sense of (3.7)
        res = st.compute_residual(x_alpha, L, a, b, c)
        Res[k] = res

        if k % (maxit // 10) == 0:

            if Plot:
                x_mean = np.mean(X, axis=1)
                show.show_point_cloud_v2(X, x_mean, x_alpha, x_opt, a, b, c, L,
                                         size_box=5)

            if Verbose:
                print(f'Residual: {res}')
                print(f'Variance: {V_k}')

        # Brownian motion for exploration term
        dB = np.random.normal(0, sdt, (d, N, M))

        # particle update step (according to SDE)
        if case == 'anisotropic':
            X = X - lam * (X - x_alpha[:, np.newaxis, :]) * dt + \
                sig * np.abs(X - x_alpha[:, np.newaxis, :]) * dB

        if case == 'isotropic':
            X = X - lam * (X - x_alpha[:, np.newaxis, :]) * dt + \
                sig * (np.linalg.norm(X - x_alpha[:, np.newaxis, :],
                                      axis=0)[np.newaxis, :, :]) * dB

        k += 1

    return Vs, Res
