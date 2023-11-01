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
This file contains an implementation of Algorithm 1 for d=1 detailed in
Section 3 of:

E. Chenchene, H. Huang, J. Qui.
A consensus-based algorithm for non-convex multiplayer games,
2023. DOI: XX.XXXXX/arXiv.XXXX.YYYYY.

"""

import numpy as np
import plots as show
import structures as st


def compute_consensus_one_dim(alpha, X, N, M, a, b, x_opt):

    Mn = np.mean(X, axis=0)
    E = 0.5 * (a[np.newaxis, :] * X - np.sum(Mn) + Mn[np.newaxis, :]
               - b[np.newaxis, :]) ** 2 + st.nncvx(X - x_opt[np.newaxis, :])
    W = np.exp(- alpha * (E - np.min(E, axis=0)[np.newaxis, :]))

    return np.sum(W * X, axis=0) / np.sum(W, axis=0)


def cbo_mpg_one_dim(N, M, a, b, x_opt, X0, dt, sig, lam, alpha, maxit=200000,
                    Plot=False, Verbose=False):

    sdt = np.sqrt(dt)

    # initialization
    X = np.copy(X0)

    # placeholder
    Vs = np.zeros(maxit)
    k = 0

    # placeholders for plots
    if Plot:
        Xs = np.zeros((4, N, M))
        Xs_alpha = np.zeros((4, M))
        snap_list = {0: 0, 10: 1, 70: 2, maxit - 1: 3}

    while k < maxit:

        # compute current consensus point
        x_alpha = compute_consensus_one_dim(alpha, X, N, M, a, b, x_opt)

        # variance in the sense of the paper
        V_k = np.sum(np.square(X - x_opt[np.newaxis, :])) / N
        Vs[k] = V_k

        if Plot:

            if k in snap_list:
                ind = snap_list[k]
                Xs[ind, :, :] = np.copy(X)
                Xs_alpha[ind, :] = np.copy(x_alpha)

        if Verbose:

            if k % (maxit // 10) == 0:
                print(f'|| Iteration: {k :> 6} |  Variance: {V_k :> 25} ||')

        # Brownian motion for exploration term
        dB = np.random.normal(0, sdt, (N, M))

        # particle update step (according to SDE)
        X = X - lam * (X - x_alpha[np.newaxis, :]) * dt + \
            sig * np.abs(X - x_alpha[np.newaxis, :]) * dB

        k += 1

    if Plot:
        show.plot_illustration(Xs, Xs_alpha, x_opt, a, b, 2.5)

    return Vs
