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
This file contains several useful functions to reproduce the results of all the
experiments in Section 3 of:

E. Chenchene, H. Huang, J. Qui.
A consensus-based algorithm for non-convex multiplayer games,
2023. DOI: XX.XXXXX/arXiv.XXXX.YYYYY.

"""

import numpy as np


# functions for Section 3.1

def init_pbl_one_dim(M):

    np.random.seed(2)
    a = np.random.rand(M) + M
    b = np.random.rand(M)

    # computing optimal solution
    R = -np.ones((M, M))
    np.fill_diagonal(R, a)
    x_opt = np.linalg.solve(R, b)

    return a, b, x_opt


def nncvx(z):

    return 10 * (1 - np.cos(10 * z)) + z ** 2


def E_one_dim(i, x_i, x_min_i, a, b, x_opt):

    return 0.5 * (a[i] * x_i - np.sum(x_min_i) - b[i]) ** 2 \
        + nncvx(x_i - x_opt[i])


# functions for Section 3.2

def E_ind(x_in, xi, a, b, c_in, L):

    return np.sum(x_in * (c_in - phi(L @ (xi + x_in), a, b)), axis=0)


def phi(z, a, b):

    return np.maximum(a - b * z, 0) ** 2


def d_phi(z, a, b):

    return - b * 2 * np.maximum(a - b * z, 0)


def compute_residual(x, L, a, b, c):

    res = c - phi(L @ np.sum(x, axis=1), a, b)[:, np.newaxis] \
            - L.T @ (d_phi(L @ np.sum(x, axis=1), a, b)[:, np.newaxis] * x)

    return np.sum(np.square(res))


def init_pbl(d, M):

    c = -np.random.rand(d, M)
    a = 1e2
    b = 1e-3
    L = 3 * np.eye(d) + np.ones((d, d))

    num_tries = 0

    while ((not np.all(c > 0)) and (num_tries <= 100)):

        x_opt = 10 * np.random.rand(d, M)

        c = phi(L @ np.sum(x_opt, axis=1), a, b)[:, np.newaxis] \
            + L.T @ (d_phi(L @ np.sum(x_opt, axis=1), a, b)[:, np.newaxis]
                     * x_opt)

        if num_tries > 99:
            print('Warning! Initialization failed. Try again.' +
                  ' If it keeps failing, try decreasing b or increasing a')
            return None

    return a, b, c, L, phi, d_phi, x_opt
