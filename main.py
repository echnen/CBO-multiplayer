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
Run this file to reproduce all the experiments in Section 3 of:

E. Chenchene, H. Huang, J. Qui.
A consensus-based algorithm for non-convex multi-player games,
2023. DOI: XX.XXXXX/arXiv.XXXX.YYYYY.

"""

import pathlib
import experiments as expm
import warnings

test = False

if __name__ == "__main__":

    pathlib.Path("results").mkdir(parents=True, exist_ok=True)

    # print("------------------------------------------------------------------")
    # print("\n\n *** Generating Figure 1 ***\n")
    # expm.illustration_1()

    # print("------------------------------------------------------------------")
    # print("\n\n *** Generating Figure 2 ***")
    # print("Note: Warnings are disabled to account for expected overflow issues.\n")

    # warnings.filterwarnings('ignore')
    # expm.parameters_test_1(Small=test)
    # expm.parameters_test_2(Small=test)
    # warnings.filterwarnings("default")

    print("------------------------------------------------------------------")
    print("\n\n *** Generating Figure 3 ***\n")
    expm.test_ani_vs_iso(Small=test)

    print("------------------------------------------------------------------")
    print("\n\n *** Generating Figure 4 ***\n")
    expm.testing_dependence_on_d()
