# A consensus-based algorithm for non-convex multiplayer games: Example code

This repository contains the experimental source code to reproduce the numerical experiments in:

* E. Chenchene, H. Huang, J. Qui. A consensus-based algorithm for non-convex multiplayer games. 2023. [ArXiv preprint](https://arxiv.org/abs/XXXX.YYYYY)

To reproduce the results of the numerical experiments in Section 3, run:
```bash
python3 main.py
```

If you find this code useful, please cite the above-mentioned paper:
```BibTeX
@article{chq2023,
  author = {Chenchene, Enis and Huang, Hui and Qui, Jinniao},
  title = {A consensus-based algorithm for non-convex multiplayer games},
  pages = {XXXX.YYYYY},
  journal = {ArXiv},
  year = {2023}
}
```

## Requirements

Please make sure to have the following Python modules installed, most of which should be standard.

* [numpy>=1.20.1](https://pypi.org/project/numpy/)
* [matplotlib>=3.3.4](https://pypi.org/project/matplotlib/)
* [tqdm>=4.66.1](https://pypi.org/project/tqdm/)

## Acknowledgments  

* | ![](<euflag.png>) | E.C. has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement no. 861137. |
  |-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
* The Department of Mathematics and Scientific Computing at the University of Graz, with which E.C. and H.H. are affiliated, is a member of NAWI Graz (https://nawigraz.at/en). J.Q. is partially supported by the National Science and Engineering Research Council of Canada (NSERC) and by the start-up funds from the University of Calgary.
  
## License  
This project is licensed under the GPLv3 license - see [LICENSE](LICENSE) for details.
