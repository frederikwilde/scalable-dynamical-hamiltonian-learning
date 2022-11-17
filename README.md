# Code and Data for our Paper [Scalably learning quantum many-body Hamiltonians from dynamical data](https://arxiv.org/abs/2209.14328)
This repository serves as a reference point on how to use the code used in our paper.
It also contains instructions on how to inspect or reproduce our numerical results.

## The Differentiable TEBD Package
For our numerical analysis we developed a differentiable implementation of the TEBD which is contained in the standalone Python package [differentiable-tebd](https://www.github.com/frederikwilde/differentiable-tebd).

### Demo
This repository contains a brief [demo](https://github.com/frederikwilde/scalable-dynamical-hamiltonian-learning/tree/main/demo) on how to generate synthetic measurement data and then learn the Hamiltonian from it.

## Numerical Results
The numerical results presented in the paper are contained in the [data repository](https://doi.org/10.5281/zenodo.7299942).
To analyze the scaling of the error in the number of measurement samples we had to generate a large amount of synthetic data.
Therefore, the total size of the data repository is about 80GB (37GB compressed).

## Citing
If you use the differentiable-tebd package or any of our numerical results, please cite our paper.
```
@misc{wilde_scalably_2022,
  doi = {10.48550/ARXIV.2209.14328},
  url = {https://arxiv.org/abs/2209.14328},
  author = {Wilde, Frederik and Kshetrimayum, Augustine and Roth, Ingo and Hangleiter, Dominik and Sweke, Ryan and Eisert, Jens},
  title = {Scalably learning quantum many-body Hamiltonians from dynamical data},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
