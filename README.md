# Code and Data for our Paper: [Scalably learning quantum many-body Hamiltonians from dynamical data](https://www.arxiv.org/abs/???)
This repository serves as a reference point on how to use the code used in our paper.
It also contains instructions on how to inspect or reproduce our numerical results.

## The Differentiable TEBD
For our numerical analysis we developed a differentiable implementation of the TEBD.
This is contained in the Github repository [differentiable-tebd](https://www.github.com/frederikwilde/differentiable-tebd).

### Demo
This repository contains a brief [demo](https://github.com/frederikwilde/scalable-dynamical-hamiltonian-learning/tree/main/demo) on how to generate synthetic measurement data and then learn the Hamiltonian from it.

## Numerical Results
The numerical results presented in the paper are contained in the data repository (to appear soon).
To analyze the scaling of the error in the number of measurement samples we had to generate a large amount of synthetic data.
Therefore, the total size of the data repository is about 100GB.

## Citing
If you use the differentiable-tebd package or any of our numerical results, please cite our paper.
```
@article{wilde_scalably_2022,
	title = {Scalably learning quantum many-body {Hamiltonians} from dynamical data},
	journal = {arXiv:2203.15846},
	author = {},
	month = ,
	year = {2022},
	note = {arXiv: },
}
```
