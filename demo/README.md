# Differentiable TEBD Tutorial
This demo shows how to use the `differentiable-tebd` Python package to generate synthetic time evolution data from some Hamiltonian.
In the second step the Hamiltonian is learned from that data.
The code contains comments and doc strings to explain each step.

### Setup
1) Clone the [differentiable-tebd package](https://github.com/frederikwilde/differentiable-tebd) and checkout version 0.2.4 of the package `git checkout 0.2.4`.
2) Create a virtual enviroment in the this directory and activate it, e.g. via `python3 -m venv venv` and then `source ./venv/bin/activate`.
3) Install dependences from the requirements file via `pip install -r requirements.txt`.
4) Install the differentiable-tebd package in the environment via `pip install -e /path/to/differentiable-tebd/`.

### Generating data
`python generate_dataset.py`

### Learn the Hamiltonian
`python main.py` outputs a `results.hdf5` file and a plot `results.pdf`.

### Plots
You can do your own plots in the `plots.ipynb` Jupyter Notebook.
Simply start a notebook server via `jupyter notebook` in the shell and play around.
