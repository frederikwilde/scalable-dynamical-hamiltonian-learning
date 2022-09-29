import h5py
import numpy as np
from time import time
from tqdm import tqdm

from differentiable_tebd.physical_models.heisenberg_disordered import hamiltonian
from differentiable_tebd.state_vector_simulation import vec_simulation
from differentiable_tebd.sampling import draw_samples

with h5py.File('dataset.hdf5', 'w') as data_file:
    # a dictionary for all meta data
    md = data_file.attrs
    md['rng_seed'] = 42
    rng = np.random.default_rng(md['rng_seed'])
    md['num_sites'] = 10
    md['true_params'] = 2 * rng.random(md['num_sites'] + 3) - 1.  # set random parameters for the Hamiltonian.
    md['time_stamps'] = np.arange(.2, 1.2, .2)
    md['num_bases'] = 100  # number of Pauli bases
    md['num_bitstrings_per_basis'] = 100  # number of measurements per Pauli basis

    # time evolution
    t_start = time()
    h = hamiltonian(md['true_params'], md['num_sites'])
    states = vec_simulation(h, md['time_stamps'], rtol=1e-13, atol=1e-13).T
    data_file.create_dataset('states', data=states)
    print(f'Time evolution finished in {time()-t_start:.1f} seconds.')
    print('log(1 - state-vector-norm) = %s' % ' '.join(['%.1f' % np.log10(1 - np.linalg.norm(v)) for v in states]))

    # create groups in the HDF5 file to store the bitstrings and bases
    group = data_file.create_group('bitstrings_list')
    group.attrs['description'] = ' '.join([
        'This group contains datasets t0, t1, ... corresponding to the times (see meta data).',
        'Each bitstring in such a dataset was measured in the Pauli basis given in the elements of the bases_list group.'])
    group = data_file.create_group('bases_list')
    group.attrs['description'] = 'This group contains datasets t0, t1, ... corresponding to the times (see meta data).'

    # sample from the states
    for i, state in enumerate(states):
        print(f"Sampling state at t={md['time_stamps'][i]:.1f}")
        bases = np.zeros((md['num_bases'], md['num_sites']), dtype=int)
        bitstrings = np.zeros((md['num_bases'], md['num_bitstrings_per_basis'], md['num_sites']), dtype=int)
        for j in tqdm(range(md['num_bases'])):
            pauli_basis = rng.integers(1, 4, size=md['num_sites'])
            strings = draw_samples(
                state,
                pauli_basis,
                md['num_bitstrings_per_basis'],
                rng=rng,
                sequential_samples=False,  # for n=24
                # basis_transforms_dir='basis_transforms',  # not faster for 18 sites!
            )
            bases[j] = pauli_basis
            bitstrings[j] = strings
        data_file['bases_list'].create_dataset(f't{i}', data=bases)
        data_file['bitstrings_list'].create_dataset(f't{i}', data=bitstrings)
