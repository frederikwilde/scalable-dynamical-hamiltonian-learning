import h5py
import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
import matplotlib.pyplot as plt

from differentiable_tebd.physical_models.heisenberg_disordered import mps_evolution_order2
from differentiable_tebd.utils.mps import mps_zero_state
from differentiable_tebd.utils.mps_qubits import probability, local_basis_transform

from adam import Adam


### SET CONFIG PARAMETERS ###
cf = SimpleNamespace()
'''Let's dump all config parameters that we set into a simple object.'''

cf.time_selection = [0, 1, 2, 3, 4]
'''Here we pick the time stamps we want to use.
[0, 2], for instance would only load data from the first and third time stamp.'''

cf.num_bases = 100
'''The number of Pauli bases we want to use.'''

cf.num_bitstrings_per_basis = 100
'''The number of bitstrings per Pauli basis.'''

cf.batch_size = 1
'''Amount of bitstrings to use per gradient step.
This determines the amount of stochasticity in the gradient.'''
if cf.num_bitstrings_per_basis % cf.batch_size != 0:
    raise ValueError('Number of bitstrings must be a multiple of batch size!')
num_batches = cf.num_bitstrings_per_basis // cf.batch_size

cf.chi = 30
'''Bond dimension of the MPS.'''

cf.deltat = .02
'''Trotter step size.'''

cf.mps_perturbation = 1e-6
'''To avoid degeneracies in the first steps of the time evolution,
where the MPS is highly sparse, we add a small perturbation to the MPS elements.'''


### LOAD DATA ###
with h5py.File('dataset.hdf5', 'r') as data_file:
    cf.true_params = jnp.array(data_file.attrs['true_params'])
    cf.num_sites = data_file.attrs['num_sites']
    cf.time_stamps = [data_file.attrs['time_stamps'][t] for t in cf.time_selection]
    tebd_steps = []  # TEBD steps to get to the next time stamp
    for i in range(1, len(cf.time_stamps)):
        time_diff = cf.time_stamps[i] - cf.time_stamps[i-1]
        tebd_steps.append(time_diff // cf.deltat)
    nba, nbi = cf.num_bases, cf.num_bitstrings_per_basis
    bases_list = [data_file[f'bases_list/t{t}'][:nba] for t in cf.time_selection]
    bitstrings_list = [data_file[f'bitstrings_list/t{t}'][:nba, :nbi] for t in cf.time_selection]


### DEFINE LOSS ###
def negative_log_likelihood(bitstring, basis, mps):
    mpsX = local_basis_transform(mps, 1)
    mpsY = local_basis_transform(mps, 2)
    return - jnp.log(probability(mpsX, mpsY, mps, bitstring, basis))
# vectorize over bitstrings
batched_nll_only_bitstrings = jax.vmap(negative_log_likelihood, in_axes=(0, None, None))
# now also vectorize over bases and JIT compile
batched_nll = jax.jit(jax.vmap(batched_nll_only_bitstrings, in_axes=(0, 0, None)))
def loss(params, mps, deltat, steps, bitstrings_list, bases_list, total_num_bitstrings):
    nll = 0.
    for i in jnp.arange(len(steps), dtype=int):
        mps, _ = mps_evolution_order2(params, deltat, steps[i], mps)
        nll += jnp.sum(batched_nll(bitstrings_list[i], bases_list[i], mps))
    return nll / total_num_bitstrings


### OPTIMIZE ### Here only with Adam for simplicity.
cf.rng_seed = 12345
'''To change the initialization, change the seed.
Not every initialization will lead to convergence to the true parameters.'''
rng = np.random.default_rng(cf.rng_seed)
cf.initial_params = rng.random(cf.true_params.size)
# This is a pretty good schedule.
# Perhaps extending the second phase will give an improvement.
cf.step_size_schedule = [(1, .05), (3, .001)]
opt = Adam(cf.initial_params)
loss_history, param_history, grad_history = [], [], []
epoch_counter = 1
data_indeces = np.arange(cf.num_bitstrings_per_basis)
try:
    for num_epochs, step_size in cf.step_size_schedule:
        opt.step_size = step_size
        for _ in range(num_epochs):
            rng.shuffle(data_indeces)
            for indeces in tqdm(data_indeces.reshape(num_batches, cf.batch_size)):
                m = mps_zero_state(cf.num_sites, cf.chi, cf.mps_perturbation, rng)
                v, g = jax.value_and_grad(loss)(
                    opt.parameters,
                    m,
                    cf.deltat,
                    tebd_steps,
                    [b[:, indeces] for b in bitstrings_list],
                    bases_list,
                    len(tebd_steps) * cf.num_bases * cf.batch_size
                )
                loss_history.append(v)
                param_history.append(opt.parameters)
                grad_history.append(g)
                opt.step(g)
            print(f'Finished epoch {epoch_counter}: error = {np.linalg.norm(param_history[-1] - cf.true_params)}')
            epoch_counter += 1
finally:
    ### SAVE RESULTS ###
    with h5py.File('results.hdf5', 'w') as f:
        f.create_dataset('loss_history', data=loss_history)
        f.create_dataset('grad_history', data=grad_history)
        f.create_dataset('param_history', data=param_history)
        for key, value in cf.__dict__.items():  # dump meta parameters into file
            f.attrs[key] = value


### PLOT RESULTS ###
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
v = 100  # the running average for the (stochastic) loss
ax1.plot(np.convolve(loss_history, np.full(v, 1/v), mode='valid'))
ax1.set_ylabel('Averaged stochastic loss')
ax1.set_xlabel('Iterations')
norm = np.linalg.norm(cf.true_params)
ax2.plot([np.linalg.norm(p - cf.true_params) / norm for p in param_history])
ax2.set_ylabel('Relative error')
ax2.set_xlabel('Iterations')
plt.savefig('results.pdf')
