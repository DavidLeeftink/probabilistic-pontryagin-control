# %%
import os
import sys
one_level_up_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))# when calling from ppc/scripts/current_experiment/. Otherwise leave this out
sys.path.append(one_level_up_dir)
import jax._src.random as prng
import jax
from jax import config, jit
import jax.numpy as jnp
import jax.random as jr
import diffrax as dfx
import matplotlib.pyplot as plt
import optax as ox
import equinox as eqx
from ppc.systems.nonlinear_dynamics import VanDerPol
from ppc.control.trajectory_optimizers import *
from ppc.control.cost_functions import QuadraticCost
from ppc.nn.node import NeuralODE, EnsembleNeuralODE
from ppc.nn.nnvectorfield import NeuralVectorField, EnsembleNeuralVectorField
from ppc.fit import *
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
PRNGKey = prng.KeyArray
key = jr.PRNGKey(1231) 
key, subkey = jr.split(key)


# %%
from ppc.control.control_task import VanderPolStabilization
env = VanderPolStabilization()
env


# %% [markdown]
# ## data generation

# %%
## data generation
t0 = env.t0
tf = env.tf
dt = env.Delta_t
dt0_dense = 1e-2

num_trials = 25
num_obs = int((tf-t0)/dt)
D_in, D_out = env.D_sys+ env.D_control, env.D_sys
noise = jnp.array([jnp.sqrt(env.measurement_noise_std) for _ in range(D_out)]) # sigma^2 = 0.05 in Hegde experiment 4.1

## ode params
stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-5, jump_ts=None)
internal_solver = dfx.Dopri5() 
dt0_internal = 2e-2

## neural network
ensemble_size = 5
data_per_ensemble = num_trials
hidden_dim = 32
layer_sizes = (D_in, hidden_dim, hidden_dim, D_out)
activation = jax.nn.elu

## training
num_iters = 2_000
init_obs_noise = 0.5
batch_size = -1 # -1 or num_trials for no batching
lr = 0.001
log_rate = 20

# %%
# initial state distribution
real_system = env.real_system
x0_distribution = lambda key: .1*(jr.uniform(key=key,minval=-1, maxval=1, shape=(2,)))
key, subkey = jr.split(key)

# randomly sample observation times. 
ts_uniform = jnp.concatenate([jnp.linspace(t0, tf, num_obs)[None] for _ in range(num_trials)], axis=0)
ts = jnp.sort(ts_uniform, axis=1)
ts_dense = jnp.concatenate([jnp.linspace(t0, tf, num_obs*3)[None] for _ in range(num_trials)], axis=0)
freqs = jnp.arange(1,num_trials+1)
indices = jnp.linspace(0.2, 1.5, num_trials)
us = jax.vmap(lambda t, i: 2*jnp.cos(5.5*jr.uniform(jr.PRNGKey(round(3**i)))*jnp.pi*t*i+jr.normal(jr.PRNGKey(round(2**i))))+
              1.5*jnp.cos(2.5*jr.uniform(jr.PRNGKey(round(4**i)))*jnp.pi*t*i+jr.normal(jr.PRNGKey(round(5**i)))), in_axes=(0,0))(ts_dense, indices)[..., None]

key, subkey = jr.split(key)


# simulate data from real system
key, subkey = jr.split(key)
data, true_y0s = env.real_system.generate_synthetic_data(subkey,
                                           num_trials,
                                           dt0=dt0_dense,
                                           ts=ts,
                                           us=us,
                                           obs_stddev=noise,
                                           ts_dense=ts_dense,
                                           x0_distribution=env.get_initial_condition,#x0_distribution,#
                                           standardize_at_initialisation=False,
                                          )

# %%
env.real_system.visualize_phaseplane(data=data, X_range=jnp.array([[-4., 4.],[-4., 4.]]))
plt.show()
# real_system.visualize_longitunidal(test_ts=ts, true_y0s=true_y0s, data=data)
# plt.show()

# %% [markdown]
# ## Train neural ODE

# %%
keys = jr.split(subkey, ensemble_size)

ensemble_datasets = jax.vmap(get_batch, in_axes=(None, None, 0, None))(data, data_per_ensemble, keys, False)
key, subkey = jr.split(key)
ensemble_datasets.ys.shape

# %%
from ppc.nn.nnvectorfield import EnsembleNeuralVectorField

ensemble_vectorfield = EnsembleNeuralVectorField(
                                ensemble_size=ensemble_size,
                                layer_sizes=layer_sizes,
                                activation=activation,
                                D_sys=real_system.D_sys,
                                D_control=real_system.D_control,
                                key=key,)

# %%
from ppc.nn.node import EnsembleNeuralODE
""" testing the neural ODE call with the newly created vectorfield class."""

# initialize p(x0)
x0_mean_init = ensemble_datasets.ys[:,:,0,:]
x0_diag_raw = jnp.zeros_like(x0_mean_init)-1.

ensemble_node = EnsembleNeuralODE(
                          ensemble_size=ensemble_size,
                          obs_noise_raw=jnp.log(jnp.exp(init_obs_noise)-1.),
                          x0_mean=x0_mean_init-1,
                          x0_diag_raw=x0_diag_raw,
                          vectorfield=ensemble_vectorfield,
                          solver=dfx.Dopri5(),
                          dt0=dt0_internal,
                          stepsize_controller=dfx.PIDController(rtol=1e-4,atol=1e-5),
                          D_sys=real_system.D_sys,
                          D_control=real_system.D_control,
                          )

# %%
ensemble_node

# %%
from ppc.fit import fit_node
from ppc.nn.node import mse_loss_ensemble

optim = ox.adam(learning_rate=lr)

opt_ensemble_node, history = fit_node(model=ensemble_node, 
         objective=mse_loss_ensemble, 
         train_data=ensemble_datasets, 
         optim = optim, 
         key=subkey,
         num_iters=num_iters,
         batch_size=batch_size,
         log_rate=log_rate,)
key, subkey = jr.split(key, 2)

plt.plot(history)
plt.show()

# %%


# %% [markdown]
# ## MPC with learned model

# %%
from ppc.control.trajectory_optimizers import *
from ppc.control.mpc import ensembleDirectMPC

H = 3
n_controls = int(H/env.Delta_t)
maxiter=15
trial_ts = jnp.linspace(env.t0, env.tf, int( ((1/env.Delta_t))*(tf-t0)))#jnp.arange(env.t0, env.tf, env.Delta_t)

sqp_solver = SLSQP(
                lb=env.lb*jnp.ones((n_controls, env.D_control)),
                ub=env.ub*jnp.ones((n_controls, env.D_control)),
                maxiter=maxiter,)

# cem_solver = CEM(
#                 lb=env.lb*jnp.ones((n_controls, env.D_control)),
#                 ub=env.ub*jnp.ones((n_controls, env.D_control)),
#                 maxiter=maxiter,
#                 pop_size=500,
#                 elite_size=int(0.13*500),
#                 alpha=0.3,
#                 init_var=1)


ensemble_mpc = ensembleDirectMPC(
            traj_optimizer=sqp_solver,
            real_system=real_system,
            internal_system=opt_ensemble_node,
            state_cost=env.state_cost,
            termination_cost=env.termination_cost,
            verbose=True,
            )

# %%
# run with ground truth as internal model
ts, ts_dense, X, ys, U, R = ensemble_mpc.simulate(
                        x0=env.get_initial_condition(subkey), 
                        ts=trial_ts,
                        Delta_t=env.Delta_t,
                        x_star=env.x_star,
                        dt0_dense=dt0_dense,
                        H=H,
                        obs_noise=0.,
                        key=subkey,
                    )
key, subkey = jr.split(key)

sqp_ensemble_trial_cost = R[-1]

plt.plot(ts_dense, R)
plt.title(f'Direct approach with NODE: Integrated Cost: {R[-1]}')
plt.ylabel('Cost ')
plt.xlabel('t')
plt.show()

# %% [markdown]
# ## CEM - ensemble

# %%
from ppc.control.trajectory_optimizers import *
from ppc.control.mpc import ensembleDirectMPC

H = 3
n_controls = int(H/env.Delta_t)
# trial_ts = jnp.arange(env.t0, env.tf, env.Delta_t)

cem_solver = CEM(
                lb=env.lb*jnp.ones((n_controls, env.D_control)),
                ub=env.ub*jnp.ones((n_controls, env.D_control)),
                maxiter=maxiter,
                pop_size=700,
                elite_size=int(0.13*500),
                alpha=0.3,
                init_var=1)


ensemble_mpc = ensembleDirectMPC(
            traj_optimizer=cem_solver,
            real_system=real_system,
            internal_system=opt_ensemble_node,
            state_cost=env.state_cost,
            termination_cost=env.termination_cost,
            verbose=True,
            )

# run with ground truth as internal model
ts, ts_dense, X, ys, U, R = ensemble_mpc.simulate(
                        x0=env.get_initial_condition(subkey), 
                        ts=trial_ts,
                        Delta_t=env.Delta_t,
                        x_star=env.x_star,
                        dt0_dense=dt0_dense,
                        H=H,
                        obs_noise=0.,
                        key=subkey,
                    )
key, subkey = jr.split(key)

cem_ensemble_trial_cost = R[-1]
plt.plot(ts_dense, R)
plt.title(f'iCEM with NODE: Integrated Cost: {R[-1]}')
plt.ylabel('Cost ')
plt.xlabel('t')
plt.show()

# %% [markdown]
# ## Adam - ensemble

# %%
from ppc.control.trajectory_optimizers import *
from ppc.control.mpc import ensembleDirectMPC

H = 3
n_controls = int(H/env.Delta_t)
# trial_ts = jnp.arange(env.t0, env.tf, env.Delta_t)

adam_solver = Adam(
                lb=env.lb*jnp.ones((n_controls, env.D_control)),
                ub=env.ub*jnp.ones((n_controls, env.D_control)),
                maxiter=maxiter,)


ensemble_mpc = ensembleDirectMPC(
            traj_optimizer=adam_solver,
            real_system=real_system,
            internal_system=opt_ensemble_node,
            state_cost=env.state_cost,
            termination_cost=env.termination_cost,
            verbose=True,
            )

# run with ground truth as internal model
ts, ts_dense, X, ys, U, R = ensemble_mpc.simulate(
                        x0=env.get_initial_condition(subkey), 
                        ts=trial_ts,
                        Delta_t=env.Delta_t,
                        x_star=env.x_star,
                        dt0_dense=dt0_dense,
                        H=H,
                        obs_noise=0.,
                        key=subkey,
                    )
key, subkey = jr.split(key)

adam_ensemble_trial_cost = R[-1]
plt.plot(ts_dense, R)
plt.title(f'Adam with NODE: Integrated Cost: {R[-1]}')
plt.ylabel('Cost ')
plt.xlabel('t')
plt.show()

# %% [markdown]
# ## BFGS - ensemble

# %%
from ppc.control.trajectory_optimizers import *
from ppc.control.mpc import ensembleDirectMPC

H = 3
n_controls = int(H/env.Delta_t)
# trial_ts = jnp.arange(env.t0, env.tf, env.Delta_t)

bfgs_solver = LBFGSB(
                lb=env.lb*jnp.ones((n_controls, env.D_control)),
                ub=env.ub*jnp.ones((n_controls, env.D_control)),
                maxiter=maxiter,)

ensemble_mpc = ensembleDirectMPC(
            traj_optimizer=bfgs_solver,
            real_system=real_system,
            internal_system=opt_ensemble_node,
            state_cost=env.state_cost,
            termination_cost=env.termination_cost,
            verbose=True,
            )

# run with ground truth as internal model
ts, ts_dense, X, ys, U, R = ensemble_mpc.simulate(
                        x0=env.get_initial_condition(subkey), 
                        ts=trial_ts,
                        Delta_t=env.Delta_t,
                        x_star=env.x_star,
                        dt0_dense=dt0_dense,
                        H=H,
                        obs_noise=0.,
                        key=subkey,
                    )
key, subkey = jr.split(key)


bfgs_ensemble_trial_cost = R[-1]
plt.plot(ts_dense, R)
plt.title(f'BFGS with NODE: Integrated Cost: {R[-1]}')
plt.ylabel('Cost ')
plt.xlabel('t')
plt.show()

# %% [markdown]
# ## indirect approach

# %%
from ppc.control.trajectory_optimizers import EnsemblePMPForward
from ppc.control.mpc import ensembleDirectMPC, ensembleIndirectMPC

n_segments = 4
pmp_solver = EnsemblePMPForward(f=opt_ensemble_node.vectorfield,
                        D_sys=real_system.D_sys, 
                        D_control=real_system.D_control,
                        ensemble_size=ensemble_size,
                        n_segments=n_segments,
                        state_cost=env.state_cost,
                        termination_cost=env.termination_cost,
                        maxiter=maxiter, 
                        lb=env.lb*jnp.ones((n_controls, real_system.D_control)), 
                        ub=env.ub*jnp.ones((n_controls, real_system.D_control)),
                        )


ensemble_indirect_mpc = ensembleIndirectMPC(traj_optimizer=pmp_solver,
            real_system=real_system,
            internal_system=opt_ensemble_node,
            state_cost=env.state_cost,
            termination_cost=env.termination_cost,
            verbose=True,
            )

# %%
# run with ground truth as internal model
H=3
ts, ts_dense, X, Y, U, R = ensemble_indirect_mpc.simulate(
                        x0=env.get_initial_condition(subkey), 
                        ts=trial_ts,
                        Delta_t=env.Delta_t,
                        x_star=env.x_star,
                        dt0_internal=dt0_internal,
                        dt0_dense=dt0_dense,
                        H=H,
                        obs_noise=0.,
                        key=subkey,
                    )
key, subkey = jr.split(key)


pmp_ensemble_trial_cost = R[-1]
plt.plot(ts_dense, R)
plt.title(f'Indirect approach with NODE: Integrated Cost: {R[-1]}')
plt.ylabel('Cost ')
plt.xlabel('t')
plt.show()

# %% [markdown]
# ### Mean Hamiltonian PMP

# %%
from ppc.control.trajectory_optimizers import EnsemblePMPForwardMeanHamiltonian
from ppc.control.mpc import ensembleDirectMPC, ensembleIndirectMPC
maxiter = 15
n_segments = 4
pmp_solver = EnsemblePMPForwardMeanHamiltonian(f=opt_ensemble_node.vectorfield,
                        D_sys=real_system.D_sys, 
                        D_control=real_system.D_control,
                        ensemble_size=ensemble_size,
                        n_segments=n_segments,
                        state_cost=env.state_cost,
                        termination_cost=env.termination_cost,
                        maxiter=maxiter, 
                        lb=env.lb*jnp.ones((n_controls, real_system.D_control)), 
                        ub=env.ub*jnp.ones((n_controls, real_system.D_control)),
                        )


ensemble_indirect_mpc = ensembleIndirectMPC(traj_optimizer=pmp_solver,
            real_system=real_system,
            internal_system=opt_ensemble_node,
            state_cost=env.state_cost,
            termination_cost=env.termination_cost,
            verbose=True,
            )

# %%
# run with ground truth as internal model
H=3
ts, ts_dense, X, Y, U, R = ensemble_indirect_mpc.simulate(
                        x0=env.get_initial_condition(subkey), 
                        ts=trial_ts,
                        Delta_t=env.Delta_t,
                        x_star=env.x_star,
                        dt0_internal=dt0_internal*2.5,
                        dt0_dense=dt0_dense,
                        H=H,
                        obs_noise=0.,
                        key=subkey,
                    )
key, subkey = jr.split(key)


pmp_ensemble_trial_cost_mean_hamiltonian = R[-1]
plt.plot(ts_dense, R)
plt.title(f'Indirect approach with NODE: Integrated Cost: {R[-1]}')
plt.ylabel('Cost ')
plt.xlabel('t')
plt.show()

# %% [markdown]
# ## Compute mean objective with SQP

# %%
print('Seed: ', key)
print('PMP_ determinsitic cost: ', pmp_ensemble_trial_cost)
print('PMP_ determinsitic cost: ', pmp_ensemble_trial_cost_mean_hamiltonian)
print('CEM determinsitic cost: ', cem_ensemble_trial_cost)
print('SQP determinsitic cost: ', sqp_ensemble_trial_cost)
print('BFGS determinsitic cost: ', bfgs_ensemble_trial_cost)
print('Adam determinsitic cost: ', adam_ensemble_trial_cost)



