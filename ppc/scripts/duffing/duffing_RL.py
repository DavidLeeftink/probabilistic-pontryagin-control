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
from ppc.control.mpc import *
# from ppc.dataset import DiffEqDataset
from ppc.nn.node import NeuralODE, EnsembleNeuralODE
from ppc.nn.nnvectorfield import NeuralVectorField, EnsembleNeuralVectorField

from ppc.fit import *
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
PRNGKey = prng.KeyArray
key = jr.PRNGKey(1231) 
key, subkey = jr.split(key)

## mean optimal control posterior seed 1

# %% [markdown]
# ## Define control problem and generate data to train NODE

# %%
from ppc.control.control_task import DuffingStabilization
env = DuffingStabilization()
env


# %% [markdown]
# ## data generation

# %%
## data generation
num_trials = 10
t0 = env.t0
tf = env.tf
dt = env.Delta_t
dt0_dense = 1e-2 # 

num_initial_trials = 1
num_obs = int((tf-t0)/dt)
D_in, D_out = env.D_sys+ env.D_control, env.D_sys
noise = jnp.array([jnp.sqrt(env.measurement_noise_std) for _ in range(D_out)]) # sigma^2 = 0.05 in Hegde experiment 4.1

## ode params
stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-5, jump_ts=None)
internal_solver = dfx.Dopri5() 
dt0_internal = 0.025

## neural network
ensemble_size = 5
data_per_ensemble = num_initial_trials
hidden_dim = 32
layer_sizes = (D_in, hidden_dim, hidden_dim, D_out)
activation = jax.nn.elu

## training
num_iters = 2_000
init_obs_noise = 0.5
batch_size = -1 # -1 or num_trials for no batching
lr = 0.0015
log_rate = 20

# MPC 
maxiter =20
H = 2
us_init = 0.*jr.normal(key=subkey**2, shape=(jnp.arange(0, H, env.Delta_t).shape[0], 1))

# %%
# initial state distribution
real_system = env.real_system
key, subkey = jr.split(key)

# randomly sample observation times. 
ts_uniform = jnp.concatenate([jnp.linspace(t0, tf, num_obs)[None] for _ in range(num_initial_trials)], axis=0)
ts = jnp.sort(ts_uniform, axis=1)
ts_dense = jnp.concatenate([jnp.linspace(env.t0, env.tf, int( ((1/dt0_dense))*(tf-t0)))[None] for _ in range(num_initial_trials)], axis=0)
freqs = jnp.arange(1,num_initial_trials+1)
indices = jnp.linspace(0.2, 1.5, num_initial_trials)
us = jax.vmap(lambda t, i:1*jnp.cos(5.*jr.uniform(jr.PRNGKey(round(3**i)))*t+jr.normal(key)))(ts_dense, indices)[..., None]
key, subkey = jr.split(key)


# simulate data from real system
key, subkey = jr.split(key)
data, true_y0s = env.real_system.generate_synthetic_data(subkey,
                                           num_initial_trials,
                                           dt0=dt0_dense,
                                           ts=ts,
                                           us=us,
                                           obs_stddev=noise,
                                           ts_dense=ts_dense,
                                           x0_distribution=env.get_initial_condition,
                                           standardize_at_initialisation=False,
                                          )

# %%


# %% [markdown]
# ## Train neural ODE

# %%
from ppc.nn.nnvectorfield import EnsembleNeuralVectorField
from ppc.nn.node import EnsembleNeuralODE
from ppc.fit import fit_node
from ppc.nn.node import mse_loss_ensemble


def train_network(data, key):
    key, subkey = jr.split(key)
    keys = jr.split(subkey, ensemble_size)


    ensemble_datasets = jax.vmap(get_batch, in_axes=(None, None, 0, None))(data, data.n, keys, False)
    print(ensemble_datasets.n, ensemble_datasets.ys.shape)
    key, subkey = jr.split(key)

    ensemble_vectorfield = EnsembleNeuralVectorField(
                                ensemble_size=ensemble_size,
                                layer_sizes=layer_sizes,
                                activation=activation,
                                D_sys=real_system.D_sys,
                                D_control=real_system.D_control,
                                key=key,)

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
                            stepsize_controller=dfx.ConstantStepSize(),#dfx.PIDController(rtol=1e-3,atol=1e-5),
                            D_sys=real_system.D_sys,
                            D_control=real_system.D_control,
                            )

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

    plt.figure()
    plt.plot(history)
    plt.xlabel('Iterations')
    plt.ylabel('Ensemble MSE')
    plt.title(f'Final loss: {history[-1]}')
    plt.show()

    return opt_ensemble_node


# %%
def run_single_trial(opt_ensemble_node, key:jr.PRNGKey):

        us_init = jnp.zeros((jnp.arange(0, H, env.Delta_t).shape[0], env.D_control))


        ## FOR PETS:
        # cem_solver = CEM(
        #         lb=env.lb*jnp.ones_like(us_init),
        #         ub=env.ub*jnp.ones_like(us_init),
        #         maxiter=maxiter,
        #         pop_size=500,
        #         elite_size=int(500*0.13),
        #         alpha=0.3)
        # ensemble_direct_mpc = ensembleDirectMPC(
        #         traj_optimizer=cem_solver,
        #         real_system=env.real_system,
        #         internal_system=opt_ensemble_node,
        #         state_cost=env.state_cost,
        #         termination_cost=env.termination_cost,
        #         verbose=True,
        #         )
        # trial_ts = jnp.linspace(env.t0, env.tf, int( ((1/env.Delta_t))*(env.tf-env.t0))) #new_trial_dataset.ts


        # ts, ts_dense, X, Y, U, R = ensemble_direct_mpc.simulate(x0=env.get_initial_condition(),
        #         ts=trial_ts,
        #         Delta_t=env.Delta_t,
        #         dt0_dense=dt0_dense,
        #         x_star=env.x_star,
        #         H=H, 
        #         )

        trial_ts = jnp.linspace(env.t0, env.tf, int( ((1/env.Delta_t))*(tf-t0)))
        # ts, ts_dense, X, Y, U, R = ensemble_direct_mpc.simulate(x0=env.get_initial_condition(),
        #         ts=trial_ts,
        #         Delta_t=env.Delta_t,
        #         dt0_dense=dt0_dense,
        #         x_star=env.x_star,
        #         H=H, 
        #         )

        key, subkey = jr.split(key)        
        n_segments = 5
        pmp_solver = EnsemblePMPForwardMeanHamiltonian(f=opt_ensemble_node.vectorfield,
                                D_sys=real_system.D_sys, 
                                D_control=real_system.D_control,
                                ensemble_size=ensemble_size,
                                n_segments=n_segments,
                                state_cost=env.state_cost,
                                termination_cost=env.termination_cost,
                                maxiter=maxiter, 
                                lb=env.lb*jnp.ones((us_init.shape[0], real_system.D_control)),
                                ub=env.ub*jnp.ones((us_init.shape[0], real_system.D_control)),)

        ensemble_indirect_mpc = ensembleIndirectMPC(traj_optimizer=pmp_solver,
                real_system=real_system,
                internal_system=opt_ensemble_node,
                state_cost=env.state_cost,
                termination_cost=env.termination_cost,
                verbose=True,
                )

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
        plt.title(f'Indirect approach with NODE: Integrated Cost: {pmp_ensemble_trial_cost}')
        plt.ylabel('Cost ')
        plt.xlabel('t')
        plt.show()

        
        # visualize 
        labels = [r'$x$', r'$\theta$', r'$\dot{x}$', r'$\dot{\theta}$']
        for i in range(X.shape[-1]):
                dim_color = f'C{i*2}'
                plt.plot(ts_dense, X[:,i], label=labels[i], color=dim_color)

        plt.axhline(y=0, color='gray', linestyle=':')
        plt.title(f'Integrated Cost: {R[-1]}')
        plt.ylabel('x')
        plt.xlabel('t')
        plt.legend()
        plt.show()

        return ts, ts_dense, X, Y, U, R


# %% [markdown]
# ## To do: the dataset should be theta standardized! can this be done post trial?

# %% [markdown]
# ## Run RL trials

# %%
data.us.shape

# %%
trial_costs = []
for i in range(num_trials):
    print(f'Starting trial {i+1}/{num_trials}')
    # train deep NODE ensemble 
    key, subkey = jr.split(key)
    opt_ensemble_node = train_network(data, subkey)

    # run trial
    key, subkey = jr.split(key)
    ts, ts_dense, X, Y, U, R = run_single_trial(opt_ensemble_node=opt_ensemble_node, key=subkey)
    final_cost = R[-1]
    trial_costs.append(final_cost)

    # augment dataset
    new_trial_dataset = DiffEqDataset(ts[None,...], Y[None,...], U[None,...], ts_dense=ts_dense[None,...], standardize_at_initialisation=False)
    print('old data: ', data.n, data.ys.shape)
    print('new trial: ', new_trial_dataset.n, new_trial_dataset.ys.shape)
    data = data + new_trial_dataset
    print('Combiend data: ',data.n, data.ys.shape)

trial_costs = jnp.array(trial_costs)
plt.figure()
plt.title('Costs over time')
plt.xlabel('Trial')
plt.ylabel('Cost')
plt.plot(jnp.arange(trial_costs.shape[0]), trial_costs)
plt.show()


# %%
trial_costs


