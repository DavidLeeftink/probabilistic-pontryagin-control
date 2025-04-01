# %%
import os
import sys
one_level_up_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../")) # when calling from ppc/scripts/current_experiment/. Otherwise leave this out
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
# from ppc.dataset import DiffEqDataset
from ppc.nn.node import NeuralODE, EnsembleNeuralODE
from ppc.nn.nnvectorfield import NeuralVectorField, EnsembleNeuralVectorField
from ppc.fit import *
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
PRNGKey = prng.KeyArray
key = jr.PRNGKey(123) 
key, subkey = jr.split(key)


# %%
from ppc.control.control_task import DuffingStabilization
env = DuffingStabilization()
env


# %% [markdown]
# ## data generation

# %%


# %%
# initial state distribution
real_system = env.real_system
# x0_distribution = lambda key: jnp.array([1.,1.])#.1*(jr.uniform(key=key, shape=(2,))-0.5)
key, subkey = jr.split(key)
t0 = env.t0
tf = env.tf

# %% [markdown]
# ## Compute mean objective with SQP

# %%

maxiter = 15
from ppc.control.mpc import *
H = 3.
us_init = 0.*jr.normal(key=subkey**2, shape=(jnp.arange(0, H, env.Delta_t).shape[0], 1))
key, subkey = jr.split(key)

sqp_solver = SLSQP(
                # pop_size=500, 
                # elite_size=50,
                # init_var=4,
            lb=env.lb*jnp.ones_like(us_init),
            ub=env.ub*jnp.ones_like(us_init),
            maxiter=maxiter,)
cem_solver = CEM(pop_size=500,
                 elite_size = int(0.13*500),
                 alpha=0.3, 
                 init_var = 1., 
                lb=env.lb*jnp.ones_like(us_init),
                ub=env.ub*jnp.ones_like(us_init),
                maxiter=maxiter,)

true_mpc = MPC(
        traj_optimizer=sqp_solver,
        real_system=env.real_system,
        internal_system=env.real_system,
        state_cost=env.state_cost,
        termination_cost=env.termination_cost,
        verbose=True,
        )

ts, ts_dense, X, Y, U, R = true_mpc.simulate(x0=env.get_initial_condition(),
        ts=jnp.linspace(env.t0, env.tf, int( ((1/env.Delta_t))*(tf-t0))),
        Delta_t=env.Delta_t,
        dt0_dense=1e-2,
        x_star=env.x_star,
        H=H, 
        )



plt.plot(ts_dense, R)
plt.title(f'Direct approach: Integrated Cost: {R[-1]}')
plt.ylabel('Cost ')
plt.xlabel('t')
plt.show()

# %% [markdown]
# ## Indirect approach

# %%
from ppc.control.trajectory_optimizers import PMPForward
from ppc.control.mpc import indirect_MPC
n_segments = 4
pmp_solver = PMPForward(f=env.real_system.f,
                maxiter=maxiter,
                D_sys=env.D_sys,
                D_control=env.D_control,
                n_segments=n_segments,
                state_cost=env.state_cost,
                termination_cost=env.termination_cost,
                lb=env.lb*jnp.ones_like(us_init[0]),
                ub=env.ub*jnp.ones_like(us_init[0]),
                )
        
indirect_mpc = indirect_MPC(traj_optimizer=pmp_solver,
    real_system=env.real_system,
    internal_system=env.real_system,
    state_cost=env.state_cost,
    termination_cost=env.termination_cost,
    verbose=True,
    )

ts, ts_dense, X, Y, U, R = indirect_mpc.simulate(x0=env.get_initial_condition(),
    ts=jnp.linspace(env.t0, env.tf, int( ((1/env.Delta_t))*(tf-t0))),
    Delta_t=env.Delta_t,
    dt0_internal=0.05,
    dt0_dense=1e-2,
    x_star=env.x_star,
    H=H,
    key=jr.PRNGKey(44))

plt.plot(ts_dense, R)
plt.title(f'Indirect approach: Integrated Cost: {R[-1]}')
plt.ylabel('Cost ')
plt.xlabel('t')
plt.show()

# %% [markdown]
# 

# %%
end_idx = int(ts_dense.shape[0]/1.5)
names = [r'$x$', r'$\sin(\theta)$', r'$\cos(\theta)$', r'$\dot{x}$', r'$\dot{\theta}$']
for i in range(env.D_sys):
    plt.plot(ts_dense[:end_idx], X[:end_idx,i], label=names[i])
plt.axhline(y=1., color='C0', linestyle=':')
plt.axhline(y=-1, color='C1', linestyle=':')
plt.axhline(y=0., color='C2', linestyle=':')
plt.axhline(y=0., color='C3', linestyle=':')
plt.axhline(y=0., color='C4', linestyle=':')
plt.legend()
plt.show()

# %%



