"""
Copyright (c) 2025 David Leeftink
 
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""
from abc import ABC
import time 
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import jax
import jax.tree_util as jtu
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
import diffrax as dfx
import optimistix
from diffrax import AbstractGlobalInterpolation
from equinox import Module
from types import SimpleNamespace
import equinox as eqx
from beartype.typing import Callable, List, Optional, Union
from jaxtyping import Float
from jaxtyping import Array, Float
from gpdx.control.trajectory_optimizers import AbstractTrajOptimizer, PMPForward, EnsemblePMPForward, EnsemblePMPForwardBackward
from gpdx.control.cost_functions import QuadraticCost
from gpdx.systems.abstract import AbstractSystem
from gpdx.nn.node import EnsembleNeuralODE
from gpdx.control.abstract_control import AbstractController
from gpdx.dataset import DiffEqDataset
import imageio
    

@dataclass
class MPC(ABC):
    """
        Model Predictive Control strategy for direct optimization. 

        Initialization args:
            traj_optimizer (AbstractTrajOptimizer) trajectory optimization method that updates the controls online.
            real_system (AbstractSystem) true system to control
            internal_system (Module) learned or (partially) known model of the system dynamics that is used for online forecasting.
            state_cost (QuadraticCost) state cost function 
            termination_cost (QuadraticCost) termination cost function            
            verbose (bool) if True, make a tqdm bar that updates the control cost at every iteration.
            control_iterpolator (Callable) interpolation function of the control signals.
             
    """
    traj_optimizer:AbstractTrajOptimizer
    real_system:AbstractSystem
    internal_system:Union[AbstractSystem, Module, eqx.Module]
    state_cost:QuadraticCost=None
    termination_cost:QuadraticCost=None
    verbose:bool=True
    control_interpolator:Callable= [None,
                                    lambda ts, ys: dfx.CubicInterpolation(ts=ts, coeffs=dfx.backward_hermite_coefficients(ts,ys)),
                                    dfx.LinearInterpolation][1]

    def simulate(self,
            x0:Array,
            ts:Array,
            Delta_t:float,
            x_star:Array,
            dt0_dense:float=1e-3,
            H:float=2, 
            obs_noise:Union[Float, Array]=0.,
            key:PRNGKey=jr.PRNGKey(42),
            ):
        """
        Simulation of full trial with MPC. Includes optimizing based on an internal dynamics model, 
        and simulation of given real system with observation noise.

        Args:
            x0 (Array) (D_sys,) Initial state. 
            ts (Array) (T,) timepoints for the full episode. Not assumed to be uniform, but known a priori. In the future could be made adaptive.
            Delta_t (float) fixed time step at which system measurements are made.
            real_system (AbstractSystem) true system that is to be controlled. Used for solving the actual state transitions with. 
                            It is assumed the agent / optimization method does not have access to this.
            dt0_dense (float) Initial step size of the real system to simulate the ground truth with.
            H (float) hroizon length - how far ahead in the future the MPC controller plans.
            objective (QuadraticIntegralCost) Cost function of the control problem, that evaluates a control function u(t).
            obs_noise (vector) (D_sys,) gaussian observation noise (standard deviation)
        """
        assert jnp.mod(Delta_t / dt0_dense,1)== 0., "$\Delta t$ (control) should be divisible by $\Delta t$ for the dense simulation."
        assert jnp.mod(1/dt0_dense, 1) == 0, "$1/ \Delta t$ dense % 1 should be 0. for correctly storing simulation "
        assert jnp.max(ts)-jnp.min(ts) >= jnp.min(ts)+H, f"Trial duration is shorter than receding horizon: trial duration ({jnp.max(ts)-jnp.min(ts)}) vs horizon ({H})"

        self.x_star = x_star
        dense_steps = jnp.floor((ts[1]-ts[0]) / dt0_dense).astype(jnp.int32)
        ts_dense = jnp.arange(ts.min(), ts.max(), dt0_dense)

        n_controls = int(H/Delta_t)

        Y = jnp.zeros((ts.shape[0], self.real_system.D_sys))
        X = jnp.zeros((ts_dense.shape[0], self.real_system.D_sys))
        U = jnp.zeros((ts_dense.shape[0], self.real_system.D_control))
        R = jnp.zeros((ts_dense.shape[0], 1))

        ts_h_init = jnp.linspace(ts[0], ts[0]+H, n_controls) 
        us = jnp.zeros((n_controls, self.real_system.D_control)) # initial x0 training. allowed to optimize longer. this one should be solved with trajectory optimization.
        # us = jr.normal(key, (n_controls, self.real_system.D_control))
        start = time.time()
        for _ in range(10):
            us = self.traj_optimizer(init_params=us, y0=x0, control_ts=ts_h_init, objective=self.objective, return_info=False)
        end = time.time()

        u_t = self.control_interpolator(ts=ts_h_init, ys=us)
        x_t = x0
        R_t = R[0]
        traj_times = []

        for i, t0 in enumerate(ts):
            key, subkey = jr.split(key)

            # initiate moving horizon, estimate state 
            ts_h = jnp.linspace(t0, t0+H, n_controls) 
            y_t = x_t + obs_noise*jr.normal(key=subkey, shape=x_t.shape)
            Y = Y.at[i].set(y_t)
            y_hat_t = y_t # state-estimation here

            # delay compensation
            # y_hat_t_next = self.internal_system(y0=y_hat_t, ts=jnp.array([t0, t0+Delta_t]), u=u_t,dt0=ts_h[1]-ts_h[0])[-1] 
            
            # optimize controls
            start = time.time()
            if i > -1:
                us_shifted = us.at[:-1].set(us[1:])
                us = self.traj_optimizer(init_params=us_shifted, y0=y_hat_t, control_ts=ts_h, objective=self.objective, return_info=False)
            else:
                us = us*0.
            end = time.time()
            traj_times.append(round(end-start,4))
            u_t_next = self.control_interpolator(ts=ts_h, ys=us) # interpolated trajectory

            # Simulate real system forward
            ts_Delta_t = jnp.linspace(t0, t0+Delta_t, dense_steps)
            xs_R_dense = self.integrate_with_cost(y0=x_t, u=u_t, ts=ts_Delta_t, dt0=dt0_dense, R0=R_t)
            xs_dense, Rs = xs_R_dense[:,:self.real_system.D_sys], xs_R_dense[:,-1:]
            us_dense = vmap(u_t_next.evaluate)(ts_Delta_t)
            
            # Update true state, store intermediate simulation values
            start_idx, end_idx = i*dense_steps, jnp.minimum((i+1)*dense_steps, ts_dense.shape[0])
            X = X.at[start_idx:end_idx].set(xs_dense)
            U = U.at[start_idx:end_idx].set(us_dense)
            R = R.at[start_idx:end_idx].set(Rs)
            x_t = xs_dense[-1]
            R_t = R[end_idx-1]
            u_t = u_t_next

            # visualization, hard coded for Lorenz currently.
            plot_freq = -1
            if i%plot_freq==plot_freq-1:
                print('traj. opt. times: ', traj_times)
                traj_times = []
                fig, ax = self.visualize_horizon(ts=ts, ts_dense=ts_dense[:end_idx], ts_h=ts_h, i=i, x_star=x_star, dense_steps=dense_steps, t0=t0, us=us, Delta_t=Delta_t, 
                          x_t=x_t, X=X[:end_idx], U=U[:end_idx], y_hat_t_next=y_hat_t)
                plt.show()

        # add termination cost.
        R = R.at[-1].set(R[-1] + self.termination_cost(X[-1], U[-1]))
 
        return ts, ts_dense, X, Y, U, R
    
    def objective(self, params:Array, y0:Array, control_ts:Array):
        u = self.control_interpolator(ts=control_ts, ys=params)
        R0 = jnp.array([0.])
        x0 = jnp.concatenate((y0, R0))

        saveat = dfx.SaveAt(t1=True)
        jump_ts = control_ts if repr(self.control_interpolator)==str(dfx.LinearInterpolation) else None
        stepsize_controller = dfx.ConstantStepSize()# dfx.PIDController(rtol=1e-4, atol=1e-5, jump_ts=jump_ts) 
        sol = dfx.diffeqsolve(
                dfx.ODETerm(self.f_state_and_objective_internal),
                solver=dfx.Dopri5(),
                saveat=saveat,
                t0=control_ts[0],
                t1=control_ts[-1],
                dt0=control_ts[1]-control_ts[0],
                stepsize_controller=stepsize_controller,
                y0=x0,
                args=u,
                ).ys
        return sol[-1,-1] + self.termination_cost(sol[-1,:-1], 0.*params[-1])
    
    def f_state_and_objective_internal(self, t:float, x:Array, args):
        u = args.evaluate(t) #if isinstance(u, dfx.AbstractGlobalInterpolation) else u
        if self.real_system.D_sys == 1:
            u = u[0]
        xdot = self.internal_system.f(t, x[:self.real_system.D_sys], args)
        R = self.state_cost(x[:self.real_system.D_sys], u)[None]

        return jnp.concatenate((xdot, R))
    
    def integrate_with_cost(self, y0:Array, u:Callable, ts:Array, dt0:float, R0:float=None):
        R0 = R0 if R0 is not None else jnp.array([0.])
        x0 = jnp.concatenate((y0, R0))
        saveat = dfx.SaveAt(ts=ts)
        stepsize_controller = dfx.PIDController(rtol=1e-4, atol=1e-6, jump_ts=ts)

        sol = dfx.diffeqsolve(
                dfx.ODETerm(self.f_state_objective_real),
                solver=dfx.Dopri5(),
                saveat=saveat,
                t0=ts[0],
                t1=ts[-1],
                dt0=dt0,
                stepsize_controller=stepsize_controller,
                y0=x0,
                args=u,
                ).ys
        return sol

    def f_state_objective_real(self, t:float, x:Array, args):
        u = args.evaluate(t) #if isinstance(u, dfx.AbstractGlobalInterpolation) else u
        if self.real_system.D_sys == 1:
            u = u[0]
        xdot = self.real_system.f(t, x[:self.real_system.D_sys], args)
        R = self.state_cost(x[:self.real_system.D_sys], u)[None]

        return jnp.concatenate((xdot, R))
    
    def estimate_state(self, t:float, y_t:Array, initial_state_dist:Callable):
        pass

    def initiate_particle_filter(self, )-> None:
        pass

    def visualize_horizon(self, ts:Array, ts_dense:Array, ts_h:Array, i:int, x_star:Array, dense_steps:int, t0:float, Delta_t:float, 
                          x_t:Array, us:Array, X:Array, U:Array, y_hat_t_next:Array):
        
        fig = plt.figure(figsize=(20,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[0.4, 0.4, 0.2])  # Width ratios

        # Create the subplots
        ax0 = fig.add_subplot(gs[0])  # First subplot
        ax1 = fig.add_subplot(gs[1])  # Second subplot
        ax2 = fig.add_subplot(gs[2])  # Third subplot
        ax = [ax0, ax1, ax2]
        H = ts_h.max() - ts_h.min()

        ax[0].plot(ts_dense[:i*dense_steps], U[:i*dense_steps], color='purple')
        dense_ts_h = jnp.linspace(ts_h.min(), ts_h.max(), ts_h.shape[0]*10)
        xs_planned = self.internal_system(y0=y_hat_t_next, ts=dense_ts_h, u=us, u_timepoints=ts_h)
        xs_planned_true = self.real_system(y0=y_hat_t_next, ts=dense_ts_h, u=us, u_timepoints=ts_h)


        ax[0].plot(ts_h, us, color='purple', alpha=0.6) # planned control signal
        ax[0].scatter(ts_h, us, color='purple') # planned control signal

        ax[0].axvline(x=t0, color='black')
        ax[0].set_title(f'u(t)')
        ax[0].set_xlim((ts.min(), ts.max()))
        ax[0].fill_between(ts, y1=self.traj_optimizer.ub[0], y2=self.traj_optimizer.lb[0], where=np.logical_and(ts >= t0, ts<=t0+H ), facecolor='gray', alpha=.2)
    
        for n in range(self.real_system.D_sys):
            dim_color = f'C{n*2}'
            ax[1].scatter(ts[:i], X[::dense_steps][:i,n], color=dim_color)
            ax[1].plot(ts_dense[:i*dense_steps], X[:i*dense_steps,n], color=dim_color)
            ax[1].scatter(t0+Delta_t, x_t[n], color='gold')
            ax[1].plot(dense_ts_h, xs_planned[:,n], color=dim_color, linestyle=':', label='Internal model prediction' if n==0 else None) # planned internal trajectory
            ax[1].plot(dense_ts_h, xs_planned_true[:,n], color=dim_color, alpha=0.6, label='True trajectory' if n==0 else None) # planned true trajectory
            ax[1].axhline(y=x_star[i], linestyle=':', color='black')

        min, max = jnp.minimum(xs_planned.min(), X.min()), jnp.maximum(xs_planned.max(), X.max())
        ax[1].fill_between(ts, y1=min, y2=max, where=np.logical_and(ts >= t0, ts<=t0+H ), facecolor='gray', alpha=.2)
        ax[1].axvline(x=t0, color='black')
        ax[1].set_xlim((ts.min(), ts.max()))
        ax[1].set_title('states x(t)')
        ax[1].legend()

        data_vis = DiffEqDataset(ts=ts_dense[None,...], ys=X[None,...])
        if self.real_system.D_sys <= 3:
            self.real_system.visualize_phaseplane(X_range=self.real_system.X_range_phaseplane, data=data_vis, ax=ax[2])
        plt.tight_layout()
        return fig, ax


    def visualize_end_trajectory(self, ts, ts_dense, X, Y, U):
        fig, ax = plt.subplots(self.real_system.D_sys+self.real_system.D_control+1,1, figsize=(12,8))
        for i in range(self.real_system.D_sys):
            ax[i].scatter(ts[:], Y[:,i], color=f'C{i}', alpha=0.3, s=5)
            ax[i].plot(ts_dense, X[:,i], label='$x_{i} (t)$', color=f'C{i}')
            ax[i].axhline(y=self.state_cost.x_star[i], color='black', linestyle=':')
        for j in range(self.real_system.D_control):
            ax[self.real_system.D_sys+j].plot(ts_dense, U, label='$u(t)$', color='purple')
            ax[i].axhline(y=self.state_cost.u_star[i], color='black', alpha=0.3)

        ax[-1].plot(ts_dense, R, color='red')

        ax[0].set_title(f'$x(t)$')
        ax[1].set_title(f'$y(t)$')
        ax[2].set_title(f'$u(t)$')
        ax[3].set_title(f'$R(t)$')
        plt.xlabel('t')
        plt.tight_layout()
        plt.show()
        print(f'Integrated cost: {R[-1]}')

        data_vis = DiffEqDataset(ts=ts_dense[None,...], ys=X[None,...])
        self.real_system.visualize_phaseplane(X_range=jnp.array([[-2., 2.],[-2., 2.]]), data=data_vis)
        return fig, ax 
    

@dataclass
class indirect_MPC(ABC):
    """
        Model Predictive Control strategy for indirect optimization via Pontryagin Maximum Principle (PMP).
        
        Initialization args:
            real_system (AbstractSystem) true system that is to be controlled. Used for solving the actual state transitions with. 
                            It is assumed the agent / optimization method does not have access to this.
            internal_system (Module) learned or (partially) known model of the system dynamics that is used for online prediction.
            state_cost (QuadraticCost) state cost function 
            termination_cost (QuadraticCost) termination cost function            
            verbose (bool) if True, make a tqdm bar that updates the control cost at every iteration.
            control_iterpolator (Callable) interpolation function of the control signals.
            traj_optimizer (AbstractTrajOptimizer) indirect trajectory optimization method that updates the controls online. 
                                If none is given, use forward PMP.

    """
    real_system:AbstractSystem
    internal_system:AbstractSystem
    traj_optimizer:AbstractTrajOptimizer
    state_cost:QuadraticCost=None
    termination_cost:QuadraticCost=None
    verbose:bool=True

    def simulate(self,
            x0:Array,
            ts:Array,
            Delta_t:float,
            x_star:Array,
            dt0_internal:float=None,
            dt0_dense:float=1e-3,
            H:float=2, 
            obs_noise:Union[Float, Array]=0.,
            key:PRNGKey=jr.PRNGKey(42),
            ):
        """
        Simulation of full trial with MPC. Includes optimizing based on an internal dynamics model, 
        and simulation of given real system with observation noise.

        Args:
            x0 (Array) (D_sys,) Initial state. 
            ts (Array) (T,) timepoints for the full episode. Not assumed to be uniform, but known a priori. In the future could be made adaptive.
            Delta_t (float) MPC measurement time interval.
            x_star (Array) goal state
            dt0_internal (float) initial solver step size used in the state-costate solver.
            dt0_dense (float) initial solver step size used in the ground truth simulation of the real system.
            H (float) hroizon length - how far ahead in the future the MPC controller plans.
            obs_noise (vector) (D_sys,) gaussian observation noise (standard deviation)
            key (PRNGkey) random key.
        """
        assert jnp.mod(Delta_t / dt0_dense,1)== 0., "$\Delta t$ (control) should be divisible by $\Delta t$ for the dense simulation."
        assert jnp.mod(1/dt0_dense, 1) == 0, "$1/ \Delta t$ dense % 1 should be 0. for correctly storing simulation "
        assert jnp.max(ts)-jnp.min(ts) >= jnp.min(ts)+H, f"Trial duration is shorter than receding horizon: trial duration ({jnp.max(ts)-jnp.min(ts)}) vs horizon ({H})"
        if dt0_internal is None:
            dt0_internal = Delta_t
        D_sys, D_control = self.internal_system.D_sys, self.internal_system.D_control
        dense_steps = jnp.floor((ts[1]-ts[0]) / dt0_dense).astype(jnp.int32)
        ts_dense = jnp.arange(ts.min(), ts.max(), dt0_dense)
        n_controls = int(H/Delta_t) 

        Y = jnp.zeros((ts.shape[0], D_sys))
        X = jnp.zeros((ts_dense.shape[0], D_sys))
        U = jnp.zeros((ts_dense.shape[0], D_control))
        R = jnp.zeros((ts_dense.shape[0], 1))

        ts_h_init = jnp.linspace(ts[0], ts[0]+H, n_controls) 
        lambda0 = jnp.zeros((D_sys)) 
        
        start = time.time()
        u_prev = dfx.LinearInterpolation(ts=ts_h_init, ys=jnp.zeros((ts_h_init.shape[0], D_control)))
        key, subkey = jr.split(key)
        lambda0, X_init, l_init = self.initial_pmp_optim(lambda0=lambda0, ts_h=ts_h_init, x0=x0, x_star=x_star, u_prev=u_prev, key=subkey)
        u_t_next, _, lambda_seq = self.lambda0_to_control(lambda0, x0, ts_h_init, u_prev)

        key, subkey = jr.split(key)
        end = time.time()
        end_idx = dense_steps
       

        x_t = x0
        R_t = R[0]
        images, traj_times = [], []

        for i, t0 in enumerate(ts):
            key, subkey = jr.split(key)

            # initiate moving horizon, estimate state 
            ts_Delta_t = jnp.linspace(t0, t0+Delta_t, dense_steps)
            ts_h = jnp.linspace(t0, t0+H, n_controls) 
            
            y_t = x_t + obs_noise*jr.normal(key=subkey, shape=x_t.shape)
            Y = Y.at[i].set(y_t)
            y_hat_t = y_t # state-estimation here

           
            # optimize controls
            start = time.time()
            if i > -1:
                u_prev = dfx.LinearInterpolation(ts=ts_h, ys=jnp.zeros((ts_h.shape[0], D_control)))
                (lambda0, shooting_states), stats = self.traj_optimizer(lambda0, ts=ts_h, x0=y_hat_t, x_star=x_star, dt0=dt0_internal, u_prev=u_prev, X_init=X_init, l_init=l_init)
                u_t_next, _, lambda_seq = self.lambda0_to_control(lambda0, y_hat_t, ts_Delta_t, u_prev)
              
                lambda0 = lambda_seq[-1]
                start_ts = jnp.array(jnp.array_split(ts_h, self.traj_optimizer.n_segments))[1:,0] 
                shooting_states = self.shift_segments_forward(segments=shooting_states,start_ts=start_ts, Delta_t=Delta_t, dt0_internal=dt0_internal, u_prev=u_prev)
                X_init, l_init = shooting_states[:,:D_sys], shooting_states[:,D_sys:]

            else:
                u_t_next = dfx.LinearInterpolation(ts=ts_dense, ys=jnp.zeros((ts_dense.shape[0], D_control)))
            end = time.time()
            traj_times.append((round(end-start,4), stats[0]['num_steps']))

            # Simulate real system forward
            xs_R_dense = self.integrate_with_cost(y0=x_t, u=u_t_next, ts=ts_Delta_t, dt0=dt0_dense, R0=R_t)
            xs_dense, Rs = xs_R_dense[:,:D_sys], xs_R_dense[:,-1:]
            us_dense = vmap(u_t_next.evaluate)(ts_Delta_t)
            
            # Update true state, store intermediate simulation values
            start_idx, end_idx = i*dense_steps, jnp.minimum((i+1)*dense_steps, ts_dense.shape[0])

            # The first index was the last state of previous iteration, or initial condition at the start.
            X = X.at[start_idx:end_idx].set(xs_dense)
            U = U.at[start_idx:end_idx].set(us_dense)
            R = R.at[start_idx:end_idx].set(Rs)
            x_t = xs_dense[-1]
            R_t = R[end_idx-1]
            u_t = u_t_next

            # visualization
            plot_freq = 10
            if i%plot_freq==plot_freq-1:                
                print('Traj. opt. times: ', traj_times)
                traj_times = []
                fig, ax = self.visualize_horizon(ts=ts, ts_dense=ts_dense[:end_idx], ts_h=ts_h, i=i, dense_steps=dense_steps, t0=t0, Delta_t=Delta_t, 
                          x_t=x_t, x_star=x_star, X=X[:end_idx], U=U[:end_idx], lambda0=lambda0, y_hat_t_next=y_hat_t, u_prev=u_prev)
                # plt.savefig(f'animation/frame_{i}.png')
                plt.show()
                # images.append(imageio.imread(f'animation/frame_{i}.png'))


        # Save as GIF
        # imageio.mimsave('animation/animation.gif', images, fps=20, loop=0)
        R = R.at[-1].set(R[-1] + self.termination_cost(X[-1], U[-1]))
 
        return ts, ts_dense, X, Y, U, R
    
    def f_state_and_objective_internal(self, t:float, x:Array, args):
        u = args.evaluate(t) 
        if self.internal_system.D_sys == 1:
            u = u[0]
        xdot = self.internal_system.f(t, x[:self.real_system.D_sys], args)
        R = self.state_cost(x[:self.real_system.D_sys], u)[None]

        return jnp.concatenate((xdot, R))
    
    def integrate_with_cost(self, y0:Array, u:Callable, ts:Array, dt0:float, R0:float=None):
        R0 = R0 if R0 is not None else jnp.array([0.])
        x0 = jnp.concatenate((y0, R0))
        saveat = dfx.SaveAt(ts=ts)
        sol = dfx.diffeqsolve(
                dfx.ODETerm(self.f_state_objective_real),
                solver=dfx.Dopri5(),
                saveat=saveat,
                t0=ts[0],
                t1=ts[-1],
                dt0=dt0,
                y0=x0,
                args=u,
                max_steps=4096*2,
                ).ys
        return sol
    
    def lambda0_to_control(self, lambda0:Array, x0:Array, ts_dense:Array, u_prev:callable)->Array:
        """
            Convert a given initial condition of the co-states lambda(0), to a callable u(t) function.
            The system is solved forward for all the given input times, and then interpolated. 

            Args:
                lambda0 (D_sys,) Array - initial costate values to solve for.
                x0 (D_sys,) initial state to solve for.
                ts_dense (T,) dense time points to save the integrated values at.

            Return:
                u_callable (Callable, diffrax.GlobalInterpolation) control function u(t) that can be called with u_callable.evaluate(t) for a t in the range of ts_dense.
                ys (T, D_sys) simulated values over the internal dynamics. 
        """
        X0 = jnp.concatenate((x0, lambda0))
        dt0 = ts_dense[1]-ts_dense[0]
        ys = self.traj_optimizer._solve_state_costate(X0, ts_dense, dt0=dt0, u_prev=u_prev)
        ys = ys.reshape(-1,ys.shape[-1])
        state_sequence, costate_sequence = ys[:,:self.real_system.D_sys], ys[:,self.real_system.D_sys:]

        u_linearizations = jax.vmap(lambda t, u: u.evaluate(t), in_axes=(0,None))(ts_dense, u_prev)
        us_matrix = vmap(self.traj_optimizer.u_opt, in_axes=(0,0,0))(state_sequence, costate_sequence, u_linearizations)
        assert us_matrix.shape[0] == ts_dense.shape[0]
        u_callable = dfx.LinearInterpolation(ts=ts_dense, ys=us_matrix)

        return u_callable, state_sequence, costate_sequence

    def f_state_objective_real(self, t:float, x:Array, args):
        u = args.evaluate(t) 
        if self.real_system.D_sys == 1:
            u = u[0]
        xdot = self.real_system.f(t, x[:self.real_system.D_sys], args)
        R = self.state_cost(x[:self.real_system.D_sys], u)[None]

        return jnp.concatenate((xdot, R))
    
    def shift_segments_forward(self, segments:Array, start_ts:Array, Delta_t:float, dt0_internal:float, u_prev:callable):
        """
            Shift previously fitted segments forward Deltat time units. 
            Note: the initial start (t=t0) is not included. 

            Args:
                segments: (S-1, 2*D_x) where S is the number of total segments including the starts. Consists of states first (x), and then co-states (lambda)
                start_ts (S-1) array. Current starting times of the segments.
                Delta_t (float) duration to integrate for.
                dt0_internal (float) internal step size to use during solving.

            Return:
                shifted_segments (S-1, 2* D_x)
        """
        subts = jax.vmap(lambda start_t: jnp.linspace(start_t, start_t+Delta_t, num=2))(start_ts)
        
        shifted_segments = jax.vmap(lambda segment_x0, ts_shift: self.traj_optimizer._solve_state_costate(segment_x0, ts_shift, dt0=dt0_internal, u_prev=u_prev))(segments, subts)
        
        return shifted_segments[:,-1,:]
    
    def initial_pmp_optim(self, lambda0:Array, ts_h:Array, x0:Array, x_star:Array, u_prev:callable=None, key:PRNGKey=jr.PRNGKey(42)):
        X_init, l_init = None, None
        for i in range(50):
            key, subkey = jr.split(key)
            lambda0 = jr.normal(subkey, lambda0.shape)*(self.traj_optimizer.ub[0]/2)
            (lambda0, shooting_states), stats = self.traj_optimizer(lambda0, ts=ts_h, x0=x0, x_star=x_star, u_prev=u_prev, X_init=X_init, l_init=l_init)
            X_init, l_init = shooting_states[:,:self.real_system.D_sys], shooting_states[:,self.real_system.D_sys:] 
            (iteration_stats, result) = stats
            if result != optimistix.RESULTS.successful:
                print(f"Initial optimization (iteration {i}) not succesfull, continuing.")
            elif result == optimistix.RESULTS.successful:
                print(f'Succesful initial optimization at iteration {i}: ', lambda0)
                break
        print(f"Initial optimization over, starting trial with initial solution.")
        return lambda0, X_init, l_init

    def visualize_horizon(self, ts:Array, ts_dense:Array, ts_h:Array, i:int, dense_steps:int, t0:float, Delta_t:float, 
                          x_t:Array, x_star:Array, X:Array, U:Array, lambda0:Array, y_hat_t_next:Array, u_prev:callable):
        fig = plt.figure(figsize=(20,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[0.4, 0.4, 0.2])  # Width ratios

        # Create the subplots
        ax0 = fig.add_subplot(gs[0])  # First subplot
        ax1 = fig.add_subplot(gs[1])  # Second subplot
        ax2 = fig.add_subplot(gs[2])  # Third subplot
        ax = [ax0, ax1, ax2]

        # ax[0].scatter(ts[:i], U[::dense_steps][:i])
        H = ts_h.max() - ts_h.min()
        ax[0].plot(ts_dense[:i*dense_steps], U[:i*dense_steps], color='purple')
        dense_ts_h = jnp.linspace(ts_h.min(), ts_h.max(), ts_h.shape[0]*10)
        u_t_planned, xs_planned, _ = self.lambda0_to_control(lambda0, y_hat_t_next, ts_dense=dense_ts_h, u_prev=u_prev)
        us_planned = vmap(u_t_planned.evaluate)(dense_ts_h)
        xs_planned_true = self.real_system(y0=y_hat_t_next, ts=dense_ts_h, u=us_planned)

        ax[0].plot(dense_ts_h, us_planned, color='purple') # planned control signal
        ax[0].axvline(x=t0, color='black')
        ax[0].set_title(f'u(t)')
        ax[0].set_xlim((ts.min(), ts.max()))
        ax[0].fill_between(ts, y1=self.traj_optimizer.ub, y2=self.traj_optimizer.lb, where=np.logical_and(ts >= t0, ts<=t0+H ), facecolor='gray', alpha=.2)

        for n in range(self.real_system.D_sys):
            dim_color = f'C{int(n*2)}'
            ax[1].scatter(ts[:i], X[::dense_steps][:i,n], color=dim_color)
            ax[1].plot(ts_dense[:i*dense_steps], X[:i*dense_steps,n], color=dim_color)
            ax[1].scatter(t0+Delta_t, x_t[n], color='gold')

            ax[1].plot(dense_ts_h, xs_planned[:,n], color=dim_color, linestyle=':', label='Internal model prediction' if n==0 else None) # planned internal trajectory
            ax[1].plot(dense_ts_h, xs_planned_true[:,n], color=dim_color,  alpha=0.6,label='True trajectory' if n==0 else None) # planned true trajectory
            ax[1].axhline(y=x_star[n], linestyle=':', color='black')


        min, max = jnp.minimum(xs_planned.min(), X.min()), jnp.maximum(xs_planned.max(), X.max())
        ax[1].fill_between(ts, y1=min, y2=max, where=np.logical_and(ts >= t0, ts<=t0+H ), facecolor='gray', alpha=.2)
        ax[1].axvline(x=t0, color='black')
        ax[1].set_xlim((ts.min(), ts.max()))
        ax[1].set_title('states x(t)')
        ax[1].legend()

        data_vis = DiffEqDataset(ts=ts_dense[None,...], ys=X[None,...])
        if self.real_system.D_sys <= 3:
            self.real_system.visualize_phaseplane(X_range=self.real_system.X_range_phaseplane, data=data_vis, ax=ax[2])
        plt.tight_layout()
        return fig, ax



@dataclass 
class ensembleDirectMPC(ABC):
    """
        Model Predictive Control strategy for direct optimization. 

        Initialization args:
            traj_optimizer (AbstractTrajOptimizer) trajectory optimization method that updates the controls online.
            real_system (AbstractSystem) true system to control
            internal_system (Module) learned or (partially) known model of the system dynamics that is used for online forecasting.
            state_cost (QuadraticCost) state cost function 
            termination_cost (QuadraticCost) termination cost function            
            verbose (bool) if True, make a tqdm bar that updates the control cost at every iteration.
            control_iterpolator (Callable) interpolation function of the control signals.
             
    """
    traj_optimizer:Union[EnsemblePMPForward,EnsemblePMPForwardBackward]
    real_system:AbstractSystem
    internal_system:EnsembleNeuralODE
    state_cost:QuadraticCost=None
    termination_cost:QuadraticCost=None
    verbose:bool=True
    control_interpolator:Callable= [None,
                                    lambda ts, ys: dfx.CubicInterpolation(ts=ts, coeffs=dfx.backward_hermite_coefficients(ts,ys)),
                                    dfx.LinearInterpolation][-1]

    def simulate(self,
            x0:Array,
            ts:Array,
            Delta_t:float,
            x_star:Array,
            dt0_dense:float=1e-3,
            dt0_internal:float=1e-1,
            H:float=2, 
            obs_noise:Union[Float, Array]=0.,
            save_path:str=None,
            key:PRNGKey=jr.PRNGKey(42),
            ):
        """
        Simulation of full trial with MPC. Includes optimizing based on an internal dynamics model, 
        and simulation of given real system with observation noise.

        Args:
            x0 (Array) (D_sys,) Initial state. 
            ts (Array) (T,) timepoints for the full episode. Not assumed to be uniform, but known a priori. In the future could be made adaptive.
            Delta_t (float) fixed time step at which system measurements are made.
            real_system (AbstractSystem) true system that is to be controlled. Used for solving the actual state transitions with. 
                            It is assumed the agent / optimization method does not have access to this.
            dt0_dense (float) Initial step size of the real system to simulate the ground truth with.
            H (float) hroizon length - how far ahead in the future the MPC controller plans.
            objective (QuadraticIntegralCost) Cost function of the control problem, that evaluates a control function u(t).
            obs_noise (vector) (D_sys,) gaussian observation noise (standard deviation)
        """
        assert jnp.mod(Delta_t / dt0_dense,1)== 0., "$\Delta t$ (control) should be divisible by $\Delta t$ for the dense simulation."
        assert jnp.mod(1/dt0_dense, 1) == 0, "$1/ \Delta t$ dense % 1 should be 0. for correctly storing simulation "
        assert jnp.max(ts)-jnp.min(ts) >= jnp.min(ts)+H, f"Trial duration is shorter than receding horizon: trial duration ({jnp.max(ts)-jnp.min(ts)}) vs horizon ({H})"

        self.x_star = x_star
        dense_steps = jnp.floor((ts[1]-ts[0]) / dt0_dense).astype(jnp.int32)
        ts_dense = jnp.arange(ts.min(), ts.max(), dt0_dense)
        n_controls = int(H/Delta_t)
        E = self.internal_system.ensemble_size

        Y = jnp.zeros((ts.shape[0], self.real_system.D_sys))
        X = jnp.zeros((ts_dense.shape[0], self.real_system.D_sys))
        U = jnp.zeros((ts_dense.shape[0], self.real_system.D_control))
        R = jnp.zeros((ts_dense.shape[0], 1))

        ts_h_init = jnp.linspace(ts[0], ts[0]+H, n_controls) 
        us = jnp.zeros((n_controls, self.real_system.D_control)) 
        start = time.time()
        for _ in range(10):
            us = self.traj_optimizer(init_params=us, y0=x0, control_ts=ts_h_init, dt0_internal=dt0_internal, objective=self.objective, return_info=False)
        end = time.time()

        u_t = self.control_interpolator(ts=ts_h_init, ys=us)
        x_t = x0
        R_t = R[0]
        images, traj_times = [], []

        for i, t0 in enumerate(ts):
            key, subkey = jr.split(key)

            # initiate moving horizon, estimate state 
            ts_h = jnp.linspace(t0+Delta_t, t0+H, n_controls) 
            y_t = x_t + obs_noise*jr.normal(key=subkey, shape=x_t.shape)
            Y = Y.at[i].set(y_t)
            y_hat_t = y_t 

            # delay compensation
            one_step_ts = jnp.array([t0, t0+Delta_t])
            one_step_us = vmap(u_t.evaluate)(one_step_ts)
            y_hat_t_next = self.internal_system(
                                ts=jnp.tile(one_step_ts, (E,1)), 
                                y0=jnp.tile(y_hat_t,(E,1)), 
                                us=jnp.tile(one_step_us, (E,1,1)),
                                dt0=(ts_h[1]-ts_h[0]), 
                                return_ensemble_mean=True)[-1]  
            
            # optimize controls
            start = time.time()
            if i > -1:
                us_shifted = us.at[:-1].set(us[1:]) 
                us = self.traj_optimizer(init_params=us_shifted, y0=y_hat_t_next, control_ts=ts_h, dt0_internal=dt0_internal, objective=self.objective, return_info=False)
            else:
                us = us*0.
            end = time.time()
            traj_times.append(round(end-start,4))
            u_t_next = self.control_interpolator(ts=ts_h, ys=us) 

            # Simulate real system forward
            ts_Delta_t = jnp.linspace(t0, t0+Delta_t, dense_steps)
            xs_R_dense = self.integrate_with_cost(y0=x_t, u=u_t, ts=ts_Delta_t, dt0=dt0_dense, R0=R_t)
            xs_dense, Rs = xs_R_dense[:,:self.real_system.D_sys], xs_R_dense[:,-1:]
            us_dense = vmap(u_t.evaluate)(ts_Delta_t)
            
            # Update true state, store intermediate simulation values
            start_idx, end_idx = i*dense_steps, jnp.minimum((i+1)*dense_steps, ts_dense.shape[0])
            X = X.at[start_idx:end_idx].set(xs_dense)
            U = U.at[start_idx:end_idx].set(us_dense)
            R = R.at[start_idx:end_idx].set(Rs)
            x_t = xs_dense[-1]
            R_t = R[end_idx-1]
            u_t = u_t_next

            # visualization
            plot_freq = 51
            if i%plot_freq==plot_freq-1:                
                print('Traj. opt. times: ', traj_times)
                traj_times = []
                fig, ax = self.visualize_horizon(ts=ts, ts_dense=ts_dense[:end_idx], ts_h=ts_h, i=i, dense_steps=dense_steps, t0=t0, Delta_t=Delta_t, 
                          x_t=x_t, x_star=x_star, us=us, X=X[:end_idx], U=U[:end_idx], y_hat_t_next=y_hat_t_next)
                plt.show()
                # plt.savefig(f'{save_path}/animation/frame_{i}.png')
                # plt.close()
                # images.append(imageio.imread(f'{save_path}/animation/frame_{i}.png'))


        # Save as GIF
        # if plot_freq>-1:
        #     imageio.mimsave(f'{save_path}/animation/animation.gif', images, fps=5, loop=0)
        R = R.at[-1].set(R[-1] + self.termination_cost(X[-1], U[-1]))

        return ts, ts_dense, X, Y, U, R
    
    def objective(self, params:Array, y0:Array, control_ts:Array, dt0_internal:float=.05):
        """
            Forward solve that does not save intermediate trajectory values forfaster computation. 

              Args:
                params (N_control, D_control) matrix with u values, correspodning to control_ts as timepoints.
                y0 (D_sys) Array of initial conditions to solve
                control_ts (D_sys) time points to save the trajectories at.

              Returns:
                score (float) mean cost function averaged over all ensemble members.
        """
        ensemble_size = self.internal_system.ensemble_size

        R0 = jnp.zeros((ensemble_size,1))
        if y0.ndim == 1:
            y0 = jnp.tile(y0, (ensemble_size,1))
        y0 = jnp.concatenate((y0, R0),axis=1)
        if control_ts.ndim == 1:
            control_ts = jnp.tile(control_ts, (ensemble_size,1))
        if params.ndim == 2:
            params = jnp.tile(params, (ensemble_size,1,1))

        args = (control_ts, params)

        t0 = control_ts[0,0]
        tf = control_ts[0,-1]
        dt0= dt0_internal#control_ts[0,1]-control_ts[0,0] # dt0 

        saveat = dfx.SaveAt(t1=True)
        jump_ts = control_ts[0] if repr(self.control_interpolator)==str(dfx.LinearInterpolation) else None
        stepsize_controller = dfx.ConstantStepSize()#dfx.PIDController(rtol=1e-3, atol=1e-4)#, jump_ts=jump_ts)
        sol = dfx.diffeqsolve(
                dfx.ODETerm(self.f_state_and_objective_internal),
                solver=dfx.Dopri5(),
                saveat=saveat,
                t0=t0,
                t1=tf,
                dt0=dt0,
                stepsize_controller=stepsize_controller,
                y0=y0,
                args=args,
                max_steps=4096*2,
                ).ys[0] # [E, D_sys+1] -> for each ensemble member: (y(T), R(T))
        
        scores = sol[:,-1] + jax.vmap(lambda x, u: self.termination_cost(x,u))(sol[:,:-1], params[:,-1])
        # return scores
        mean_score = jnp.mean(scores)  
        return mean_score
    
    def f_state_and_objective_internal(self, t:float, x:Array, args):
        (ts, us) = args
        u_vals = self.vmap_eval_control(t, ts, us)     # (E,D_control)
        xdot = self.internal_system.f(t, x[:,:self.real_system.D_sys], args) # (E, D_sys)
        R = jax.vmap(self.state_cost, in_axes=(0,0))(x[:,:self.real_system.D_sys], u_vals) # (E, D_sys)
        return jnp.concatenate((xdot, R[:,None]), axis=1)

    def vmap_eval_control(self, t, ts, us):
        def eval_control(t, ts, us):
            return dfx.LinearInterpolation(ts=ts, ys=us).evaluate(t)
        return jax.vmap(eval_control, in_axes= (None,0,0))(t, ts, us)
        
    def integrate_with_cost(self, y0:Array, u:Callable, ts:Array, dt0:float, R0:float=None):
        """Integrate true dynamics model forward."""
        R0 = R0 if R0 is not None else jnp.array([0.])
        x0 = jnp.concatenate((y0, R0))
        stepsize_controller = dfx.ConstantStepSize()#dfx.PIDController(rtol=1e-4, atol=1e-5)

        saveat = dfx.SaveAt(ts=ts)
        sol = dfx.diffeqsolve(
                dfx.ODETerm(self.f_state_objective_real),
                solver=dfx.Dopri5(),
                saveat=saveat,
                t0=ts[0],
                t1=ts[-1],
                dt0=dt0,
                stepsize_controller=stepsize_controller,
                y0=x0,
                args=u,
                max_steps=4096*2,
                ).ys
        return sol

    def f_state_objective_real(self, t:float, x:Array, args):
        u = args.evaluate(t) #if isinstance(u, dfx.AbstractGlobalInterpolation) else u
        if self.real_system.D_sys == 1:
            u = u[0]
        xdot = self.real_system.f(t, x[:self.real_system.D_sys], args)
        R = self.state_cost(x[:self.real_system.D_sys], u)[None]

        return jnp.concatenate((xdot, R))

    def visualize_horizon(self, ts:Array, ts_dense:Array, ts_h:Array, i:int, x_star:Array, dense_steps:int, t0:float, Delta_t:float, 
                          x_t:Array, us:Array, X:Array, U:Array, y_hat_t_next:Array):
        """
            MPC visualization with planned horizon for 2 dimensional system.
        """
        fig = plt.figure(figsize=(20,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[0.4, 0.4, 0.2])  # Width ratios

        # Create the subplots
        
        axs = []
        for i in range(3):
            axs.append(fig.add_subplot(gs[i]))
        
        E = self.internal_system.ensemble_size
        H = ts_h.max() - ts_h.min()

        dense_ts_h = jnp.linspace(ts_h.min(), ts_h.max(), ts_h.shape[0]*10)
        ensemble_dense_ts_h = jnp.tile(dense_ts_h, (E,1))
        ensemble_ts_h = jnp.tile(ts_h, (E,1))
        ensemble_us = jnp.tile(us, (E,1,1))
        ensemble_y_hat_t_next = jnp.tile(y_hat_t_next, (E,1))
        xs_planned = self.internal_system(y0=ensemble_y_hat_t_next, ts=ensemble_dense_ts_h, us=ensemble_us, u_timepoints=ensemble_ts_h,) # []
        xs_planned_true = self.real_system(y0=y_hat_t_next, ts=dense_ts_h, u=us, u_timepoints=ts_h)


        axs[0].plot(ts_dense, U, color='purple')
        axs[0].plot(ts_h, us, color='purple', alpha=0.6) # planned control signal
        axs[0].scatter(ts_h, us, color='purple') # planned control signal

        axs[0].axvline(x=t0, color='black')
        axs[0].set_title(f'u(t)')
        axs[0].set_xlim((ts.min(), ts.max()))
        axs[0].fill_between(ts, y1=self.traj_optimizer.ub[0], y2=self.traj_optimizer.lb[0], where=np.logical_and(ts >= t0, ts<=t0+H ), facecolor='gray', alpha=.2)
    
        for n in range(self.real_system.D_sys):
            dim_color = f'C{n*2}'
            axs[1].scatter(ts_dense, X[:,n], color=dim_color)
            axs[1].plot(ts_dense, X[:,n], color=dim_color)
            axs[1].scatter(t0+Delta_t, x_t[n], color='gold')
            for e in range(self.internal_system.ensemble_size):
                axs[1].plot(dense_ts_h, xs_planned[e,:,n], color=dim_color, alpha=0.2, label='Sampled internal model prediction' if n+e==0 else None) # planned internal trajectory samples
            axs[1].plot(dense_ts_h, jnp.mean(xs_planned[:,:,n],axis=0), color=dim_color, alpha=0.8, label='Mean Internal model prediction' if n==0 else None) # planned internal trajectory average         
            axs[1].plot(dense_ts_h, xs_planned_true[:,n], color='black', alpha=0.6, label='True trajectory' if n==0 else None) # planned true trajectory
            axs[1].axhline(y=x_star[i], linestyle=':', color='black')

        min, max = jnp.minimum(xs_planned.min(), X.min()), jnp.maximum(xs_planned.max(), X.max())
        axs[1].fill_between(ts, y1=min, y2=max, where=np.logical_and(ts >= t0, ts<=t0+H ), facecolor='gray', alpha=.2)
        axs[1].axvline(x=t0, color='black')
        axs[1].set_xlim((ts.min(), ts.max()))
        axs[1].set_title('states x(t)')
        axs[1].legend()

        data_vis = DiffEqDataset(ts=ts_dense[None,...], ys=X[None,...])
        self.real_system.visualize_phaseplane(X_range=self.real_system.X_range_phaseplane, data=data_vis, ax=axs[2])
        plt.tight_layout()
        return fig, axs
    

@dataclass 
class ensembleIndirectMPC(ABC):
    """
        Model Predictive Control strategy with the forward Pontryagin method for trajectory optimization. 

        Initialization args:
            traj_optimizer (AbstractTrajOptimizer) trajectory optimization method that updates the controls online.
            real_system (AbstractSystem) true system to control
            internal_system (Module) learned or (partially) known model of the system dynamics that is used for online forecasting.
            state_cost (QuadraticCost) state cost function 
            termination_cost (QuadraticCost) termination cost function            
            verbose (bool) if True, make a tqdm bar that updates the control cost at every iteration.
            control_iterpolator (Callable) interpolation function of the control signals.
                   
    """
    traj_optimizer:AbstractTrajOptimizer
    real_system:AbstractSystem
    internal_system:EnsembleNeuralODE
    state_cost:QuadraticCost=None
    termination_cost:QuadraticCost=None
    verbose:bool=True
    
    def simulate(self,
            x0:Array,
            ts:Array,
            Delta_t:float,
            x_star:Array,
            dt0_internal:float=None,
            dt0_dense:float=1e-2,
            H:float=4, 
            obs_noise:Union[Float, Array]=0.,
            save_path:str=None,
            key:PRNGKey=jr.PRNGKey(42),
            standardize_x:callable=None,
            inverse_standardize_x:callable=None,
            standardize_u:callable=None,
            inverse_standardize_u:callable=None,
            ):
        """
        Simulation of full trial with MPC. Includes optimizing based on an internal dynamics model, 
        and simulation of given real system with observation noise.

        Args:
            x0 (Array) (D_sys,) Initial state. 
            ts (Array) (T,) timepoints for the full episode. Not assumed to be uniform, but known a priori. In the future could be made adaptive.
            Delta_t (float) MPC measurement time interval.
            x_star (Array) goal state
            dt0_internal (float) initial solver step size used in the state-costate solver.
            dt0_dense (float) initial solver step size used in the ground truth simulation of the real system.
            H (float) hroizon length - how far ahead in the future the MPC controller plans.
            obs_noise (vector) (D_sys,) gaussian observation noise (standard deviation)
            key (PRNGkey) random key.

        Returns:
            ts (Array)          time points of measurements during the trial
            ts_dense (Array)    dense time points from true system integrator, not accessible to the algorithm.
            X (Array)           true states during the trial, without noise. Saved at dense intervals.
            Y (Array)           measurements of the same system, saved at MPC time interval (Delta_t)
            U (Array)           controls made during the trial, saved at dense intervals.
            R (Array)           cumulative reward during the trial, saved at dense intervals.
        """
        assert jnp.mod(Delta_t / dt0_dense,1)== 0., "$\Delta t$ (control) should be divisible by $\Delta t$ for the dense simulation."
        assert jnp.mod(1/dt0_dense, 1) == 0, "$1/ \Delta t$ dense % 1 should be 0. for correctly storing simulation "
        assert jnp.max(ts)-jnp.min(ts) >= jnp.min(ts)+H, f"Trial duration is shorter than receding horizon: trial duration ({jnp.max(ts)-jnp.min(ts)}) vs horizon ({H})"
        if dt0_internal is None:
            dt0_internal = Delta_t
        D_sys, D_control = self.internal_system.D_sys, self.internal_system.D_control
        E = self.internal_system.ensemble_size
        dense_steps = jnp.floor((ts[1]-ts[0]) / dt0_dense).astype(jnp.int32)
        ts_dense = jnp.arange(ts.min(), ts.max(), dt0_dense)
        n_controls = int(H/Delta_t) 

        Y = jnp.zeros((ts.shape[0], D_sys))
        X = jnp.zeros((ts_dense.shape[0], D_sys))
        U = jnp.zeros((ts_dense.shape[0], D_control))
        R = jnp.zeros((ts_dense.shape[0], 1))

        ts_h_init = jnp.linspace(ts[0], ts[0]+H, n_controls) 
        lambda0 = jnp.zeros((E, D_sys)) 
        
        start = time.time()
        X_init, l_init = None, None

        key, subkey = jr.split(key)

        lambda0, X_init, l_init = self.initial_pmp_optim(lambda0=lambda0, ts_h=ts_h_init, x0=x0, x_star=x_star, key=subkey)
        u_t, _, lambda_seq = self.lambda0_to_control(lambda0, x0, ts_h_init, inverse_standardize_u=inverse_standardize_u)
        
        key, subkey = jr.split(key)

        end = time.time()
        x_t = x0
        R_t = R[0]
        images, traj_times = [], []

        for i, t0 in enumerate(ts):
            key, subkey = jr.split(key)

            # initiate moving horizon, estimate state 
            ts_Delta_t = jnp.linspace(t0, t0+Delta_t, dense_steps)
            ts_h = jnp.linspace(t0+Delta_t, t0+H, n_controls) 
            ts_Delta_t_next = jnp.linspace(t0+Delta_t, t0+Delta_t*2, dense_steps)
            
            y_t = x_t + obs_noise*jr.normal(key=subkey, shape=x_t.shape)
            Y = Y.at[i].set(y_t)
            y_hat_t = y_t # state-estimation here


            # delay compensation
            one_step_ts = jnp.array([t0, t0+Delta_t])
            one_step_us = vmap(u_t.evaluate)(one_step_ts)
            
            y_hat_t_next = self.internal_system(
                                ts=jnp.tile(one_step_ts, (E,1)), 
                                y0=jnp.tile(y_hat_t,(E,1)), 
                                us=jnp.tile(one_step_us, (E,1,1)),
                                dt0=(ts_h[1]-ts_h[0])/2, 
                                return_ensemble_mean=False)[:,-1] 
         
            # optimize controls
            start = time.time()
            if i > -1:         
                (lambda0, shooting_states), stats = self.traj_optimizer(lambda0, ts=ts_h, x0=y_hat_t_next, x_star=x_star, dt0=dt0_internal, X_init=X_init, l_init=l_init)
                u_t_next, state_seq, lambda_seq = self.lambda0_to_control(lambda0, y_hat_t_next, ts_Delta_t_next, inverse_standardize_u=inverse_standardize_u)
                shooting_states = self.shift_segments_forward(segments=shooting_states, ts_h=ts_h, Delta_t=Delta_t, dt0_internal=dt0_internal)
                lambda0 = lambda_seq[:,-1,:]
                X_init, l_init = shooting_states[...,:D_sys], shooting_states[...,D_sys:]                
            else: 
                u_t_next = dfx.LinearInterpolation(ts=ts_dense, ys=jnp.zeros((ts_dense.shape[0], D_control)))
            end = time.time()
            traj_times.append((round(end-start,4), stats[0]['num_steps'].item()))

            # Simulate real system forward
            xs_R_dense = self.integrate_with_cost(y0=x_t, u=u_t, ts=ts_Delta_t, dt0=dt0_dense, R0=R_t)
            xs_dense, Rs = xs_R_dense[:,:D_sys], xs_R_dense[:,-1:]
            us_dense = vmap(u_t.evaluate)(ts_Delta_t)
            
            # Update true state, store intermediate simulation values
            start_idx, end_idx = i*dense_steps, jnp.minimum((i+1)*dense_steps, ts_dense.shape[0])
            X = X.at[start_idx:end_idx].set(xs_dense)
            U = U.at[start_idx:end_idx].set(us_dense)
            R = R.at[start_idx:end_idx].set(Rs)
            x_t = xs_dense[-1]
            R_t = R[end_idx-1]
            u_t = u_t_next

            # visualization
            plot_freq = 50
            if i%plot_freq==plot_freq-1:                
                print('Traj. opt. times: ', traj_times)
                traj_times = []

                fig, ax = self.visualize_horizon(ts=ts, ts_dense=ts_dense[:end_idx], ts_h=ts_h, i=i, dense_steps=dense_steps, t0=t0, Delta_t=Delta_t, 
                          x_t=x_t, x_star=x_star, X=X[:end_idx], U=U[:end_idx], lambda0=lambda0, y_hat_t_next=y_hat_t_next)
                plt.show()
                # plt.savefig(f'{save_path}/animation/frame_{i}.png')
                # plt.close()
                # images.append(imageio.imread(f'{save_path}/animation/frame_{i}.png'))


        # Save as GIF
        # if plot_freq > -1:
        #     imageio.mimsave(f'{save_path}/animation/animation.gif', images, fps=5, loop=0)
        R = R.at[-1].set(R[-1] + self.termination_cost(X[-1], U[-1]))

 
        return ts, ts_dense, X, Y, U, R

    def integrate_with_cost(self, y0:Array, u:Callable, ts:Array, dt0:float, R0:float=None):
        R0 = R0 if R0 is not None else jnp.array([0.])
        x0 = jnp.concatenate((y0, R0))
        saveat = dfx.SaveAt(ts=ts)
        step_size_controller = dfx.ConstantStepSize()
        sol = dfx.diffeqsolve(
                dfx.ODETerm(self.f_state_objective_real),
                solver=dfx.Dopri5(),
                saveat=saveat,
                t0=ts[0],
                t1=ts[-1],
                dt0=dt0,
                stepsize_controller=step_size_controller,
                y0=x0,
                args=u,
                ).ys
        return sol
    
    def control_policy(self, us:Array):
        """
            Takes a sequence of control actions us for different ensemble members.
            
            Args:
                us (E, T, D_control)

            Returns: 
                mean_us (T, D_control)
        """
        return jnp.mean(us, axis=0)
    
    def lambda0_to_control(self, lambda0:Array, x0:Array, ts_dense:Array, inverse_standardize_u:callable=None)->Array:
        """
            Convert a given initial condition of the co-states lambda(0), to a callable u(t) function.
            The system is solved forward for all the given input times, and then interpolated. 

            Args:
                lambda0 (E, D_sys) Array - initial costate values to solve for.
                x0 (D_sys) initial state to solve for.
                ts_dense (T,) dense time points to save the integrated values at.

            Return:
                u_callable (Callable, diffrax.GlobalInterpolation) control function u(t) that can be called with u_callable.evaluate(t) for a t in the range of ts_dense.
                state_sequence (T, D_sys) simulated values over the internal dynamics. 
                costate_sequence (T, D_sys) simulated values over the internal co-states.
        """
        E, _ = lambda0.shape
        if x0.ndim == 1:
            x0_tiled = jnp.tile(x0, (E,1))
        else:
            x0_tiled = x0
        ts_dense_tiled = jnp.tile(ts_dense, (E,1))
        
        X0 = jnp.concatenate((x0_tiled, lambda0), axis=-1)
        dt0 = ts_dense[1]-ts_dense[0]
        
        ys = self.traj_optimizer._solve_state_costate(X0, ts_dense_tiled, dt0=dt0) # (E, T, D_sys+D_sys)
        state_sequence, costate_sequence = ys[:,:,:self.real_system.D_sys], ys[:,:,self.real_system.D_sys:]
        
        # vmap over ensembles, vmap over timepoints.
        us_matrix = vmap(self.traj_optimizer.u_opt, in_axes=(1,1,1))(ts_dense_tiled, state_sequence, costate_sequence).transpose(1,0,2)
        us_matrix = self.control_policy(us_matrix)      # (E, T, D_control) -> (T, D_control)

        assert us_matrix.shape[0] == ts_dense.shape[0]

        u_callable = dfx.LinearInterpolation(ts=ts_dense, ys=us_matrix)


        return u_callable, state_sequence, costate_sequence

    def f_state_objective_real(self, t:float, x:Array, args):
        u = args.evaluate(t) #if isinstance(u, dfx.AbstractGlobalInterpolation) else u
        if self.real_system.D_sys == 1:
            u = u[0]
        xdot = self.real_system.f(t, x[:self.real_system.D_sys], args)
        R = self.state_cost(x[:self.real_system.D_sys], u)[None]

        return jnp.concatenate((xdot, R))
    
    def initial_pmp_optim(self, lambda0:Array, ts_h:Array, x0:Array, x_star:Array, key:PRNGKey=jr.PRNGKey(42)):

        """
        
        self.traj_optimizer(lambda0, ts=ts_h, x0=y_hat_t_next, x_star=x_star, dt0=dt0_internal, X_init=X_init, l_init=l_init)
        """
        X_init, l_init = None, None
        for i in range(50):
            key, subkey = jr.split(key)
            lambda0 = jr.normal(subkey, lambda0.shape)*(self.traj_optimizer.ub[0]/2)
            (lambda0, shooting_states), stats = self.traj_optimizer(lambda0, ts=ts_h, x0=x0, x_star=x_star, X_init=X_init, l_init=l_init)
            X_init, l_init = shooting_states[...,:self.real_system.D_sys], shooting_states[...,self.real_system.D_sys:] 
            (iteration_stats, result) = stats
            if result != optimistix.RESULTS.successful:
                print(f"Initial optimization (iteration {i}) not succesfull, continuing.")
            elif result == optimistix.RESULTS.successful:
                print(f'Succesful initial optimization at iteration {i}: ', lambda0)
                break
        print(f"Initial optimization over, starting trial with initial solution.")
        return lambda0, X_init, l_init
    
    def shift_segments_forward(self, segments:Array, ts_h:Array, Delta_t:float, dt0_internal:float):
        """
            Shift previously fitted segments forward Deltat time units. 
            Note: the initial start (t=t0) is not included. 

            Args:
                segments: (E,S-1, 2*D_x) where S is the number of total segments including the starts. Consists of states first (x), and then co-states (lambda)
                ts_h (S-1) array. Current starting times of the segments.
                Delta_t (float) duration to integrate for.
                dt0_internal (float) internal step size to use during solving.

            Return:
                shifted_segments (S-1, 2* D_x)

            where S is the number of segments, D_x is the system dimensionality and E is the ensemble size.
        """
        start_ts = jnp.array(jnp.array_split(ts_h, self.traj_optimizer.n_segments))[1:,0]  # [S-1,]

        subts = jax.vmap(lambda start_t: jnp.linspace(start_t, start_t+Delta_t, num=2))(start_ts) # [S-1, 2] - integration interval, from segmentstart -> (segmentstart + Delta_t).
        
        shifted_segments = jax.vmap(lambda segment_x0, ts_shift: self.traj_optimizer._solve_state_costate(segment_x0, ts_shift[None,...], dt0=dt0_internal),in_axes=(1,0), out_axes=(1))(segments, subts) # output shape: [E, S-1, 2, 2*D_x]
        
        return shifted_segments[:,:,-1,:]  # (E,S-1, 2*D_x)

    def visualize_horizon(self, ts:Array, ts_dense:Array, ts_h:Array, i:int, dense_steps:int, t0:float, Delta_t:float, 
                          x_t:Array, x_star:Array, X:Array, U:Array, lambda0:Array, y_hat_t_next:Array):
        fig = plt.figure(figsize=(20,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[0.4, 0.4, 0.2])  # Width ratios

        # Create the subplots
        ax0 = fig.add_subplot(gs[0])  # First subplot
        ax1 = fig.add_subplot(gs[1])  # Second subplot
        ax2 = fig.add_subplot(gs[2])  # Third subplot
        ax = [ax0, ax1, ax2]

        # ax[0].scatter(ts[:i], U[::dense_steps][:i])
        H = ts_h.max() - ts_h.min()
        ax[0].plot(ts_dense[:i*dense_steps], U[:i*dense_steps], color='purple')
        dense_ts_h = jnp.linspace(ts_h.min(), ts_h.max(), ts_h.shape[0]*10)

        u_t_planned, xs_planned, _ = self.lambda0_to_control(lambda0, y_hat_t_next, ts_dense=dense_ts_h)
        xs_planned = jnp.mean(xs_planned, axis=0)
        us_planned = vmap(u_t_planned.evaluate)(dense_ts_h)

        xs_planned_true = self.real_system(y0=y_hat_t_next.mean(axis=0), ts=dense_ts_h, u=us_planned)


        ax[0].plot(dense_ts_h, us_planned, color='purple') # planned control signal
        ax[0].axvline(x=t0, color='black')
        ax[0].set_title(f'u(t)')
        ax[0].set_xlim((ts.min(), ts.max()))
        # ax[0].fill_between(ts, y1=self.traj_optimizer.ub, y2=self.traj_optimizer.lb, where=np.logical_and(ts >= t0, ts<=t0+H ), facecolor='gray', alpha=.2)

        for n in range(self.real_system.D_sys):
            dim_color = f'C{int(n*2)}'
            ax[1].scatter(ts[:i], X[::dense_steps][:i,n], color=dim_color)
            ax[1].plot(ts_dense[:i*dense_steps], X[:i*dense_steps,n], color=dim_color)
            ax[1].scatter(t0+Delta_t, x_t[n], color='gold')

            ax[1].plot(dense_ts_h, xs_planned[:,n], color=dim_color, linestyle=':', label='Internal model prediction' if n==0 else None) # planned internal trajectory
            ax[1].plot(dense_ts_h, xs_planned_true[:,n], color=dim_color,  alpha=0.6,label='True trajectory' if n==0 else None) # planned true trajectory
            ax[1].axhline(y=x_star[n], linestyle=':', color='black')


        min, max = jnp.minimum(xs_planned.min(), X.min()), jnp.maximum(xs_planned.max(), X.max())
        ax[1].fill_between(ts, y1=min, y2=max, where=np.logical_and(ts >= t0, ts<=t0+H ), facecolor='gray', alpha=.2)
        ax[1].axvline(x=t0, color='black')
        ax[1].set_xlim((ts.min(), ts.max()))
        ax[1].set_title('states x(t)')
        ax[1].legend()

        data_vis = DiffEqDataset(ts=ts_dense[None,...], ys=X[None,...])
        self.real_system.visualize_phaseplane(X_range=self.real_system.X_range_phaseplane, data=data_vis, ax=ax[2])
        plt.tight_layout()
        return fig, ax

