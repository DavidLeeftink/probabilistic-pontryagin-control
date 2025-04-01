"""
Copyright (c) 2025 David Leeftink
 
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""
import jax
import jax.nn as jnn
import jax.random as jr
from jax.random import PRNGKey
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import ppc
from ppc.nn.nnvectorfield import NeuralVectorField, EnsembleNeuralVectorField
from dataclasses import dataclass
from jaxtyping import Array, Float
from beartype.typing import Optional, Tuple, Callable
from types import SimpleNamespace

class NeuralODE(eqx.Module):
    """
    Neural ODE class.

    Parameters:
        vectorfield (NeuralVectorfield)
        solver (Abstract)
        stepsize_controller (dfx.AbstractStepSizeController)
        dt0 (float) initial step size during solving. If none are given, use the time difference between the first two observations.
        encoder (eqx.Module) 
        decoder (eqx.Module)
        D_sys (int) dimensionality of the system
        D_control (int) dimensionality of the control input
        control_interpolater (str) String referring to the interpolation function that takes a discrete timepoints and control values, 
                                and interpolates them to a continuous-time function. Currently, only 'linear' is implemented.
    """
    vectorfield:NeuralVectorField
    solver:dfx.AbstractSolver
    stepsize_controller:dfx.AbstractStepSizeController
    obs_noise_raw:Array = None
    x0_mean:Array = None
    x0_diag_raw:Array = None
    dt0:float = None
    encoder:eqx.Module = None
    decoder:eqx.Module = None
    D_sys:int = None
    D_control:int = None
    control_interpolator:str = 'linear'

    def __post_init__(self)-> None:
        if self.D_sys is None:
            self.D_sys = self.vectorfield.D_sys
        if self.D_control is None:
            self.D_control = self.vectorfield.D_control
        if self.obs_noise_raw is None:
            self.obs_noise_raw = jnp.ones((self.D_sys,))*-1
        if self.x0_mean is None:
            self.x0_mean = jnp.zeros((self.D_sys,))
        if self.x0_diag_raw is None:
            self.x0_diag_raw = jnp.ones((self.D_sys,))*-1

    def __call__(self, ts:Array, y0:Array=None, u=None, dt0:float=None, u_timepoints:Array=None, ) -> Array:
        r""" Solve the ODE.

        Args:
            ts (Array) (T,) time points to solve over.
            y0 (Array) (D_sys,) initial states to solve for. If used with encoder, then this has is given as input to the encoder.
                                If not given, sample an initial condition from the learned p(x(0)) distribution.
            u (Callable) control function u(t).
            dt0 (float) initial step size for adaptive solver. If left empty, this will default to the first time different in ts.
            u_timepoints (T,) Array - if a matrix is given for u, then u_timepoints should be the corresponding time points at which these controls are executed.

        Return:
            ys (T,D_sys) integrated state trajectory.
        """  
        t0, tf = ts[0], ts[-1]
        dt0 = ts[1]-ts[0] if dt0 is None else dt0
        saveat = dfx.SaveAt(ts=ts)

        if u is not None:
            if isinstance(u, dfx.AbstractGlobalInterpolation) or isinstance(u, Callable):
                pass
            elif u_timepoints is not None:
                u = self.interpolate_controls(ts=u_timepoints, us=u)
            else:
                u = self.interpolate_controls(ts=ts, us=u)
            f = lambda t, x, args: self.f(t, x, u)
        else:
            f = self.f
        terms = dfx.ODETerm(f) 

        # encode initial condition y0 to latent space
        if self.encoder is not None:  
            y0 = self.encoder(0., y0)

        # solve integral (possibly in latent space)
        if isinstance(self.solver, dfx.AbstractSolver):
            ys_pred = dfx.diffeqsolve(
                            terms=terms,
                            solver=self.solver,
                            t0=t0,
                            t1=tf,
                            dt0=dt0,
                            y0=y0,
                            args=u,
                            stepsize_controller=self.stepsize_controller,
                            saveat=saveat,).ys
        else:   
            ys_pred = self.odeint(f=self.vectorfield, y0=y0, ts=ts[0])
        
        # Decode latent integrated ODE
        if self.decoder is not None:
            ys_pred = jax.vmap(lambda x_: self.decoder(0., x_), in_axes=(0))(ys_pred)

        return ys_pred
    
    def interpolate_controls(self, us:Array, ts:Array):
        """
            Apply linear interpolation to a set of control signals at fixed timepoins.
            Assumes single trajectory.

        Args:
            us (Array) control inputs at given timepoints
            ts (Array) time points. 
        
        Return: 
            u (dfx.Interpolation class) controller that gives u(t) by calling u.evaluate(t)
        """        
        if us is None:
            u = SimpleNamespace(evaluate=lambda t: jnp.zeros((self.D_control,)))
        else:
            if self.control_interpolator == 'linear':
                u = dfx.LinearInterpolation(ts=ts, ys=us)
            else:
                raise NotImplementedError("Only linear interpolation added so far")
        return u
    
    def f(self, t, x, u=None):
        return self.vectorfield(t, x, u)
        
    def obs_noise(self):
        return jax.nn.softplus(self.obs_noise_raw)
    
    def x0_diag(self):
        return jax.nn.softplus(self.x0_diag_raw)
          

class EnsembleNeuralODE(NeuralODE):
    ensemble_size:int = None    
    
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ensemble_size is None:
            self.ensemble_size = self.vectorfield.ensemble_size

    def __call__(self, ts:Array, y0:Array, us:Array=None, dt0:float=None, u_timepoints:Array=None, return_ensemble_mean=False, jump_ts:Array=None) -> Array:
        r""" Solve the ODE according to the ensemble of neural ODEs.

        Args:
            ts (Array) (E, T) time points to solve over.
            y0 (Array) (E, D_sys) initial states to solve for. If used with encoder, then this has is given as input to the encoder.
            us (Array) (E, T, D_control) control function values.
            dt0 (float) initial step size for adaptive solver. If left empty, this will default to the first time different in ts.
            u_timepoints (E, T) Array - if a matrix is given for u, then u_timepoints should be the corresponding time points at which these controls are executed.
            return_ensemble_mean (Bool) if true, average over all the ensemble members dynamics. Used when combined with optimizers that optimize the mean of all trajectories.
            jump_ts (E,T_jump) Array Timepoints at which the control inputs u(t) are known to have discontinuous jumps. Tihs helps the automatic step size solver. 

        Return:
            ys (E, T,D_sys) integrated state trajectory if return_ensemble_mean  is False, 
                (T, D_sys) if return_ensemble_mean is True.
        """  
        t0, tf = ts[0,0], ts[0,-1]
        dt0 = ts[0,1]-ts[0,0] if dt0 is None else dt0
        saveat = dfx.SaveAt(ts=ts[0])
        terms = dfx.ODETerm(self.f) 
        args = (ts, us) if u_timepoints is None else (u_timepoints, us)

        stepsize_controller = self.stepsize_controller

        # solve integral (possibly in latent space)
        if isinstance(self.solver, dfx.AbstractSolver):
            ys_pred = dfx.diffeqsolve(
                            terms=terms,
                            solver=self.solver,
                            t0=t0,
                            t1=tf,
                            dt0=dt0,
                            y0=y0,
                            args=args,
                            stepsize_controller=stepsize_controller,
                            max_steps=4096*2,
                            saveat=saveat,).ys.transpose(1,0,2)
    
        if return_ensemble_mean:
            ys_pred = jnp.mean(ys_pred, axis=0)
        return ys_pred

def mse_loss_ensemble(ensemble_model, ensemble_datasets, key=None):
    """ Apply a forward pass and compute the mean squared error.

    Args: 
        ensemble model (Module) vmapped trainable model for E ensemble members.
        ensemble_dataset (list of E DiffEqDatasets) 
            Contains:   ys [E, N_e, T, D_sys] 
                        ts [E, N_e, T] 
                        us [E, N_e, T, D_control] 
            where E is the number of ensemble members,
                N_e is the number of trajectories per ensemble member,
                T is the number of time steps. 

    Return:
        mse (Float)
    """
    ys, ts, us, ts_dense = ensemble_datasets.ys, ensemble_datasets.ts, ensemble_datasets.us, ensemble_datasets.ts_dense
    dt0 = ensemble_model.dt0
    y0 = ys[:,:,0,:]
    
    """        
        todo: ensemble NODE class does not parallize ts across ensemble, 
            it assumes all the time points of observations are the same during integration. 
    """
    # if ts_dense is None:
    #     ys_pred = jax.vmap(ensemble_model.__call__, in_axes=(1,1,1,None,None,None), out_axes=(1))(ts, y0, us, dt0, None, False) # (ts, y0, us, dt0, u_timepoints, return_ensemble_mean, jump_ts)
    # else:
    ys_pred = jax.vmap(ensemble_model.__call__, in_axes=(1,1,1,None,1, None), out_axes=(1))(ts, y0, us, dt0, ts_dense, False)

    def per_member(ys, random_idx):
        def per_traj(ys, random_idx):
            return ys[random_idx,:]
        return jax.vmap(per_traj, in_axes=(0,0))(ys, random_idx)
    
    def get_sampled_datapoints(ys, random_idx):
        assert ys.shape[:3] == random_idx.shape[:3], f"Indices and dataset should be of the same size but are instead: {ys.shape} (data) vs {random_idx.shape} (indices)"
        return jax.vmap(per_member, in_axes=(0,0))(ys, random_idx)


    # bootstrapping the predictions.
    E, N_e, T, D_sys = ys.shape
    bootstrapped_idx = jr.choice(key=key,a=jnp.arange(T), shape=(E,N_e,T), replace=True)
    bootstrapped_ys = get_sampled_datapoints(ys, bootstrapped_idx)
    bootstrapped_ys_pred = get_sampled_datapoints(ys_pred, bootstrapped_idx)

    mse_term = jnp.mean((bootstrapped_ys - bootstrapped_ys_pred) ** 2)
    # print(bootstrapped_ys.shape, bootstrapped_ys_pred.shape, mse_term)
    # mse_term = jnp.mean((ys - ys_pred)** 2)

    return mse_term