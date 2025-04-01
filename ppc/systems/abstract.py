"""
Copyright (c) 2025 David Leeftink
 
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
import diffrax as dfx
from diffrax import AbstractGlobalInterpolation
import equinox as eqx
from beartype.typing import Callable, List, Optional, Union
from jaxtyping import Float, Num
from types import SimpleNamespace
from jaxtyping import Array, Float
import ppc
from ppc.dataset import DiffEqDataset


class AbstractSystem(eqx.Module):
    r"""Abstract class for systems. Contains basic properties and plotting functionalities. """
    D_sys:int = None
    D_control:int = None
    solver:Callable = None
    name:str = None
    control_interpolator:str = 'linear'

     
    def __post_init__(self):
        assert self.D_sys is not None, "System dimensionality not specificed (D_sys attribute)."
        assert self.solver is not None, "No numerical solver given."
    
    def __call__(self, y0:Array, ts:Array, u=None, dt0:float=None, u_timepoints:Array=None, stepsize_controller:dfx.AbstractStepSizeController=dfx.ConstantStepSize())-> Array:
        """
            Solve x(t) = x(0) + \int_0^T f(x(\tau)) d\tau

        Args:
            y0 (Array) (D_sys,) initial states to solve for
            ts (Array) (T,) time points to solve over.
            u (Callable) control function u(t).
            dt0 (float) initial step size for adaptive solver. If left empty, this will default to the first time different in ts.
            u_timepoints (T,) Array - if a matrix is given for u, then u_timepoints should be the corresponding time points at which these controls are executed.

        Return:
            ys (T,D_sys) integrated state trajectory.
        """
        t0, tf = ts[0], ts[-1]
        dt0 = ts[1]-ts[0] if dt0 is None else dt0
        saveat = dfx.SaveAt(ts=ts)
        terms = dfx.ODETerm(self.f) 

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

        return dfx.diffeqsolve(terms, 
                                self.solver, 
                                t0=t0, 
                                t1=tf, 
                                dt0=dt0, 
                                y0=y0, 
                                args=u, 
                                stepsize_controller=stepsize_controller,
                                saveat=saveat, 
                                max_steps=1_000_000,
                                ).ys  

    def f(self, t:float, x:Float[Array, "n D_sys"], u=None):
        """
            Abstract method for differential ODE term f(x,u)

        Args:
            t (float) : time of evaluation, used for time-varying systems or for control inputs.
            x (Array) : state at which the ODE is evaluated
            u (dict) Control interpolator. Can be called with 'u.evaluate(t)'. 
        
        Returns
        --------
            \dot{x} (n, D_sys) - Float Array.
        """
        raise NotImplementedError("Abstract method should not be called")

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
        assert len(ts.shape) == 1, "Shape of time points should be: (number of timepoints)"
                
        if us is None:
            u = SimpleNamespace(evaluate=lambda t: jnp.zeros((self.D_control,)))
        else:
            assert len(us.shape) == 2, "Shape of controls should be: (number of timepoints, number of control dimensions)"
            assert ts.shape[0]==us.shape[0], "Number of control sequences does not match the number of time point sequences"
            if self.control_interpolator == 'linear':
                u = dfx.LinearInterpolation(ts=ts, ys=us)
            else:
                raise NotImplementedError("Only linear interpolation added so far")
        return u

    def generate_synthetic_data(self, 
                                key:PRNGKey,
                                num_sequences:int, 
                                dt0:float,
                                ts:Float[Array, "N_seq T"], 
                                obs_stddev:Float[Array, "N_seq"],
                                x0_distribution:Callable,
                                us:Array = None,
                                ts_dense:Float[Array, "N_Seq T_Dense"]=None,
                                T_scalar:float=None,
                                standardize_at_initialisation:bool = True,
                                ):
        """
        Args:
            key (jr.PRNGKey)  
            num_sequences (int) Number of sequences to generate
            dt0 (float) initial step size to solve with.
            ts (Float Array) N x T matrix. The time points are assumed different for each sequence, but of the same length.
            obs_stddev (Array) Observation standard deviation, used for adding i.i.d. Gaussian noise
            x0_distribution (callable) Function that generates initial states of the system.
            T_scalar (float) optional scalar to increase or decrease the time units to a desired scale
            us()
            standardize_at_initialization (bool) if True, return a dataset that is standardized but also a function to transform back to the original scale.

        Return: 
            data (DiffEqDataset) Simulated dataset with Gaussian i.i.d. noise
            true_y0s (Array) Ground truth initial states that were used to similate the data with.

        """
        assert len(ts.shape) == 2, f"The given time points are assumed to be of shape ()"
        assert ts.shape[0] == num_sequences, f"Number of sequences ({num_sequences}) does not match the number of given time inputs ({ts.shape[0]}). If all sequences share the same time points, please give N copies as input."
        assert isinstance(x0_distribution, Callable), f"Given x0 distribution is not callable ({x0_distribution})"
        assert x0_distribution(key).shape[0] == self.D_sys, f"Given x0 distribution gives initial states of shape ({x0_distribution(1, key).shape[0]}), while the system dimensionality is: {self.D_sys} "
        
        if us is not None:
            assert len(us.shape) == 3, f"Control inputs are ssumed to be of the shape: (num_sequences, num_timepoints, num_controldimensions). Given control matrix instead has shape: {us.shape}"
            assert us.shape[0] == ts.shape[0] , "First dimension of control input us (number of sequences) should have the same dimension as first dimension of the time inputs ts."
            if ts_dense is None:
                assert us.shape[1] == ts.shape[-1], "Second dimension of control input us (number of timepoints) should have the same dimension as last dimension of the time inputs ts."
            else:
                assert us.shape[1] == ts_dense.shape[-1], "Second dimension of control input us (number of timepoints) should have the same dimension as last dimension of the time inputs dense_ts."
            assert us.shape[-1] == self.D_control, f"Given control matrix us ({us.shape[-1]}) does not have the control dimensionality corresponding to this sytem ({self.D_control})"
            assert self.D_control > 0, "The system's control dimensionality is set to 0, meaning no controls us should be given. If the system should be controlled, the dimensionality has to be set during initialization."
        _, T = ts.shape
        ys = jnp.zeros((num_sequences, T, self.D_sys)) # todo: this line can be removed.
        # u_list = jnp.array([self.interpolate_controls(us[s], ts[s]) for s in jnp.arange(us.shape[0])])

        keys = jr.split(key, num_sequences+1)
        y0s = jnp.array([x0_distribution(keys[n]) for n in range(num_sequences)])
        if ts_dense is None:
            ys = vmap(self, in_axes=(0,0,0,None))(y0s, ts, us, dt0)
        else:
            ys = vmap(self, in_axes=(0,0,0,None,0))(y0s, ts, us, dt0, ts_dense)
        ys += jr.normal(keys[-1], shape=ys.shape) * obs_stddev

        return DiffEqDataset(ts=ts, ys=ys, us=us, ts_dense=ts_dense, standardize_at_initialisation=standardize_at_initialisation, T_scalar = T_scalar), y0s
    
    def visualize_longitunidal(self, test_ts:Array, true_y0s:Array, data:DiffEqDataset=None, ax=None, S:float=1.):
        """
        PLot observed states over time and control inputs.

        Args:
            test_ts (Array) timepoitns to visualize, and solve true ODE over.
            true_y0s (Array) N,D Array with ground truth initial conditions. 
            data (DiffEqDataset) Observations to visualize.
            ax (Matplotlib axes) If given, use these axis to plot. Must be of size D_sys+D_control.

        Return:
            ax (Matplotlib axes) 
        """
        num_trials = data.n if data is not None else 1
        test_signal = []

        for i in range(num_trials):
            saveat = dfx.SaveAt(ts=test_ts[i])
            t0, tf = test_ts[i,0], test_ts[i,-1]
            dt0 = test_ts[i,1]-test_ts[i,0]

            if data is not None and data.us is not None:
                if data.ts_dense is None:
                    u = self.interpolate_controls(us=data.us[i], ts=data.ts[i])
                else:
                    u = self.interpolate_controls(us=data.us[i], ts=data.ts_dense[i])
                f = lambda t, x, args: self.f(t, x, u)
            else:
                f = self.f
        
            sol = dfx.diffeqsolve(
                dfx.ODETerm(f), self.solver, t0, tf, dt0, true_y0s[i], saveat=saveat, max_steps=100_000,
            )
            test_signal.append(sol.ys)

        test_signal = jnp.array(test_signal)
        if ax is None:
            fig, ax = plt.subplots(self.D_sys+self.D_control,num_trials, figsize=(12,12*num_trials), sharex=True, sharey=True)
            ax = ax[None,:] if num_trials == 1 else ax
        
        if data is not None:
            ys, ts = data.ys, data.ts
            if data._original_ys_mean is not None and data._original_ys_std is not None:
                ys = data.inverse_standardize(ys)
            if data.T_scalar is not None:
                ts = data.inverse_scale_timepoints(ts)

            if data.us is not None:
                us = data.us
            else:
                us = jnp.zeros(test_ts.shape)

        for s in range(num_trials):
            # visualize states over time for an observed sequence
            for i in jnp.arange(self.D_sys):
                ax[s,i].set_title(f'$x_{i+1}$')
                ax[s,i].set_xlabel('t')
                ax[s,i].set_ylabel(f'$f(x_{i+1})$')
                ax[s,i].plot(test_ts[s], test_signal[s,:,i], color='black', label='True function', alpha=0.4)
                ax[s,i].set_xlim((test_ts.min(), test_ts.max()))
                if data is not None:
                    ax[s,i].scatter(ts[s], ys[s,:,i], color=f'black', marker='x', s=30, alpha=0.5)

            # visualize control inputs u(t) over time for an observed sequence
            for j in jnp.arange(self.D_sys, self.D_sys+self.D_control):
                if data is not None:
                    print(s, j, ax.shape)
                    if data.ts_dense is None:
                        jnp.concatenate((us[s], us[s]))
                        ax[s,j].plot(ts[s], us[s], color='purple', label='u(t)', alpha=0.5)
                    else:
                        ax[s,j].plot(ts[s], data.us[s], color='purple', label='u(t)', alpha=0.5)
                # ax[s,j].scatter(ts[s], us[s], color='purple', alpha=0.5)
                ax[s,j].set_xlim((test_ts[s].min(), test_ts[s].max()))
                ax[s,i].set_title(f'$u_{j+1}(t)$')
                ax[s,i].set_xlabel('t')
                ax[s,i].set_ylabel(f'$u_{j+1}(t)$')
          

    
        fig.suptitle(f'Time series of {self.name} system')
        plt.tight_layout()
        return ax
    
    
class OneDimensionalSystem(AbstractSystem):
    """
        Abstract class for system visualizations of one-dimensional systems.
    """
    def __post_init__(self):
        assert self.D_sys == 1, f"One dimensional system class is used but the system has dimensionality {self.D_sys}"
        super().__post_init__()

    def visualize_phaseplane(self, 
                             f_true:Callable, 
                             x_range:Float[Array, "2"], 
                             caption:str=None,
                             ax=None,
                             arrow_locs:Float[Array, "A"]=None,
                             ):
        """
            Plot a 1D phaseplane, including the null-cline at f(x)=0.

        Args:
            f_true (Callable) True 1D ODE function
            x_range
        """
        # assert f_true(0, x_range).shape[-1] == 1, f"Given true system function is not 1 dimensional but {f_true(0, x_range).shape[-1]}"
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(4,4))
        
        return ax


class TwoDimensionalSystem(AbstractSystem):

    def __post_init__(self):
        assert self.D_sys == 2, f"Two dimensional system class is used but the system has dimensionality {self.D_sys}"
        super().__post_init__()

    def visualize_phaseplane(self, X_range, ax=None, data:DiffEqDataset=None):
        """
        Create stream plot of the two dimensional system according to the given phase plane range.

        Args:
            x_range (2, 2) for each dimension, this should contain a min and max value of the axis.
            ax (matplotlib axis) if given, use this axis to visualize the stream plot in.
            ys (Array) If given, add the observed trajectory to the phase plane.

        Return:
            ax (matplotlib axis) axis that contains the stream plot.
        """
        assert X_range.shape[0] == self.D_sys, f"System dimensionality ({self.D_sys}) does no match with "
        assert X_range.shape[1] == 2, f"Last axis of given X_range should contain a min and a max value, but has instead shape: {X_range.shape[1]}"
        
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(4,4))
        X_linspace = jnp.linspace(X_range[0,0], X_range[0,1], 100)[:,None]
        Y_linspace = jnp.linspace(X_range[1,0], X_range[1,1], 100)[:,None]

        X, Y = np.meshgrid(X_linspace.squeeze(), Y_linspace.squeeze())
        U = self.f(0,[X, Y])[0]
        V = self.f(0,[X, Y])[1]
        ax.streamplot(X,Y, U,V,color='gray')

        if data is not None:
            ys = data.inverse_standardize(data.ys) if data.standardize_at_initialisation else data.ys
            for sequence in ys:
                ax.plot(sequence[:,0], sequence[:,1], alpha=1., linewidth=5)
        return ax


class ThreeDimensionalSystem(AbstractSystem):

    def __post_init__(self):
        super().__post_init__()
        assert self.D_sys == 3, f"Three dimensional system class is used but the system has dimensionality {self.D_sys}"

    def visualize_phaseplane(self):
        pass 
    

class MultiDimensionalSystem(AbstractSystem):

    def __post_init__(self):
        super().__post_init__()