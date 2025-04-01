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
from equinox import Module
from beartype.typing import Callable, List, Optional, Union
from jaxtyping import Float, Num
from types import SimpleNamespace
from jaxtyping import Array, Float
import ppc
from ppc.dataset import DiffEqDataset
from ppc.systems.abstract import AbstractSystem
from ppc.systems.nonlinear_dynamics import VanDerPol, DuffingOscillator
from ppc.systems.classic_control import angle_normalize, Cartpole
from ppc.control.cost_functions import QuadraticCost


class AbstractControlTask(Module):
    r"""Abstract class for control systems, that contains the true system, as well as the objective.
        Intended to have one task, easily shared acorss experiments and models.

        real_system (AbstractSystem) the system to control, that has to be learned, where
                                    D_sys is the system's dimensionality,
                                    D_control is the dimensionality of the control vector.
        R (D_control, D_control) Array - PSD matrix of the control cost
        Q (D_sys, D_sys) Array - PSD matrix of the state cost
        x_star (D_sys,) Array goal state       (assumed to be static for the time being)
        u_star (D_control,) Array -  goal control vector.
        lb (D_control,) Array or (T_h, D_control) minimum allowed control input u(t)
        ub (D_control,) Array or (T_h, D_control) maximum allowed control input u(t)
        state_cost (QuadraticCost) state cost l(x,u). If not given, one is constructed based on the given R and Q matrices.
        state_cost (QuadraticCost) termiantion cost Phi(x,u), If not given, this is constructed based on the given Q_f matrix.
        t0 (float) - start time of each episode
        tf (float) - and time of each episode
        Delta_t (float) - measuring interval.
        Q_f (D_sys, D_sys) Array - PSD matrix of the state cost at the termination state.
        name (str) name given to the control setting

    """

    real_system:AbstractSystem
    D_sys:int
    D_control:int
    R:Array
    Q:Array
    x_star:Array
    u_star:Array
    y0:Array
    t0:float
    tf:float
    Delta_t:float
    measurement_noise_std:float
    lb:Array
    ub:Array
    state_cost:QuadraticCost = None
    termination_cost:QuadraticCost=None
    H:float = None 
    Q_f:Array = None
    name:str = None
    S:float = 1.
     
    def __check_init__(self)-> None:
        assert jnp.sum(jnp.linalg.cholesky(self.R))> -jnp.inf, "Cost matrix R failed positive-semi definite test of positive eigenvalues." # cholesky call will fail if not PSD
        assert jnp.sum(jnp.linalg.cholesky(self.Q))> -jnp.inf, "Cost matrix Q failed positive-semi definite test of positive eigenvalues." # cholesky call will fail if not PSD
        if self.Q_f is not None:
            assert jnp.sum(jnp.linalg.cholesky(self.Q_f))> -jnp.inf, "Cost matrix R failed positive-semi definite test of positive eigenvalues." # cholesky call will fail if not PSD

    def solve_true_system(self)-> None:
        raise NotImplementedError


class VanderPolStabilization(AbstractControlTask):

    def __init__(self):
        self.D_sys, self.D_control = 2, 1
        self.real_system = VanDerPol(solver=dfx.Dopri8(), D_sys=self.D_sys, D_control=self.D_control)
        
        self.R = jnp.eye(self.D_control)*.5
        self.Q = jnp.eye(self.D_sys)*1.
        self.Q_f = jnp.eye(self.D_sys)*1.

        self.x_star = jnp.zeros((self.real_system.D_sys))
        self.u_star = jnp.zeros((self.real_system.D_control,))
        
        self.lb = -2.
        self.ub = 2.

        self.y0 = jnp.array([1., 1.]) 
        self.t0 = 0.
        self.tf = 5.
        self.Delta_t = 0.05
        self.measurement_noise_std = 0.01 #  standard deviation of Gaussian noise added on top of environment
        
        self.state_cost = QuadraticCost(x_star=self.x_star, u_star=self.u_star, R=self.R, Q=self.Q)
        self.termination_cost = QuadraticCost(x_star=self.x_star, u_star=self.u_star, Q=self.Q_f, R=0.*jnp.eye(self.D_control))

        self.name = 'Van der Pol stabilization control task'
       

    def get_initial_condition(self, key=None)->Array:
        return self.y0
    

class DuffingStabilization(AbstractControlTask):

    def __init__(self):
        self.D_sys, self.D_control = 2, 1
        self.real_system = DuffingOscillator(solver=dfx.Dopri8(), D_sys=self.D_sys, D_control=self.D_control)
        
        self.R = jnp.eye(self.D_control)*1.
        self.Q = jnp.eye(self.D_sys)*5.
        self.Q_f = jnp.eye(self.D_sys)*5.

        self.x_star = jnp.array([0., 0.])
        self.u_star = jnp.zeros((self.real_system.D_control,))
        
        self.lb = -2.
        self.ub = 2.

        self.y0 = jnp.array([1.5, 1.]) 
        self.t0 = 0.
        self.tf = 5.
        self.Delta_t = 0.05
        self.measurement_noise_std = 0.01 #  standard deviation of Gaussian noise added on top of environment
        
        self.state_cost = QuadraticCost(x_star=self.x_star, u_star=self.u_star, R=self.R, Q=self.Q)
        self.termination_cost = QuadraticCost(x_star=self.x_star, u_star=self.u_star, Q=self.Q_f, R=0.*jnp.eye(self.D_control))

        self.name = 'Van der Pol stabilization control task'
       

    def get_initial_condition(self, key=None)->Array:
        return self.y0
    

class CartPoleTask(AbstractControlTask):
    """
        Based on Brunton, p. 301. 
    """
    def __init__(self):
        self.D_sys, self.D_control = 4, 1
        self.real_system = Cartpole(solver=dfx.Dopri8(), D_sys=self.D_sys, D_control=self.D_control)
        
        self.R = jnp.eye(self.D_control)*.05*5.
        self.Q = jnp.eye(self.D_sys)*jnp.array([1., 1., .1, .1])
        self.Q_f = jnp.eye(self.D_sys)*jnp.array([1., 5., 1., 1.])

        self.x_star = jnp.array([1., jnp.pi, 0., 0.])
        self.u_star = jnp.zeros((self.real_system.D_control,))
        
        self.lb = jnp.array([-20./5.]) # max torque from open AI gym
        self.ub = jnp.array([20./5.])  # negative max torque from open AI gym 

        # self.y0 = jnp.array([0., jnp.pi-0.4, 0., 0.])
        self.y0 = jnp.array([0.,0., 0., 0.]) #+ jr.normal(jr.PRNGKey(43), jnp.zeros((4)).shape)
        self.t0 = 0.
        self.tf = 5.
        self.Delta_t = .02
        self.measurement_noise_std = 0.0 #  standard deviation of Gaussian noise added on top of environment
        
        def angle_normalize_cartpole(x):
            theta = x[1] if x.ndim == 1 else x[:,1]
            diff = theta - jnp.pi
            normalized_diff = jnp.arctan2(jnp.sin(diff), jnp.cos(diff))
            theta_adjusted = normalized_diff + jnp.pi
            return x.at[1].set(theta_adjusted) if x.ndim == 1 else x.at[:,1].set(theta_adjusted)
            
        self.state_cost = QuadraticCost(x_star=self.x_star, u_star=self.u_star, R=self.R, Q=self.Q, transform=angle_normalize_cartpole)
        self.termination_cost = QuadraticCost(x_star=self.x_star, u_star=self.u_star, Q=self.Q_f, R=0.*jnp.eye(self.D_control), transform=angle_normalize_cartpole)

        self.name = 'Inverted Pendulum'

    def get_initial_condition(self, key=None)->Array:
        return self.y0 #+ jr.normal(key, self.y0.shape)*0.2
