"""
Copyright (c) 2025 David Leeftink
 
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""
from dataclasses import dataclass
import jax.numpy as jnp
from beartype.typing import (
    Callable,
    List,
    Optional,
    Union,
)
from types import SimpleNamespace
import jax.numpy as jnp
from jax import vmap
from jax.random import PRNGKey
from jaxtyping import (
    Float,
    Num,
)
import diffrax as d
from jaxtyping import Array, Float   
from ppc.systems.abstract import *    


class VanDerPol(TwoDimensionalSystem):
    name:str= r'Van der Pol'
    mu:float = 1.5
    D_sys:int = 2
    D_control:int = 0 # For controlled, set to 1 at initialization.
    X_range_phaseplane=jnp.array([[-3., 3.],[-4., 4.]])

    def f(self, t:float, x:Float[Array, "D_sys"], u=None):
        """
            Van der Pol oscillator:

                \dot{x} = y
                \dot{y} = μ ( 1-x^2) y - x  + u(t)
    
        Args:
            t (float) : time of evaluation, only used for time-varying systems.
            x (Array) : state at which the ODE is evaluated
            u (Callable): control input interpolation object. call as 'u(t,x)', where t is a float representing the time point.

        Returns
        --------
            \dot{x} (n, D_sys) - Float Array.
        """
        assert len(x) == self.D_sys
        if u is not None:            
            u = u.evaluate(t) if isinstance(u, dfx.AbstractGlobalInterpolation) else u
            u = u[0]
        else:
            u = 0.
            
        x, w = x[0], x[1]
        g = lambda x: x
        return jnp.array([w - self.mu * ( x**3 / 3 - x), 
                          -g(x) + u]) # F form 
    
    def visualize_phaseplane(self, X_range, ax=None, data=None):
        ax = super().visualize_phaseplane(X_range, ax, data)
        ax.set_title(f'{self.name} - phase plane')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        return ax
    

class VanDerPolDerivative(TwoDimensionalSystem):
    name:str= r'Van der Pol derivative'
    mu:float = 1.5
    D_sys:int = 2
    D_control:int = 0 # For controlled, this should be set to 1 during initialization.

    def f(self, t:float, x:Float[Array, "D_sys"], u=None):
        """
            Van der Pol oscillator:

                \dot{x} = y
                \dot{y} = μ ( 1-x^2) y - x  + u(t)
    
        Args:
            t (float) : time of evaluation, only used for time-varying systems.
            x (Array) : state at which the ODE is evaluated
            u (Callable): control input interpolation object. call as 'u(t, x)', where t is a float representing the time point.

        Returns
        --------
            \dot{x} (n, D_sys) - Float Array.
        """
        assert len(x) == self.D_sys
        if u is not None:
            u1 = u(t,x)[0]
        else:
            u1 = 0.
        x, w = x[0], x[1]
        return jnp.array([w, -x + self.mu * w*(1-x**2) + u1])  # f(x) form
    
    def visualize_phaseplane(self, X_range, ax=None, data=None):
        ax = super().visualize_phaseplane(X_range, ax, data)
        ax.set_title(f'{self.name} - phase plane')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        return ax
    

class FitzhughNagumo(TwoDimensionalSystem):
    name:str=r'Fitzhugh-Nagumo'
    a:float = 0.
    b:float = 0.
    c:float = 0.08
    R:float = 1.
    D_sys:int = 2
    D_control:int = 0 # For controlled, this should be set to 1 during initialization.

    def f(self, t:float, x:Float[Array, "D_sys"], u=None):
        """
            Fitzhugh Nagumo:

                \dot{x} = 
                \dot{y} = 
    
        Args:
            t (float) : time of evaluation, only used for time-varying systems.
            x (Array) : state at which the ODE is evaluated
            u (Callable): control input interpolation object. call as 'u(t,x)', where t is a float representing the time point.

        Returns
        --------
            \dot{x} (n, D_sys) - Float Array.
        """
        assert len(x) == self.D_sys
        if u is not None:
            u1 = u(t,x)[0]
        else:
            u1 = 0.
        x, y = x[0], x[1]
       
        return jnp.array([x - x**3 / 3 - y +self.R * u1, 
                          self.c * (x + self.a - self.b * y)]) 
    
    def visualize_phaseplane(self, X_range, ax=None, data=None):
        ax = super().visualize_phaseplane(X_range, ax, data)
        ax.set_title(f'{self.name} - phase plane')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        return ax
    

class DuffingOscillator(TwoDimensionalSystem):
    name:str=r'Duffing Oscillator'

    alpha:float = -1.
    beta:float = 2.
    delta:float = .2
    gamma:float = 1.
    omega:float = 5.

    D_sys:int = 2
    D_control:int = 1

    X_range_phaseplane=jnp.array([[-3., 3.],[-2., 2.]])


    def f(self, t:float, x:Float[Array, "D_sys"], u=None):
        """
            Duffing oscillator:

                \dot{x} = y  
                \dot{y} = - αx - βx^3 + δy - γ cos(wt)   
    
        Args:
            t (float) : time of evaluation, only used for time-varying systems.
            x (Array) : state at which the ODE is evaluated
            args (dict) Additional arguments to pass
        
        Returns
        --------
            \dot{x} (n, D_sys) - Float Array.
        """
        assert len(x) == self.D_sys
        if u is not None:            
            u = u.evaluate(t) if isinstance(u, dfx.AbstractGlobalInterpolation) else u
            u = u[0]
        else:
            u = 0.
        
        x, y = x[0], x[1]
        dx = y
        dy = - self.alpha * x - self.beta * x**3 - self.delta * y +  self.gamma*u # self.gamma * jnp.cos(self.omega*t)+
        return jnp.array([dx, dy]) 
    
    def visualize_phaseplane(self, X_range, ax=None, data=None):
        ax = super().visualize_phaseplane(X_range, ax, data)
        ax.set_title(f'{self.name} - phase plane')
        ax.set_xlabel(r'$x$)')
        ax.set_ylabel(r'$y$')
        return ax