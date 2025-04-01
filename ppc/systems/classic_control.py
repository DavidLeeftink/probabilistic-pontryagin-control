"""
Copyright (c) 2025 David Leeftink
 
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""
import jax.numpy as jnp
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Num,
)
import diffrax as dfx
from jaxtyping import (
    Array, 
    Float,
)   
from ppc.systems.abstract import *


class LTI(MultiDimensionalSystem):
    name:str= 'LTI'
    A:Array = None
    D_control:int = 0
   

    def __post_init__(self)->None:
        assert self.A is not None, f"No system matrix A given. "
        assert self.A.shape[0] == self.A.shape[1], f"System matrix A should be a square matrix, but instead has shape: {self.A.shape}"
        self.D_sys = self.A.shape[0]

    def f(self, t:float, x:Float[Array, "D_sys"], args=None):
        """
            Linear Time Invariant (LTI) system

                \dot{x} = Ax 
    
        Args:
            t (float) : time of evaluation, only used for time-varying systems.
            x (Array) : state at which the ODE is evaluated
            args (dict) Additional arguments to pass
        
        Returns
        --------
            \dot{x} (n, D_sys) - Float Array.
        """
        assert len(x) == self.D_sys
        u_t = args
        return self.A@x
    
    def visualize_phaseplane(self, X_range, ax=None, data=None):
        ax = super().visualize_phaseplane(X_range, ax, data)
        ax.set_title(f'{self.name} - phase plane')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        return ax
    
class LTIControlled(LTI):
    name:str= 'LTI with control'
    B:Array = None
   

    def __post_init__(self)->None:
        super().__post_init__()
        if self.B is None:
            self.D_control = 0
            self.B = jnp.zeros((self.A.shape[0]))
        else:
            self.D_control = self.B.shape[0]        

    def f(self, t:float, x:Float[Array, "D_sys"], args):
        """
            Linear Time Invariant (LTI) system

                \dot{x} = Ax + Bu
    
        Args:
            t (float) : time of evaluation, only used for time-varying systems.
            x (Array) : state at which the ODE is evaluated
            args (dict) Additional arguments to pass
        
        Returns
        --------
            \dot{x} (n, D_sys) - Float Array.
        """
        assert len(x) == self.D_sys
        u_t = args
        return self.A@x + self.B@u_t.evaluate(t)
    
    def visualize_phaseplane(self, X_range, ax=None, data=None):
        ax = super().visualize_phaseplane(X_range, ax, data)
        ax.set_title(f'{self.name} - phase plane')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        return ax
    

class InvertedPendulum(TwoDimensionalSystem):
    name:str='Inverted Pendulum'
    B:Array = None
    g:float = 9.81
    l:float = 1.0
    m:float = 1.
    max_speed = 8.
    X_range_phaseplane = jnp.array([[-4., 4.],[-6., 6.]])

    def f(self, t:float, x:Float[Array, "D_sys"], u=None):
        """
        Pendulum swing-up.
    
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

        g, m, l = self.g, self.m, self.l        
        
        xdot = y
        ydot = -(g/l)*jnp.sin(x) + u
        ydot = jnp.clip(ydot, -self.max_speed, self.max_speed)
        return jnp.array([xdot, ydot])


class InvertedPendulumV2(ThreeDimensionalSystem):
    name:str='Inverted Pendulum'
    B:Array = None
    g:float = 9.81
    b:float = 1.
    l:float = 1.0
    m:float = 1.
    max_speed = 8.
    X_range_phaseplane = jnp.array([[-4., 4.],[-6., 6.]])

    def f(self, t:float, x:Float[Array, "D_sys"], u=None):
        """
        Pendulum swing-up.
    
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

        ctheta, stheta, thetadot = x[0], x[1], x[2]

        # Derivatives
        dc_dt = -stheta * thetadot
        ds_dt = ctheta * thetadot
        
        b = self.b 
        theta_ddot = -(self.g / self.l) * stheta - b * thetadot + u / (self.m * self.l**2)
        
        thetadot = jnp.clip(thetadot, -self.max_speed, self.max_speed)
        
        return jnp.array([dc_dt, ds_dt, theta_ddot])

    def visualize_phaseplane(self, X_range, ax=None, data=None):
        return ax
        
def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    

class Cartpole(MultiDimensionalSystem):
    name:str='Cartpole'
    g:float = 9.81
    l:float = 1. # pendulum length
    m:float = .1 # pendulum mass 
    M:float = 1. # cart mass
    delta:float = 1. # cart damping
    max_speed = 8.

    X_range_phaseplane = jnp.array([[-4., 4.],[-6., 6.], [-4., 4.],[-6., 6.]])
   

    def f(self, t:float, x:Float[Array, "D_sys"], u=None):
        """

    
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
        
        x, theta, dx, dtheta = x

        # Eq. 6.1
        ddx = ((self.l * self.m * jnp.sin(theta) * dtheta ** 2 + u + self.m * self.g * jnp.cos(theta) * jnp.sin(theta))
            / (self.M + self.m * (1 - jnp.cos(theta) ** 2)))
        ddx = jnp.squeeze(ddx)

        # Eq. 6.2
        ddtheta = - ((self.l * self.m * jnp.cos(theta) * dtheta ** 2 + 5*u * jnp.cos(theta)
                    + (self.M + self.m) * self.g * jnp.sin(theta))
                    / (self.l * self.M + self.l * self.m * (1 - jnp.cos(theta) ** 2)))
        ddtheta = jnp.squeeze(ddtheta)
        return jnp.array([dx, dtheta, ddx, ddtheta])
    
    def visualize_phaseplane(self, X_range, ax=None, data=None):
        pass


class Cartpolev2(MultiDimensionalSystem):
    name:str='Cartpole'
    g:float = 9.81
    l:float = 1. # pendulum length
    m:float = .1 # pendulum mass 
    M:float = 1. # cart mass
    delta:float = 1. # cart damping
    max_speed = 8.

    X_range_phaseplane = jnp.array([[-4., 4.],[-6., 6.]])
   

    def f(self, t:float, x:Float[Array, "D_sys"], u=None):
        """

    
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
        
        x, ctheta, stheta, dx, dtheta = x

        theta = jnp.arctan2(stheta, ctheta)
        C = ctheta**2 + stheta**2
    
        ctheta, stheta = ctheta / jnp.sqrt(C), stheta / jnp.sqrt(C)
        # state_4d = jnp.array([x, theta, xdot, thetadot])

        # Eq. 6.1
        ddx = ((self.l * self.m * jnp.sin(theta) * dtheta ** 2 + u + self.m * self.g * jnp.cos(theta) * jnp.sin(theta))
            / (self.M + self.m * (1 - jnp.cos(theta) ** 2)))
        ddx = jnp.squeeze(ddx)

        # Eq. 6.2
        ddtheta = - ((self.l * self.m * jnp.cos(theta) * dtheta ** 2 + u * jnp.cos(theta)
                    + (self.M + self.m) * self.g * jnp.sin(theta))
                    / (self.l * self.M + self.l * self.m * (1 - jnp.cos(theta) ** 2)))
        ddtheta = jnp.squeeze(ddtheta)

        dc_theta = -stheta * dtheta/C  # Correct: d/dt cos(theta) = -sin(theta) * thetadot
        ds_theta = ctheta * dtheta/C

        return jnp.array([dx, dc_theta, ds_theta, ddx, ddtheta])
    
    def visualize_phaseplane(self, X_range, ax=None, data=None):
        pass