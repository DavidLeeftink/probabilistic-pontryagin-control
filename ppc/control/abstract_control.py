"""
Copyright (c) 2025 David Leeftink
 
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""
from abc import ABC
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
import diffrax as dfx
from diffrax import AbstractGlobalInterpolation
from beartype.typing import Callable, List, Optional, Union
from jaxtyping import Float, Num
from types import SimpleNamespace
from jaxtyping import Array, Float


@dataclass
class AbstractController(ABC):
    """
    Interface for controllers, such as model predictive control (MPC), iLQR, or deep RL policies.

    To be applicable to discrete-time and continuous-time setting, the controllers return 
    an interpolated control function u(t). For discrete-time controllers, this will be a 
    zero-order hold (ZOH) interpolation scheme.
    """
    D_control:int
    control_interpolator:str='linear'

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x:Array, t:float):
        """
        For a given (distribution over) state x, compute the control u. 

        Args:
            x (Array) (N,) 
            t (float) timepoint of evaluation

        Returns:
            u_t (GlobalInterpolation) control function u(t), which can be called via 'u_t.evaluate(t)' .
        """
        raise NotImplementedError("Abstract method should not be called")

    def get_interpolator(self, us:Array, ts:Array):
        """
        Apply interpolation to a set of control signals at fixed timepoins.
        Assumes single trajectory.

        Args:
            us (Array) control inputs at given timepoints
            ts (Array) time points. 
        
        Return: 
            u_t (dfx.Interpolation class) interpolator object that gives u(t) by calling u_t.evaluate(t)
        """
        if us is None:
            return SimpleNamespace(evaluate=lambda t: jnp.zeros((self.D_control,)))
        else:
            assert ts.shape[0]==us.shape[0], "Number of control sequences does not match the number of time point sequences"
            assert len(us.shape) == 2, "Shape of controls should be: (number of timepoints, number of control dimensions)"

            if self.control_interpolator == 'linear':
                return dfx.LinearInterpolation(ts=ts, ys=us)
            else:
                raise NotImplementedError("Only linear interpolation is added for now.")
