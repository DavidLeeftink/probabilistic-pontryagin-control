"""
Copyright (c) 2025 David Leeftink
 
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""
from abc import ABC
from dataclasses import dataclass
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
import diffrax as dfx
from diffrax import AbstractGlobalInterpolation
from equinox import Module
import equinox as eqx
from beartype.typing import Callable, List, Optional, Union
from jaxtyping import Float, Num
from types import SimpleNamespace
from jaxtyping import Array, Float
from .abstract_control import AbstractController
from .trajectory_optimizers import AbstractTrajOptimizer
from ppc.systems.abstract import AbstractSystem

@dataclass 
class QuadraticCost(ABC):
    """
        Quadratic cost function L(x,u) = x^T Q x + u^T R u, for a single state. 
        Can be integrated as part of the ODE to obtain the trajectory cost.

        Parameters: 
            Q (Array) State cost matrix of shape (D_sys,D_sys). Typically identity matrix with scaled diagonal. Should be PSD.
            R (Array) Control cost matrix of shape (D_control, D_control). Should be PSD.
    """
    Q:Array = None
    R:Array = None
    x_star:Array = None
    u_star:Array = None
    transform:callable=None

    def __call__(self, x:Array, u:Array):
        if self.transform is not None:
            x = self.transform(x)
        x = x-self.x_star
        u = u-self.u_star
        return x.T@self.Q@x + u.T@self.R@u

    def __post_init__(self):
        if self.x_star is None:
            self.x_star = jnp.zeros((self.Q.shape[0]),)
        if self.u_star is None:
            self.u_star = jnp.zeros((self.R.shape[0]),)

        self._check_shapes()
        # self._check_PSD()

    def _check_shapes(self):
        assert self.Q.shape[0] == self.x_star.shape[0], f"Q and x* do not have the same dimensionality: Q shape: {self.Q.shape[0]} and x* shape: {self.x_star.shape[0]}"
        assert self.R.shape[0] == self.u_star.shape[0], f"R and u* do not have the same dimensionality: R shape: {self.R.shape[0]} and u* shape: {self.u_star.shape[0]}"

    # def _check_PSD(self):
    #     assert jnp.all(jnp.linalg.eigh(self.Q)>=0.), "Q is not positive-semidefinite matrix. "
    #     assert jnp.all(jnp.linalg.eigh(self.R)>=0.), "R is not positive-semidefinite matrix. " 