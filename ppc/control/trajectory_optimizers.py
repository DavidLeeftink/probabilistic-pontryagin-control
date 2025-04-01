
"""
Copyright (c) 2025 David Leeftink
 
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""
from dataclasses import dataclass
from abc import ABC
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import vmap, grad
import jax.numpy as jnp
import jax.random as jr
import jaxopt
from jaxopt import ScipyMinimize
from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src import projection
from jaxopt._src.tree_util import tree_sub
import scipy
from scipy.optimize import Bounds, NonlinearConstraint, minimize
import diffrax as dfx
import optax
import lineax
import optimistix
import equinox as eqx
from diffrax import AbstractGlobalInterpolation
from equinox import Module
from beartype.typing import Callable, List, Optional, Union
from jaxtyping import Float, Num
from types import SimpleNamespace
from jaxtyping import Array, Float
import colorednoise
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union
import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
from jax.tree_util import register_pytree_node_class
from jaxopt._src import base, implicit_diff as idf
from jaxopt._src import projection#, LbfgsInvHessProductPyTree
from jaxopt._src.tree_util import tree_sub
from jaxopt.projection import projection_non_negative, projection_box
from scipy.optimize import LbfgsInvHessProduct
import numpy as onp
from jaxopt._src.scipy_wrappers import *


@dataclass
class AbstractTrajOptimizer(ABC):
    """
        Adapter design pattern for trajectory optimization methods, supported for L-BFGS, SLSQP and (i)CEM. 
        future methods will include: iLQR, Path Integral controller.        

        Args:
            objective (Callable) Cost function that is to be minimized, integrates dynamics over given timepoints.
                                            input: 
                                                init_params (Array) initial optimization parameters, 
                                                y0 (Array) initial state
                                                control_ts (Array) time points corresponding to the initial parameters.
                                                
                                            output:
                                                and a real-valued cost as output.
    """
    objective:Callable

    def __call__(self, init_params:Array, return_info:bool=False)-> Callable:
        """
        Solve trajectory optimization problem. 

        Args:
            init_params
            ts
            maxiter

        Returns:
            u_t (Callable) / us (Array) TBD
        """
        pass


@dataclass
class Adam(AbstractTrajOptimizer):
    """
        Adam method for first-order optimization (under bounded constraints), using Optax.

        Args:
            learning_rate (float) 
            maxiter (int) maximum number of iterations to execute.
            lb (Array) (M, D_control) used for box projection if given. lower bounds of the constraints, where M is the number of control constraints and D_control is the dimensionality of the control signal.
            ub (Array) (M, D_control) used for box projection if given. upper bounds of the constraints, where M is the number of control constraints and D_control is the dimensionality of the control signal.
            
    """
    step_size:float = 5e-2
    maxiter:int = 5_00
    lb:Array = None
    ub:Array = None

    def __init__(self, 
                 objective:Callable=None, 
                 step_size:float=0.05,
                 maxiter:int=5_00, 
                 lb:Array=None, 
                 ub:Array=None)-> None:
        if lb is not None and ub is not None:
            assert lb.shape == ub.shape, f"Shapes of the lower and upper bounds are not identical. lb shape: {lb.shape} and ub shape: {ub.shape}"
        assert type(maxiter)==int, f"The maximum number of iterations should be an integer, but is instead of type: {type(maxiter)}"
        assert step_size > 0., f"Learning rate should be positive, but is instead: {step_size}"
        super().__init__(objective)
        self.maxiter = maxiter
        self.optim = optax.adam(learning_rate=step_size)
        self.jitted_update = jax.jit(self.optim.update)
        self._compiled_call = None

        if lb is not None and ub is not None:
            self.lb = lb
            self.ub = ub

    def _call_impl(self, init_params: Array, y0:Array, control_ts:Array) -> Callable[..., Any]:
        state = self.optim.init(init_params)
        params = init_params
        init_carry = (params, state)

        # Optimisation step.
        def step(carry, _):
            params, opt_state = carry
            loss_val, loss_gradient = jax.value_and_grad(self.objective)(params, y0, control_ts)
            updates, opt_state = self.jitted_update(loss_gradient, opt_state, params)
            params = optax.apply_updates(params, updates)
            if self.lb is not None and self.ub is not None:
               params = optax.projections.projection_box(params, self.lb, self.ub)

            carry = params, opt_state
            return carry, loss_val

        (params, _), history = jax.lax.scan(step, init_carry, jnp.arange(self.maxiter))
        return (params, history)
    
    def __call__(self, init_params: Array, y0:Array, control_ts:Array, return_info: bool = False, dt0_internal=None, objective:Callable=None) -> Any:
        if self.objective is None:
            if objective is not None:
                self.objective = objective
            else:
                raise AttributeError("Objective function is not given to the optimization class. This can be doen during initialization or in the __call__ (current function)")
            self.objective = objective
            
        if self._compiled_call is None:
            self._compiled_call = jax.jit(self._call_impl)
        
        params, history = self._compiled_call(init_params, y0, control_ts)
        
        return (params, history) if return_info else params
    

@dataclass
class ProjectedGradientBox(AbstractTrajOptimizer):
    """
        Projected gradient method for first-order optimization under (bounded) constraints, using Jaxopt.
        For now, this is fixed for box constraints on the control inputs.
                
        Args:
            step_size (float) a stepsize to use. If <= 0, use backtracking line search.
            maxiter (int) maximum number of iterations to execute.
            lb (Array) (M, D_control) currently not implemented. lower bounds of the constraints, where M is the number of control constraints and D_control is the dimensionality of the control signal.
            ub (Array) (M, D_control) currently not implemented. upper bounds of the constraints, where M is the number of control constraints and D_control is the dimensionality of the control signal.
            
    """
    maxiter:int = 5_00
    step_size:float = None 
    lb:Array = None
    ub:Array = None

    def __init__(self, 
                 objective:Callable=None,
                 lb:Array=None,
                 ub:Array=None,
                 step_size:float=None,
                 maxiter:int=5_00,
                 )-> None:
        
        assert lb.shape == ub.shape, f"Shapes of the lower and upper bounds are not identical. lb shape: {lb.shape} and ub shape: {ub.shape}"
        assert type(maxiter)==int, f"The maximum number of iterations should be an integer, but is instead of type: {type(maxiter)}"
        if step_size is not None:
            assert type(step_size) == float, f"Learning rate (or step size) should be of type float, but is instead: {type(step_size)}"
        super().__init__(objective)
        self.maxiter = maxiter
        if self.objective is not None:
            self.solver = jaxopt.ProjectedGradient(fun=self.objective, 
                                                projection=projection_box, 
                                                maxiter=maxiter, 
                                                stepsize=0. if step_size is None else step_size, 
                                                implicit_diff=False)
        self._compiled_call = None

        if lb is not None and ub is not None:
            self.lb = lb
            self.ub = ub

    def _call_impl(self, init_params: Array, y0:Array, control_ts:Array) -> Callable[..., Any]:
        return self.solver.run(init_params, hyperparams_proj=(self.lb, self.ub), y0=y0, control_ts=control_ts)

    def __call__(self, init_params: Array, y0:Array, control_ts:Array, return_info: bool = False, objective=None, dt0_internal=None) -> Any:
        if self.objective is None:
            if objective is not None:
                self.objective = objective
                self.solver = jaxopt.ProjectedGradient(fun=self.objective, 
                                                projection=projection_box, 
                                                maxiter=self.maxiter, 
                                                stepsize=0. if self.step_size is None else self.step_size, 
                                                implicit_diff=False)
            else:
                raise AttributeError("Objective function is not given to the optimization class. This can be doen during initialization or in the __call__ (current function)")
        if self._compiled_call is None:
            self._compiled_call = jax.jit(self._call_impl)
        res = self._compiled_call(init_params=init_params, y0=y0, control_ts=control_ts)
        return res if return_info else res.params
       

@dataclass
class LBFGSB(AbstractTrajOptimizer):
    """
        Adapter design pattern for using the BFGS Scipy optimizer based on JaxOpt. 
    """
    objective:Callable=None
    solver:jaxopt.ScipyBoundedMinimize = None
    lb:Array = None
    ub:Array = None
    step_size:Float = 0.

    def __init__(self, objective:Callable=None, lb:Array=None, ub:Array=None, maxiter:int=5_00, step_size:float=0.):
        """
            L-BFGS-B method for second-order optimization under bounded constraints, based on JaxOpt's Scipy class.

            Args:
                lb (Array) (M, D_control) lower bounds of the constraints, where M is the number of control constraints and D_control is the dimensionality of the control signal.
                ub (Array) (M, D_control) upper bounds of the constraints, where M is the number of control constraints and D_control is the dimensionality of the control signal.
                maxiter (int) maximum number of iterations to execute.
                callback (Calalble) callback function for the optimizer, that allows for measuring statistics in between optimizer runs.
 
        """
        assert lb.shape == ub.shape, f"Shapes of the lower and upper bounds are not identical. lb shape: {lb.shape} and ub shape: {ub.shape}"
        assert type(maxiter)==int, f"The maixmum number of iterations should be an integer, but is instead of type: {type(maxiter)}"
        super().__init__(objective)
        self._compiled_call = None
        self.maxiter = maxiter
        self.step_size = step_size
        if self.objective is not None:
            self.solver = jaxopt.LBFGSB(fun=self.objective, 
                                    stepsize=step_size,
                                    verbose=1,
                                    maxiter=maxiter
                                    )
        self.lb = lb
        self.ub = ub

    def _call_impl(self, init_params: Array, y0:Array, control_ts:Array) -> Callable[..., Any]:
        res = self.solver.run(init_params, bounds=(self.lb, self.ub), y0=y0, control_ts=control_ts)
        return res #if return_info else res.params

    def __call__(self, init_params: Array, y0:Array, control_ts:Array, return_info:bool=False, objective=None, dt0_internal=None) -> Callable[..., Any]:
        if self.objective is None:
            if objective is not None:
                self.objective = objective
                self.solver = jaxopt.LBFGSB(fun=self.objective, 
                                    stepsize=self.step_size,
                                    verbose=False,
                                    maxiter=self.maxiter,)
            else:
                raise AttributeError("Objective function is not given to the optimization class. This can be doen during initialization or in the __call__ (current function)")
        if self._compiled_call is None:
            self._compiled_call = jax.jit(self._call_impl)
        
        params, history = self._compiled_call(init_params, y0, control_ts)
        # params, history = self._call_impl(init_params, y0, control_ts)
        return (params, history) if return_info else params


@dataclass(eq=False)
class ScipyBoundedMinimizeCopy(ScipyMinimize):
  """`scipy.optimize.minimize` wrapper.

  This wrapper is for minimization subject to box constraints only.

  Attributes:
    fun: a smooth function of the form `fun(x, *args, **kwargs)`.
    method: the `method` argument for `scipy.optimize.minimize`.
    tol: the `tol` argument for `scipy.optimize.minimize`.
    options: the `options` argument for `scipy.optimize.minimize`.
    dtype: if not None, cast all NumPy arrays to this dtype. Note that some
      methods relying on FORTRAN code, such as the `L-BFGS-B` solver for
      `scipy.optimize.minimize`, require casting to float64.
    jit: whether to JIT-compile JAX-based values and grad evals.
    implicit_diff_solve: the linear system solver to use.
    has_aux: whether function `fun` outputs one (False) or more values (True).
      When True it will be assumed by default that `fun(...)[0]` is the
      objective.
  """

  def _fixed_point_fun(self, sol, bounds, args, kwargs):
    step = tree_sub(sol, self._grad_fun(sol, *args, **kwargs))
    return projection.projection_box(step, bounds)

  def optimality_fun(self, sol, bounds, *args, **kwargs):
    """Optimality function mapping compatible with `@custom_root`."""
    fp = self._fixed_point_fun(sol, bounds, args, kwargs)
    return tree_sub(fp, sol)

  def run(self,
          init_params: Any,
          bounds: Optional[Any],
          *args,
          **kwargs) -> base.OptStep:
    """Runs the solver.

    Args:
      init_params: pytree containing the initial parameters.
      bounds: an optional tuple `(lb, ub)` of pytrees with structure identical
        to `init_params`, representing box constraints.
      constraints (Tuple) of (func, lb, ub). For now, these are assumed to be non-linear constraints. 
                    The lb and ub indicate lower and upper values of the constraint evaluation 
      *args: additional positional arguments to be passed to `fun`.
      **kwargs: additional keyword arguments to be passed to `fun`.
    Returns:
      (params, info).
    """
    # Sets up the "JAX-SciPy" bridge.
    pytree_topology = pytree_topology_from_example(init_params)
    onp_to_jnp = make_onp_to_jnp(pytree_topology)

    # wrap the callback so its arguments are of the same kind as fun
    if self.callback is not None:
      def scipy_callback(x_onp: onp.ndarray):
        x_jnp = onp_to_jnp(x_onp)
        return self.callback(x_jnp)
    else:
      scipy_callback = None

    def scipy_fun(x_onp: onp.ndarray) -> Tuple[onp.ndarray, onp.ndarray]:
      x_jnp = onp_to_jnp(x_onp)
      value, grads = self._value_and_grad_fun(x_jnp, *args, **kwargs)
      return onp.asarray(value, self.dtype), jnp_to_onp(grads, self.dtype)

    if bounds is not None:
      bounds = scipy.optimize.Bounds(lb=jnp_to_onp(bounds[0], self.dtype),
                                   ub=jnp_to_onp(bounds[1], self.dtype))
    constraints = [(lambda x: x[0]-x[1], .5, .6),
                   (lambda x: x[-2]*x[-1], 2.5, 2.6),]
    scipy_constraints = [scipy.optimize.NonlinearConstraint(constr_i, lb_i, ub_i) for (constr_i, lb_i, ub_i) in constraints]

    res = scipy.optimize.minimize(scipy_fun, jnp_to_onp(init_params, self.dtype),
                                jac=True,
                                tol=self.tol,
                                bounds=bounds,
                                # constraints=scipy_constraints,
                                method=self.method,
                                callback=scipy_callback,
                                options=self.options)

    params = tree_util.tree_map(jnp.asarray, onp_to_jnp(res.x))

    if hasattr(res, 'hess_inv'):
      if isinstance(res.hess_inv, scipy.optimize.LbfgsInvHessProduct):
        hess_inv = LbfgsInvHessProductPyTree(res.hess_inv.sk,
                                             res.hess_inv.yk)
      elif isinstance(res.hess_inv, onp.ndarray):
        hess_inv = jnp.asarray(res.hess_inv)
    else:
      hess_inv = None

    try:
      num_hess_eval = jnp.asarray(res.nhev, base.NUM_EVAL_DTYPE)
    except AttributeError:
      num_hess_eval = jnp.array(0, base.NUM_EVAL_DTYPE)
    info = ScipyMinimizeInfo(fun_val=jnp.asarray(res.fun),
                             success=res.success,
                             status=res.status,
                             iter_num=res.nit,
                             hess_inv=hess_inv,
                             num_fun_eval=jnp.asarray(res.nfev, base.NUM_EVAL_DTYPE),
                             num_jac_eval=jnp.asarray(res.njev, base.NUM_EVAL_DTYPE),
                             num_hess_eval=num_hess_eval)
    return base.OptStep(params, info)
  
  
@dataclass
class SLSQP(AbstractTrajOptimizer):
    """
        SLSQP Scipy optimizer based on JaxOpt. 
    """
    objective:Callable=None
    solver = None
    lb:Array = None
    ub:Array = None

    def __init__(self, objective:Callable=None, lb:Array=None, ub:Array=None, maxiter:int=5_00, constraints:list=None, callback:Callable=None):
        """
            SLSQP method for second-order optimization under bounded constraints, based on JaxOpt's Scipy class.

            Args:
                lb (Array) (M, D_control) lower bounds of the constraints, where M is the number of control constraints and D_control is the dimensionality of the control signal.
                ub (Array) (M, D_control) upper bounds of the constraints, where M is the number of control constraints and D_control is the dimensionality of the control signal.
                maxiter (int) maximum number of iterations to execute.
                callback (Calalble) callback function for the optimizer, that allows for measuring statistics in between optimizer runs.
 
        """
        assert lb.shape == ub.shape, f"Shapes of the lower and upper bounds are not identical. lb shape: {lb.shape} and ub shape: {ub.shape}"
        assert type(maxiter)==int, f"The maximum number of iterations should be an integer, but is instead of type: {type(maxiter)}"
        super().__init__(objective)
        self.callback = callback
        self.maxiter = maxiter
        if objective is not None:
            self.solver = ScipyBoundedMinimizeCopy(fun=self.objective, maxiter=maxiter, method='SLSQP', callback=callback)
        self.lb = lb
        self.ub = ub

    def __call__(self, init_params: Array, y0:Array, control_ts:Array, return_info:bool=False, objective:Callable=None, dt0_internal=None) -> Callable[..., Any]:
        if self.objective is None:
            if objective is not None:
                self.objective = objective
                self.solver = ScipyBoundedMinimizeCopy(fun=self.objective, maxiter=self.maxiter, method='SLSQP', callback=self.callback)
            else:
                raise AttributeError("Objective function is not given to the optimization class. This can be doen during initialization or in the __call__ (current function)")
        res = self.solver.run(init_params, bounds=(self.lb, self.ub), y0=y0, control_ts=control_ts)
        return res if return_info else res.params
    
    
@dataclass
class CEM(AbstractTrajOptimizer):
    """
        Cross Entropy Method (CEM) for trajectory optimization. Uses a constrained variance based on upper and lower bounds of control input constraints.
        Adopts an MPC adopted version of the CEM method, where a momentum term alpha is applied. Setting alpha to 0. 

        Initialization args:
            objective (Callable)
            pop_size (int) 
            elite_size (int)
            alpha (float) momentum term
            init_var (float) variance of sampling distribution for the initial iteration
            maxiter (int) maximum number of iterations to optimize.
            lb (Array) (M, D_control) lowerbounds of control inputs. Must be the same shape as controls
            ub (Array) (M, D_control) upperbounds of control inputs. Must be the same shape as controls
            eps_accept (float) epsilon term for early stopping. Currently not yet supported.
    
    """
    pop_size:int=100
    elite_size:int=10
    alpha:float = 0.
    init_var:float = 1.
    maxiter:int = 5_00
    objective:Callable=None
    lb:Array = None
    ub:Array = None
    eps_accept:float=1e-4   
        
    def __post_init__(self) -> None:
        self.__check_inputs()
        self._compiled_call = None
        if self.objective is not None:
            self.parallel_objective = vmap(self.objective, in_axes=(0,None,None))

    def _call_impl(self, init_params: Array, y0:Array, control_ts:Array, dt0_internal:float, key:jr.PRNGKey=jr.PRNGKey(42)) -> Callable[..., Any]:
        assert len(init_params.shape)==2, f"init_params is assumed to be of shape (M, D_control) but is instead has dimensions {init_params.shape}"
        maxiter = self.maxiter
        M, D_control = init_params.shape
        params = init_params
        var = self.init_var * jnp.ones((M, D_control))
        keys = jr.split(key, maxiter)
    
        init_carry = (params, var, y0, control_ts, dt0_internal)
        final_carry, history = jax.lax.scan(f=self.jitted_single_iteration, init=init_carry, xs=keys)
        (opt_params, opt_var, _, _, _) = final_carry

        return (opt_params, history)
    
    def single_iteration(self, carry, key:jr.PRNGKey):
        """
            Generates a population by sampling from a Gaussian using 'params' as mean and 'var' as variance, 
                then evaluates the population members and updates the population mean and variance.

            This function is jitted at class initialization.

            Args:
                params (N_control, D_control) Array - control actions, this is the population mean
                var (N_control, D_control) Array - variance of the control actions.
                y0 (D_sys,) Array - initial condition of MPC used in forward simulation
                control_ts (N_control,) - time points corresponding to the control inputs of 'params', helpful during numerical integration with adaptive step sizes.
                key (PRNGKey) - jr random key.

            Returns:
                updated_params (N_control, D_control) Array - updated control actions, this is the population mean
                updated_var (N_control, D_control) Array - updated variance of the control actions.

        """
        params, var, y0, control_ts, dt0_internal = carry
        M,D_control = var.shape
        if self.lb is not None and self.ub is not None:
            lb_distance, ub_distance = params-self.lb, self.ub - params
            constrained_var = jnp.minimum(jnp.minimum((lb_distance / 2)**2, (ub_distance/ 2)**2), var)
        else:
            constrained_var = var
        
        candidates = params + jr.normal(key, (self.pop_size, M, D_control))*jnp.sqrt(constrained_var)
        scores = self.parallel_objective(candidates, y0, control_ts)#, dt0_internal)
        assert len(scores.shape)==1, f"This is supposed to be a one-dimensional array, but is instead of dimensionality {scores.shape}"
        elite_idx = scores.argsort()[:self.elite_size]
        elite_params = candidates[elite_idx]
        elite_mean, elite_var = jnp.mean(elite_params, axis=0), jnp.var(elite_params, axis=0)

        updated_params = self.alpha*params + (1-self.alpha)*elite_mean
        updated_var = self.alpha*constrained_var + (1-self.alpha) * elite_var

        elite_scores = scores[elite_idx]
        stats = (elite_scores.min(), elite_scores.mean(), elite_scores.max())

        carry = (updated_params, updated_var, y0, control_ts, dt0_internal)
        return carry, stats
        
    def __call__(self, init_params: Array, y0:Array, control_ts:Array, dt0_internal:float=0.05, key:jr.PRNGKey=jr.PRNGKey(41), return_info: bool = False, objective:Callable=None) -> Any:
        if self.objective is None:
            if objective is not None:
                self.objective = objective
                self.parallel_objective = vmap(objective, in_axes=(0,None,None))#,None))
            else:
                raise AttributeError("Objective function is not given to the optimization class. This can be done during initialization, or alternatively in the __call__ (the current function)")
        if self._compiled_call is None:
            self.jitted_single_iteration = jax.jit(self.single_iteration)
            self._compiled_call = self._call_impl
        
        params, history = self._compiled_call(init_params, y0, control_ts, dt0_internal, key)
        # params, history = self._call_impl(init_params, y0, control_ts, dt0_internal, key)
        return (params, history) if return_info else params

    
    def __check_inputs(self)-> None:
        assert self.pop_size >= self.elite_size, f"Elite size ({self.elite_size}) is larger than population size ({self.pop_size})"
        assert 0. <= self.alpha and self.alpha <= 1., f"Alpha should be in the range: 0 < alpha < 1, but is instead: {self.alpha}"
        assert type(self.maxiter)==int, f"max iterations should be an integer, but is instead {self.maxiter}"
        assert self.init_var > 0., f"Initial variance of sampling distribution should be larger or equal to 0., but is instead: {self.init_var}"
        if self.lb is not None and self.ub is not None:
            assert self.lb.shape == self.ub.shape, f"Upper bounds and lower bound shapes are not identical."
        assert self.eps_accept >= 0., f"Acceptance criterion epsilon should not be negative, but is instead: {self.eps_accept}"


@dataclass
class iCEM(CEM):
    """
        Improved CEM (iCEM) adapted from Sample-efficient Cross-Entropy Method for Real-time Planning, Pinneri et. al, (2020) (https://proceedings.mlr.press/v155/pinneri21a/pinneri21a.pdf).

        Extends CEM with:
            * colored noise sampling:
                 S(f) = (1 / f)**beta
                    pink noise:   exponent beta = 1
                    brown noise:            exponent beta = 2
            * elite buffer of previous solutions 
            * population size decay
            * executing best evaluation 
            * clipping at the action boundaries

        Args:
            elite_buffer (Array) (S, M, D_control) S elite members of previous solutions that are put in the population every iteration.
            gamma (float) reduction factor of the population size
            beta (float) colored-noise exponent
    """
    beta:float = 1. 
    gamma:float = 0.99
    elite_buffer:Array = None

    def __post_init__(self) -> None:
        self.__check_inputs()
        return super().__post_init__()

    def _call_impl(self, init_params: Array, y0:Array, control_ts:Array, key:jr.PRNGKey=jr.PRNGKey(42), maxiter:int=None) -> Callable[..., Any]:
        assert len(init_params.shape)==2, f"init_params is assumed to be of shape (M, D_control) but is instead has dimensions {init_params.shape}"
        maxiter = self.maxiter if maxiter is None else maxiter
        M, D_control = init_params.shape

        params = init_params
        var = self.init_var * jnp.ones((M, D_control))
        keys = jr.split(key, maxiter)
        
        init_carry = (params, var, y0, control_ts)
        final_carry, history = jax.lax.scan(f=self.jitted_single_iteration, init=init_carry, xs=keys)

        (opt_params, opt_var, _, _) = final_carry
        # best_solution = candidates[scores.argmin()]

        return (opt_params, history)
           
    def single_iteration(self, carry, key:jr.PRNGKey):
        """
            Generates a population by sampling from a Gaussian using 'params' as mean and 'var' as variance, 
                then evaluates the population members and updates the population mean and variance.

            This function is jitted at class initialization.

            Args:
                params (N_control, D_control) Array - control actions, this is the population mean
                var (N_control, D_control) Array - variance of the control actions.
                y0 (D_sys,) Array - initial condition of MPC used in forward simulation
                control_ts (N_control,) - time points corresponding to the control inputs of 'params', helpful during numerical integration with adaptive step sizes.
                key (PRNGKey) - jr random key.

            Returns:
                updated_params (N_control, D_control) Array - updated control actions, this is the population mean
                updated_var (N_control, D_control) Array - updated variance of the control actions.

        """
        params, var, y0, control_ts = carry
        M,D_control = var.shape

        candidates = params + self.get_colored_noise(shape=(self.pop_size, M, D_control), key=key)*jnp.sqrt(var)
        if self.elite_buffer is not None:
            candidates  = jnp.concatenate((candidates, self.elite_buffer), axis=0)
        if self.lb is not None and self.ub is not None:
            candidates = candidates.clip(min=self.lb, max=self.ub)

        scores = self.parallel_objective(candidates, y0, control_ts)
        elite_idx = scores.argsort()[:self.elite_size]
        elite_params = candidates[elite_idx]
        elite_mean, elite_var = jnp.mean(elite_params, axis=0), jnp.var(elite_params, axis=0)
        updated_params = self.alpha*params + (1-self.alpha)*elite_mean
        updated_var = self.alpha*var + (1-self.alpha) * elite_var

        elite_scores = scores[elite_idx]
        stats = (elite_scores.min(), elite_scores.mean(), elite_scores.max())

        carry = (updated_params, updated_var, y0, control_ts)
        return carry, stats
    

    def __call__(self, init_params: Array, y0:Array, control_ts:Array, key:jr.PRNGKey=jr.PRNGKey(42), maxiter:int=None, return_info: bool = False, objective:Callable=None, dt0_internal=None) -> Any:
        if self.objective is None:
            if objective is not None:
                self.objective = objective
                self.parallel_objective = vmap(objective, in_axes=(0,None,None))
            else:
                raise AttributeError("Objective function is not given to the optimization class. This can be doen during initialization or in the __call__ (current function)")
        if self._compiled_call is None:
            self._compiled_call = jax.jit(self._call_impl)
            self.jitted_single_iteration = jax.jit(self.single_iteration)

        
        params, history = self._compiled_call(init_params, y0, control_ts, key, maxiter)
        
        return (params, history) if return_info else params
    
    def get_colored_noise(self, shape:tuple, key:jr.PRNGKey)-> None:
        """
            get colored noise from colorednoise package.

            Based on the algorithm in:
            Timmer, J. and Koenig, M.:
            On generating power law noise.
            Astron. Astrophys. 300, 707-710 (1995)

            Normalised to unit variance

            The power-spectrum of the generated noise is proportional to

            S(f) = (1 / f)**beta
            flicker / pink noise:   exponent beta = 1
            brown noise:            exponent beta = 2

            Furthermore, the autocorrelation decays proportional to lag**-gamma
            with gamma = 1 - beta for 0 < beta < 1.
            There may be finite-size issues for beta close to one.

            Args:
                shape (tuple) (pop_size, num_steps, D_control) shape of the random noise samples
                key (PRNGKey) random key to generate noise with. 
            Return:
                samples (Array) (pop_size, num_steps, D_control) 
        """
        pop_size, num_steps, D_control = shape
        return colorednoise.powerlaw_psd_gaussian(exponent=self.beta, size=(pop_size, D_control, num_steps)).swapaxes(1,2)

    def __check_inputs(self)-> None:
        assert self.beta >= 0., f"beta parameter should be a float greater or equal than 0., but is instead: {self.beta}"
        assert self.gamma >= 0., f"gamma parameter should be a float greater or equal than 0., but is instead: {self.gamma}"
        if self.elite_buffer is not None:
            assert len(self.elite_buffer.shape) == 3, f"Elite buffer should be of dimensionaility: (S, M, D_control) but instead has dimensionality {self.elite_buffer.shape}"
            S, M, D_control = self.elite_buffer.shape
            assert self.lb.shape[0] == M and self.lb.shape[1] == D_control, "Shapes of the elite buffer are not matching the given shapes of the control lower bounds"
            assert self.ub.shape[0] == M and self.ub.shape[1] == D_control, "Shapes of the elite buffer are not matching the given shapes of the control upper bounds" 


@dataclass
class PMPForward(ABC):
    """
        2-point boundary optimization based on the Pontryagin Maximum Principle (PMP).
        Computes a forward solution of the states and a forward integration of the adjoint (co-state).
        Then, Newton's method is used for root-finding of \lambda (0) 
        Designed to be vmap'able, and allow for parallel optimization over dynamics functions.

        Args:
            f (Callable) dynamics function f(x, u, t)
            maxiter (int) maximum number of Newton steps
            D_sys (int) system dimensionality
            D_control (int) control dimensionality
            n_segments (int) number of shooting segments. For single shooting, this is 1.
            state_cost (Callable) state-based cost-funtion L(x,u) 
            termiantion_cost (Callable) termination cost function Phi(x(T))
            solver (dfx.AbstractSolver) Diffrax solver for integrating \dot{x}. Defaults to Dopri5.
            lb (float) (M, D_control) lowerbounds of control inputs. Assumign a convex form of the Hamiltonian for now.
            ub (float) (M, D_control) upperbounds of control inputs. Assuming a convex form of the Hamiltonian for

    """
    f:Callable
    maxiter:int
    D_sys:int
    D_control:int
    n_segments:int=1 # 1 segment = single shooting forward PMP
    state_cost:Callable = None
    termination_cost:Callable = None
    solver:dfx.AbstractSolver = None
    lb:Array = None
    ub:Array = None
   
    def __post_init__(self) -> None:
        self.__check_inputs()
        self._compiled_call = None
        if self.solver is None:
            self.solver = dfx.Dopri5()
        if self.state_cost is None:
            raise NotImplementedError("todo: handle cases with only a termination")
            self.state_cost = lambda xs, us: 0.
        if self.termination_cost is None:
            self.termination_cost = self.state_cost

        # precompute the gradient functions of the Hamiltonian, dH/dx and dH/du .
        self.dH_dx = jax.grad(self.hamiltonian, argnums=1)
        self.dH_du = jax.grad(self.hamiltonian, argnums=2)
        self.termination_grad = jax.grad(self.termination_cost, argnums=0)

    def hamiltonian(self, t:float, x_t:Array, u:Array, lambda_:Array):
        """
            Compute hamiltonian H(t, x, u)
        """
        return self.state_cost(x=x_t, u=u) + jnp.dot(lambda_, self.f(t=t, x=x_t, u=u))
    
    def _call_impl(self, lambda0_init:Array, ts:Array, x0:Array, x_star=None, X_init=None, l_init=None, dt0=None, u_prev:callable=None) -> Callable[..., Any]:
        """
            Solve forward PMP boundary problem. 

            Args:
                lambda0_init (Array) boundary value for lambda(0)
                ts (Array) Time points to evaluate integral on
                x0 (Array) initial system state x(0)
                maxiter (int) maximum iterations to repeat the PMP.

            Return:
                optimal_lambda0 (Array) Optimized control inputs u*
        """   
        termination_grad = jax.grad(self.termination_cost, argnums=0)

        if X_init is None:
            X_init = jnp.linspace(x0, x_star, self.n_segments)[1:] # todo: make this of length n_segments, and then index [1:] to not copy x0 two times.
        if l_init is None:
            l_init = jnp.linspace(lambda0_init, jnp.zeros_like(lambda0_init), self.n_segments)[1:] # todo: make this of length n_segments, and then index [1:] to not copy lambda0 two times.
        S_init = jnp.concatenate([X_init, l_init], axis=1)

        params = (lambda0_init, S_init) 
        subts = jnp.array(jnp.array_split(ts, self.n_segments)) 
        first_vals = jnp.concatenate((subts[1:,0],subts[-1,-1:]), axis=0)[:,None] 
        subts = jnp.concatenate((subts,first_vals), axis=1)
        vmap_solve = vmap(self._solve_state_costate, in_axes=(0,0,None,None))
        u_prev = dfx.LinearInterpolation(ts=ts, ys=jnp.zeros((ts.shape[0], self.D_control))) if u_prev is None else u_prev
        
        def lambda_call(params, args):
            u_prev = args

            ## solve states and costates forward 
            lambda0, S_t = params
            X0 = jnp.concatenate((x0, lambda0))[None,:]
            S = jnp.concatenate((X0, S_t), axis=0) 
            ys = vmap_solve(S, subts, dt0, u_prev) #[n_segment, T, D]

            ## Compute continuity conditions of shooting segments.
            continuity_conditions = jnp.abs(ys[0:-1,-1] - S_t)

            ## Compute λ(T) − ∇E(x(T))
            x_T, lambda_T = ys[-1, -1, :self.D_sys], ys[-1, -1, self.D_sys:]
            u_T = self.u_opt(x_T, lambda_T, u_prev.evaluate(ts[-1]))
            grad_E = lambda_T - termination_grad(x_T, u_T)

            return (grad_E, continuity_conditions)
    
        solver = optimistix.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
        sol = optimistix.least_squares(fn=lambda_call, solver=solver, y0=params, args=u_prev, max_steps=self.maxiter, throw=False)

        return sol.value, (sol.stats, sol.result)

    def u_opt(self, x:Array, lambda_:Array, u_linearization:Array):
        """
            Compute optimal controls u^*(t)_{exp} for a given x(t) and lambda(t).
            This is the controls that minmizes the Hamiltonian H. 

            Assuming convex quadratic form of H = u*u + p*u + q ,
                s.t. u^* = -1/2 @ grad{dH}{du}.

            Args:
                xs (D_sys,) x(t)
                lambdas (D_lambdas) lambda(t)

            Return:
                us (D_control) u^*(t)
        """
        u_placeholder = jnp.zeros((self.D_control,))#u_linearization 
        p = self.dH_du(0., x, u_placeholder, lambda_,) 
        u_opt = -1/2 * jnp.linalg.inv(self.state_cost.R).squeeze() * p # H's unconstrained minimizer 
        u_opt = jnp.minimum(u_opt, self.ub) # constrain to interval [lb, ub]
        u_opt = jnp.maximum(u_opt, self.lb)
        return u_opt

    def f_state_costate(self, t, x, args=None):
        u_prev = args
        u_linearization = u_prev.evaluate(t)

        x_t, lambda_t = x[:self.D_sys], x[self.D_sys:]
        u_t = self.u_opt(x_t, lambda_t, u_linearization)

        # 
        xdot = self.f(t, x_t, u_t) 
        lambda_dot = -self.dH_dx(t, x_t, u_t, lambda_t)

        return jnp.concatenate((xdot, lambda_dot), axis=0)
    
    def _solve_state_costate(self, x0:Array, ts:Array, dt0:float, u_prev:callable)-> Array:
        assert x0.shape[0] == int(2*self.D_sys), f"Initial state x0 is assumed to be the tuple (x0, lambda0), but is instead of dimension {x0.shape}"
        saveat = dfx.SaveAt(ts=ts)
        dt0 = (ts[1]-ts[0]) if dt0 is None else dt0
        stepsize_controller = dfx.ConstantStepSize()#dfx.PIDController(rtol=1e-3, atol=1e-4)#
        return dfx.diffeqsolve(
                dfx.ODETerm(self.f_state_costate),
                solver=self.solver,
                saveat=saveat,
                t0=ts[0],
                t1=ts[-1],
                dt0=dt0,
                stepsize_controller=stepsize_controller,
                y0=x0,
                args=u_prev,
                adjoint=dfx.DirectAdjoint(),
                max_steps=4069*4,
                ).ys
    
    def __call__(self, lambda0_init: Array, ts:Array, x0:Array, x_star:Array=None, X_init=None, l_init=None, dt0=None,u_prev=None,return_info=False) -> Callable[..., Any]:
        if self._compiled_call is None:
            self._compiled_call = jax.jit(self._call_impl)
        sol_value, stats = self._compiled_call(lambda0_init=lambda0_init, ts=ts, x0=x0, x_star=x_star,X_init=X_init, l_init=l_init, dt0=dt0, u_prev=u_prev)
        # sol_value, stats = self._call_impl(lambda0_init=lambda0_init, ts=ts, x0=x0, x_star=x_star, X_init=X_init, l_init=l_init, dt0=dt0, u_prev=u_prev)
        return sol_value, stats

    def __check_inputs(self) -> None:
        assert self.termination_cost is not None or self.state_cost is not None, "Neither state or termination cost function has been given for initialization."


@dataclass
class EnsemblePMPForward(ABC):
    """
        Ensemble Neural ODE variant for 2-point boundary optimization based on the Pontryagin Maximum Principle (PMP).
        Computes a forward solution of the states and a forward integration of the adjoint (co-state).
        Then, Newton's method is used for root-finding of \lambda (0) 
        Designed to be vmap'able, and allow for parallel optimization over dynamics functions.

        Args:
            f (Callable) dynamics function f(x, u, t)
            maxiter (int) maximum number of Newton steps
            D_sys (int) system dimensionality
            D_control (int) control dimensionality
            n_segments (int) number of shooting segments. For single shooting, this is 1.
            state_cost (Callable) state-based cost-funtion L(x,u) 
            termiantion_cost (Callable) termination cost function Phi(x(T))
            solver (dfx.AbstractSolver) Diffrax solver for integrating \dot{x}. Defaults to Dopri5.
            lb (float) (M, D_control) lowerbounds of control inputs. Assumign a convex form of the Hamiltonian for now.
            ub (float) (M, D_control) upperbounds of control inputs. Assuming a convex form of the Hamiltonian for

    """
    f:Callable
    maxiter:int
    D_sys:int
    D_control:int
    ensemble_size:int
    n_segments:int=1 # 1 segment = single shooting forward PMP
    state_cost:Callable = None
    termination_cost:Callable = None
    solver:dfx.AbstractSolver = None
    lb:Array = None
    ub:Array = None
   
    def __post_init__(self) -> None:
        self.__check_inputs()
        self._compiled_call = None
        if self.solver is None:
            self.solver = dfx.Dopri5()
        if self.state_cost is None:
            raise NotImplementedError("Cases with only a termination are not yet supported.")
            self.state_cost = lambda xs, us: 0.
        if self.termination_cost is None:
            self.termination_cost = self.state_cost

        self.termination_grad = jax.grad(self.termination_cost, argnums=0)

    def dH_dx(self, xs:Array, us:Array, lambdas_:Array):
        """
            Gradient of the Hamiltonian w.r.t. state x, for each ensemble member.
            dH(x,u)/dx

            Return: 
                dH_dx (E,1)     
        """
        grad_xdot = self.f.evaluate_ensemble_dH_dx(t=0., xs=xs, us=us, lambdas_=lambdas_)
        grad_cost = vmap(grad(self.state_cost, argnums=0))(xs, us)
        return grad_cost + grad_xdot

    def dH_du(self, xs:Array, us:Array, lambdas_:Array):
        """
            Gradient of the Hamiltonian w.r.t. control input u, for each ensemble member.
            dH(x,u)/du 

            Return: 
                dH_du (E,1)     
        """
        grad_xdot = self.f.evaluate_ensemble_dH_du(t=0., xs=xs, us=us, lambdas_=lambdas_)
        grad_cost = vmap(grad(self.state_cost, argnums=1))(xs, us)
        return grad_xdot+grad_cost
    
    def _call_impl(self, lambda0_init:Array, ts:Array, x0:Array, x_star:Array=None, X_init:Array=None, l_init=None, dt0:float=0.05) -> Callable[..., Any]:
        """
            Computes the forward PMP method for a given ensemble of neural ODEs.

            Args:
                lambda0_init (E,D_sys) initial guesses for lambda0, for each ensemble member.
                ts (T,) time points. Note that these are shared across ensemble members.
                x0 (D_sys) initial condition. Note that this is shared across ensemble members.
                x_star (D_sys) Goal state, used for initializing the shooting segments.
                X_init (E, N_segments-1, D_sys) initial state values for shooting segments. 
                        If left to None, this linearly interpolates between the initial state x0 and the goal state x_star.
                l_init (E,N_segments-1, D_sys) initial clostate values for shooting segments.
                        If left to None, this linearly interpolates between the initial costate lambda0 and zero. 

            Return:
                lambda_sol (Tuple):  lambda(0)   - (E,D_sys)                 the initial values of lambda(0)
                                     S           - (E,N_shooting-1, 2*D_sys) the values of the shooting states and costates
                stats - convergence statistics
        """
        E = self.ensemble_size
        termination_grad = jax.grad(self.termination_cost, argnums=0)
        if x0.ndim == 1:
            tiled_y0s = jnp.tile(x0, (E,1))
        elif x0.ndim == 2:
            tiled_y0s = x0 
            
  
        if X_init is None:
            X_init = jnp.linspace(tiled_y0s, jnp.tile(x_star, (E,1)), self.n_segments).transpose(1,0,2)[:,1:] 
        if l_init is None:
            l_init = jnp.linspace(lambda0_init, jnp.zeros_like(lambda0_init), self.n_segments).transpose(1,0,2)[:,1:] 
      
        S_init = jnp.concatenate([X_init, l_init], axis=-1)

        params = (lambda0_init, S_init)


        ts_dense_segments = jnp.array(jnp.array_split(ts, self.n_segments))  
        first_vals = jnp.concatenate((ts_dense_segments[1:,0],ts_dense_segments[-1,-1:]), axis=0)[:,None] 
        ts_dense_segments = jnp.concatenate((ts_dense_segments,first_vals), axis=1)
        ts_dense_segments_tiled = jnp.tile(ts_dense_segments, (E,1,1))

        vmap_solve_segments = jax.vmap(self._solve_state_costate, in_axes=(1,1,None), out_axes=1) # vmap over segments.

        def lambda_call(params, args):
            lambda0, S_t = params 

            X0 = jnp.concatenate((tiled_y0s, lambda0), axis=-1)[:,None,:]
            S = jnp.concatenate((X0, S_t), axis=1)  

            preds = vmap_solve_segments(S, ts_dense_segments_tiled, dt0) 

            ## Continuity
            continuity_conditions = preds[:,0:-1,-1] - S_t  

            ## Compute λ(T) − ∇E(x(T))
            x_T, lambda_T = preds[:,-1, -1, :self.D_sys], preds[:,-1, -1, self.D_sys:]  
            u_T = self.u_opt(ts[-1], x_T, lambda_T)

            grad_E = jax.vmap(termination_grad)(x_T, u_T) 
            diff = lambda_T - grad_E 

            return (diff, continuity_conditions)

        solver = optimistix.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
        sol = optimistix.least_squares(fn=lambda_call, solver=solver, y0=params, max_steps=self.maxiter, throw=False, tags=lineax.diagonal_tag)

        return sol.value, (sol.stats, sol.result)

    def u_opt(self, t:float, xs:Array, lambdas_:Array):
        """
            Compute optimal controls u^*(t)_{exp} for a given x(t) and lambda(t).
            This is the controls that minmizes the Hamiltonian H. 

            Assuming convex quadratic form of H = u*u + p*u + q ,
                s.t. u^* = -1/2 @ grad{dH}{du}.

            Args:
                xs (E, D_sys,) x(t)
                lambdas (E, D_lambdas) lambda(t)

            Return:
                us (D_control) u^*(t)
        """
        E = self.ensemble_size
        u_placeholders = jnp.zeros((E,self.D_control))
        dH_du = self.dH_du(xs, u_placeholders, lambdas_) # (E,)
        
        # H's unconstrained minimizer
        u_opt = vmap(lambda p:-1/2 * jnp.linalg.inv(self.state_cost.R).squeeze() * p, in_axes=(0,))(dH_du) 
        
        # take mean of controls per ensemble member
        u_mean = u_opt

        # constrained minimizer
        u_mean = vmap(self.project_on_bounds, in_axes=(0,))(u_mean)

        return u_mean
    
    def project_on_bounds(self, u_opt:Array):
        u_opt = jnp.minimum(u_opt, self.ub[0]) # constrain to interval [lb, ub]. not timevarying yet.
        u_opt = jnp.maximum(u_opt, self.lb[0])
        return u_opt

    def f_state_costate(self, t, x, args=None):
        """
            Ensemble State-costate dynamics differential function.

            Arguments:
                t (float) time point of evaluation
                x (Array) (E, D) where D = D_sys + D_control + D_constraints. 
        """
        xs, lambdas_ = x[:,:self.D_sys], x[:,self.D_sys:]
        us = self.u_opt(t, xs, lambdas_)
       
        xdots = self.f.pmp_forward(t=t, xs=xs, us=us)
        lambdas_dot = -self.dH_dx(xs=xs, us=us, lambdas_=lambdas_)
        
        return jnp.concatenate((xdots, lambdas_dot), axis=1)

    def f_state_costate_cost(self, t, x, args=None):
        """
            Ensemble State-costate and cost differential function.

            Arguments:
                t (float) time point of evaluation
                x (Array) (E, D) where D = D_sys + D_control + D_constraints. 
        """
        xs, lambdas_ = x[:,:self.D_sys], x[:,self.D_sys:-1]
        us = self.u_opt(t, xs, lambdas_)
        
        xdots = self.f.pmp_forward(t=t, xs=xs, us=us) 
        lambdas_dot = -self.dH_dx(xs=xs, us=us, lambdas_=lambdas_) 
        cost = vmap(self.state_cost)(xs, us)[:,None]
        
        return jnp.concatenate((xdots, lambdas_dot, cost), axis=1) 

    def _solve_state_costate(self, x0:Array, ts:Array, dt0:float=None)-> Array:
        """
        Solve states x(0) and co-states lambda(0) in batch forward for a single time point sequence.

        Arguments:
            x0 (Array) (E,D) where D = D_sys + D_control + D_constraints
            ts (Array) (E,T). Currently only integrates the first ensemble's time points (assumes all ensembles have the same time points).
            dt0 (float) initial step size during solving.

        Return:
            traj (Array) (E,T,D) where D = D_sys + D_control + D_constraints

        """
        assert x0.shape[-1] >= int(2*self.D_sys), f"Initial state x0 is assumed to be larger or equal then {self.D_sys + self.D_control}, but is instead of dimension {x0.shape}"
        saveat = dfx.SaveAt(ts=ts[0,:])
        dt0 = (ts[0,1]-ts[0,0]) if dt0 is None else dt0
        t0, tf = ts[0,0], ts[0,-1]
        stepsize_controller = dfx.PIDController(atol=1e-3, rtol=1e-4) # dfx.ConstantStepSize() #
        return dfx.diffeqsolve(
                dfx.ODETerm(self.f_state_costate),
                solver=self.solver,
                saveat=saveat,
                t0=t0,
                t1=tf,
                dt0=dt0,
                y0=x0,
                args=None,
                stepsize_controller=stepsize_controller,
                adjoint=dfx.DirectAdjoint(),
                ).ys.transpose(1,0,2)
    
    def _solve_state_costate_cost(self, x0:Array, ts:Array, dt0:float=None)-> Array:
        """
        Solve states x(0),  co-states lambda(0) and cost C in batch forward for a single time point sequence.

        Arguments:
            x0 (Array) (E,D) where D = D_sys + D_control + D_constraints
            ts (Array) (E,T). Currently only integrates the first ensemble's time points (assumes all ensembles have the same time points).
            dt0 (float) initial step size during solving.

        Return:
            traj (Array) (E,T,D) where D = D_sys + D_control + D_constraints

        """
        assert x0.shape[-1] >= int(2*self.D_sys), f"Initial state x0 is assumed to be larger or equal then {self.D_sys + self.D_control}, but is instead of dimension {x0.shape}"
        x0 = jnp.concatenate((x0, jnp.zeros((x0.shape[0],1))),axis=-1)        
        saveat = dfx.SaveAt(ts=ts[0,:])
        dt0 = (ts[0,1]-ts[0,0]) if dt0 is None else dt0
        t0, tf = ts[0,0], ts[0,-1]
        return dfx.diffeqsolve(
                dfx.ODETerm(self.f_state_costate_cost),
                solver=self.solver,
                saveat=saveat,
                t0=t0,
                t1=tf,
                dt0=dt0,
                y0=x0,
                args=None,
                adjoint=dfx.DirectAdjoint(),
                ).ys.transpose(1,0,2)
    
    def __call__(self, lambda0_init: Array, ts:Array, x0:Array, x_star:Array=None, X_init=None, l_init=None, dt0=None,return_info=False) -> Callable[..., Any]:
        if self._compiled_call is None:
            self._compiled_call = jax.jit(self._call_impl)
        sol_value, stats = self._compiled_call(lambda0_init=lambda0_init, ts=ts, x0=x0, x_star=x_star,X_init=X_init, l_init=l_init, dt0=dt0)
        # sol_value, stats = self._call_impl(lambda0_init=lambda0_init, ts=ts, x0=x0, x_star=x_star, X_init=X_init, l_init=l_init, dt0=dt0)
        return sol_value, stats

    def __check_inputs(self) -> None:
        assert self.termination_cost is not None or self.state_cost is not None, "Neither state or termination cost function has been given for initialization."


@dataclass
class EnsemblePMPForwardMeanHamiltonian(ABC):
    """
        Ensemble Neural ODE variant for 2-point boundary optimization based on the Pontryagin Maximum Principle (PMP).
        Computes a forward solution of the states and a forward integration of the adjoint (co-state).
        Then, Newton's method is used for root-finding of \lambda (0) 
        Designed to be vmap'able, and allow for parallel optimization over dynamics functions.

        Args:
            f (Callable) dynamics function f(x, u, t)
            maxiter (int) maximum number of Newton steps
            D_sys (int) system dimensionality
            D_control (int) control dimensionality
            n_segments (int) number of shooting segments. For single shooting, this is 1.
            state_cost (Callable) state-based cost-funtion L(x,u) 
            termiantion_cost (Callable) termination cost function Phi(x(T))
            solver (dfx.AbstractSolver) Diffrax solver for integrating \dot{x}. Defaults to Dopri5.
            lb (float) (M, D_control) lowerbounds of control inputs. Assumign a convex form of the Hamiltonian for now.
            ub (float) (M, D_control) upperbounds of control inputs. Assuming a convex form of the Hamiltonian for

    """
    f:Callable
    maxiter:int
    D_sys:int
    D_control:int
    ensemble_size:int
    n_segments:int=1 # 1 segment = single shooting forward PMP
    state_cost:Callable = None
    termination_cost:Callable = None
    solver:dfx.AbstractSolver = None
    lb:Array = None
    ub:Array = None
    standardize_x:callable=None
    standardize_u:callable = None
    inverse_standardize_x:callable=None
    inverse_standardize_u:callable = None
    sigma_x:float = None
    sigma_u:float = None
   
    def __post_init__(self) -> None:
        self.__check_inputs()
        self._compiled_call = None
        if self.solver is None:
            self.solver = dfx.Dopri5()
        if self.state_cost is None:
            raise NotImplementedError("todo: handle cases with only a termination")
            self.state_cost = lambda xs, us: 0.
        if self.termination_cost is None:
            self.termination_cost = self.state_cost

        self.termination_grad = jax.grad(self.termination_cost, argnums=0)

    def dH_dx(self, xs:Array, us:Array, lambdas_:Array):
        """
            Gradient of the Hamiltonian w.r.t. state x, for each ensemble member.
            dH(x,u)/dx

            Return: 
                dH_dx (E,1)     
        """      
        grad_cost = vmap(grad(self.state_cost, argnums=0))(xs, us) # full scale for the cost.
        grad_xdot = self.f.evaluate_ensemble_dH_dx(t=0., xs=xs, us=us, lambdas_=lambdas_)

        return grad_cost + grad_xdot

    def dH_du(self, xs:Array, us:Array, lambdas_:Array):
        """
            Gradient of the Hamiltonian w.r.t. control input u, for each ensemble member.
            dH(x,u)/du 

            Return: 
                dH_du (E,1)     
        """
        grad_cost = vmap(grad(self.state_cost, argnums=1))(xs, us)      
        grad_udot = self.f.evaluate_ensemble_dH_du(t=0., xs=xs, us=us, lambdas_=lambdas_)
    
        return grad_udot+grad_cost
    
    def _call_impl(self, lambda0_init:Array, ts:Array, x0:Array, x_star:Array=None, X_init:Array=None, l_init=None, dt0:float=0.05) -> Callable[..., Any]:
        """
            Computes the forward PMP method for a given ensemble of neural ODEs.

            Args:
                lambda0_init (E,D_sys) initial guesses for lambda0, for each ensemble member.
                ts (T,) time points. Note that these are shared across ensemble members.
                x0 (D_sys) initial condition. Note that this is shared across ensemble members.
                x_star (D_sys) Goal state, used for initializing the shooting segments.
                X_init (E, N_segments-1, D_sys) initial state values for shooting segments. 
                        If left to None, this linearly interpolates between the initial state x0 and the goal state x_star.
                l_init (E,N_segments-1, D_sys) initial clostate values for shooting segments.
                        If left to None, this linearly interpolates between the initial costate lambda0 and zero. 

            Return:
                lambda_sol (Tuple):  lambda(0)   - (E,D_sys)                 the initial values of lambda(0)
                                     S           - (E,N_shooting-1, 2*D_sys) the values of the shooting states and costates
                stats - convergence statistics
        """
        E = self.ensemble_size
        termination_grad = jax.grad(self.termination_cost, argnums=0)
        if x0.ndim == 1:
            tiled_y0s = jnp.tile(x0, (E,1))
        elif x0.ndim == 2:
            tiled_y0s = x0 
            
        if X_init is None:
            X_init = jnp.linspace(tiled_y0s, jnp.tile(x_star, (E,1)), self.n_segments).transpose(1,0,2)[:,1:] 
        if l_init is None:
            l_init = jnp.linspace(lambda0_init, jnp.zeros_like(lambda0_init), self.n_segments).transpose(1,0,2)[:,1:] 
      
        S_init = jnp.concatenate([X_init, l_init], axis=-1)

        params = (lambda0_init, S_init)


        ts_dense_segments = jnp.array(jnp.array_split(ts, self.n_segments)) 
        first_vals = jnp.concatenate((ts_dense_segments[1:,0],ts_dense_segments[-1,-1:]), axis=0)[:,None] 
        ts_dense_segments = jnp.concatenate((ts_dense_segments,first_vals), axis=1) 
        ts_dense_segments_tiled = jnp.tile(ts_dense_segments, (E,1,1))

        vmap_solve_segments = jax.vmap(self._solve_state_costate, in_axes=(1,1,None), out_axes=1) # vmap over segments.

        def lambda_call(params, args):
            lambda0, S_t = params 

            X0 = jnp.concatenate((tiled_y0s, lambda0), axis=-1)[:,None,:] 
            S = jnp.concatenate((X0, S_t), axis=1)  

            preds = vmap_solve_segments(S, ts_dense_segments_tiled, dt0) 

            ## Continuity
            continuity_conditions = preds[:,0:-1,-1] - S_t  

            ## Compute λ(T) − ∇E(x(T))
            x_T, lambda_T = preds[:,-1, -1, :self.D_sys], preds[:,-1, -1, self.D_sys:] 
            u_T = self.u_opt(ts[-1], x_T, lambda_T)

            grad_E = jax.vmap(termination_grad)(x_T, u_T) 
            diff = lambda_T - grad_E 

            return (diff, continuity_conditions)

        solver = optimistix.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
        sol = optimistix.least_squares(fn=lambda_call, solver=solver, y0=params, max_steps=self.maxiter, throw=False)

        return sol.value, (sol.stats, sol.result)

    def u_opt(self, t:float, xs:Array, lambdas_:Array):
        """
            Compute optimal controls u^*(t)_{exp} for a given x(t) and lambda(t).
            This is the controls that minmizes the Hamiltonian H. 

            Assuming convex quadratic form of H = u*u + p*u + q ,
                s.t. u^* = -1/2 @ grad{dH}{du}.

            Args:
                xs (E, D_sys,) x(t)
                lambdas (E, D_lambdas) lambda(t)

            Return:
                us (D_control) u^*(t)
        """
        E = self.ensemble_size
        u_placeholders = jnp.zeros((E,self.D_control))

        dH_du = self.dH_du(xs, u_placeholders, lambdas_) # (E,)
        
        # H's unconstrained minimizer
        u_opt = vmap(lambda p:-1/2 * jnp.linalg.inv(self.state_cost.R).squeeze() * p, in_axes=(0,))(dH_du) 
        
        # take mean of controls per ensemble member
        u_mean = u_opt*0. + u_opt.mean(axis=0)   

        # constrained minimizer
        u_mean = vmap(self.project_on_bounds, in_axes=(0,))(u_mean)

        return u_mean
    
    def project_on_bounds(self, u_opt:Array):
        u_opt = jnp.minimum(u_opt, self.ub[0]) # constrain to interval [lb, ub]. not timevarying yet.
        u_opt = jnp.maximum(u_opt, self.lb[0])
        return u_opt

    def f_state_costate(self, t, x, args=None):
        """
            Ensemble State-costate dynamics differential function.

            Arguments:
                t (float) time point of evaluation
                x (Array) (E, D) where D = D_sys + D_control + D_constraints. 
        """

        xs, lambdas_ = x[:,:self.D_sys], x[:,self.D_sys:]
  
        us = self.u_opt(t, xs, lambdas_)
        lambdas_dot = -self.dH_dx(xs=xs, us=us, lambdas_=lambdas_)

        ## scale down while evaluating f() and scale back up.
        xdots = self.f.pmp_forward(t=t, xs=xs, us=us) 
        
        return jnp.concatenate((xdots, lambdas_dot), axis=1) 

    def _solve_state_costate(self, x0:Array, ts:Array, dt0:float=None)-> Array:
        """
        Solve states x(0) and co-states lambda(0) in batch forward for a single time point sequence.

        Arguments:
            x0 (Array) (E,D) where D = D_sys + D_control + D_constraints
            ts (Array) (E,T). Currently only integrates the first ensemble's time points (assumes all ensembles have the same time points).
            dt0 (float) initial step size during solving.

        Return:
            traj (Array) (E,T,D) where D = D_sys + D_control + D_constraints

        """
        assert x0.shape[-1] >= int(2*self.D_sys), f"Initial state x0 is assumed to be larger or equal then {self.D_sys + self.D_control}, but is instead of dimension {x0.shape}"
        saveat = dfx.SaveAt(ts=ts[0,:])
        dt0 = (ts[0,1]-ts[0,0]) if dt0 is None else dt0
        t0, tf = ts[0,0], ts[0,-1]
        stepsize_controller = dfx.PIDController(atol=1e-3, rtol=1e-4) #dfx.ConstantStepSize()
        return dfx.diffeqsolve(
                dfx.ODETerm(self.f_state_costate),
                solver=self.solver,
                saveat=saveat,
                t0=t0,
                t1=tf,
                dt0=dt0,
                y0=x0,
                args=None,
                stepsize_controller=stepsize_controller,
                adjoint=dfx.DirectAdjoint(),
                ).ys.transpose(1,0,2)

    def _solve_state_costate_cost(self, x0:Array, ts:Array, dt0:float=None)-> Array:
        """
        Solve states x(0),  co-states lambda(0) and cost C in batch forward for a single time point sequence.

        Arguments:
            x0 (Array) (E,D) where D = D_sys + D_control + D_constraints
            ts (Array) (E,T). Currently only integrates the first ensemble's time points (assumes all ensembles have the same time points).
            dt0 (float) initial step size during solving.

        Return:
            traj (Array) (E,T,D) where D = D_sys + D_control + D_constraints

        """
        assert x0.shape[-1] >= int(2*self.D_sys), f"Initial state x0 is assumed to be larger or equal then {self.D_sys + self.D_control}, but is instead of dimension {x0.shape}"
        x0 = jnp.concatenate((x0, jnp.zeros((x0.shape[0],1))),axis=-1)        
        saveat = dfx.SaveAt(ts=ts[0,:])
        dt0 = (ts[0,1]-ts[0,0]) if dt0 is None else dt0
        t0, tf = ts[0,0], ts[0,-1]
        return dfx.diffeqsolve(
                dfx.ODETerm(self.f_state_costate_cost),
                solver=self.solver,
                saveat=saveat,
                t0=t0,
                t1=tf,
                dt0=dt0,
                y0=x0,
                args=None,
                adjoint=dfx.DirectAdjoint(),
                ).ys.transpose(1,0,2)
    
    def __call__(self, lambda0_init: Array, ts:Array, x0:Array, x_star:Array=None, X_init=None, l_init=None, dt0=None,return_info=False) -> Callable[..., Any]:
        if self._compiled_call is None:
            self._compiled_call = jax.jit(self._call_impl)
        sol_value, stats = self._compiled_call(lambda0_init=lambda0_init, ts=ts, x0=x0, x_star=x_star,X_init=X_init, l_init=l_init, dt0=dt0)
        # sol_value, stats = self._call_impl(lambda0_init=lambda0_init, ts=ts, x0=x0, x_star=x_star, X_init=X_init, l_init=l_init, dt0=dt0)
        return sol_value, stats

    def __check_inputs(self) -> None:
        assert self.termination_cost is not None or self.state_cost is not None, "Neither state or termination cost function has been given for initialization."