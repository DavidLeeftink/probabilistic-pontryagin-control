"""
Copyright (c) 2025 David Leeftink
 
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn
import diffrax as dfx
from jaxtyping import Array
from beartype.typing import Callable
from dataclasses import dataclass

class NeuralVectorField(eqx.Module):
    """
    Vectorfield modeled by a neural network.
    Based on the diffrax examples using the Equinox library.

    Parameters:
        layers (tuple) list of weights and biases. e.g.: (layer1size, layer2size, ..., layerN_size)
        activation (Callable) Non-linear activation function. Uses the same activation for all layers for now.
        final_activation (Callable) Non-linear activation function for final layer. If none, then no transformation is applied. 
        D_sys (int) vectorfield dimensionality
        D_control (int) dimensionality of control input
        _name (str)
    """
    layers:tuple[eqx.nn.Linear, ...]
    activation: Callable
    final_activation:Callable
    D_sys:int
    observation_model:callable= None
    inverse_observation_model:callable= None

    D_control:int=0
    _name:str='Neural ODE'

    def __init__(self, 
                 layer_sizes:tuple, 
                 D_sys:int,
                 D_control:int=0,
                 activation:callable=jnn.elu, 
                 final_activation:callable=lambda x:x, 
                 observation_model:callable = None,
                 inverse_observation_model:callable = None,
                 *, 
                 key:jr.PRNGKey,
                 ):
        keys = jr.split(key, len(layer_sizes))
        self.layers = [eqx.nn.Linear(in_features=layer_sizes[i],
                                     out_features=layer_sizes[i+1],
                                     use_bias=True,
                                     key=keys[i]) 
                        for i in range(len(layer_sizes)-1)]
        self.D_sys = D_sys
        self.D_control = D_control
        self.activation = activation
        self.final_activation = final_activation
        self.observation_model = observation_model
        self.inverse_observation_model = inverse_observation_model

        
    def __call__(self, t, x, u=None):
        """
        Forward function of f(x,u)

        Args:
            t (float) time point of evaluation
            x (Array) input
            args () additional arguments of ODE, e.g. external force.
        
        Return:
            y (Array) output
        """
        if u is not None:
            if isinstance(u, dfx.AbstractGlobalInterpolation):
                input = jnp.concatenate((x,u.evaluate(t)))
            else:
                input = jnp.concatenate((x,u))
        else:
            input = x

        if self.observation_model is not None:
            input = self.observation_model(input)
            
        x = self.layers[0](input)
        x = self.activation(x)

        for layer in self.layers[1:-1]:
            x = layer(x)
            x = self.activation(x)

        x = self.layers[-1](x)
        x = self.final_activation(x)

        if self.inverse_observation_model is not None:
            x = self.inverse_observation_model(x)
            
        return x
    

class EnsembleNeuralVectorField(eqx.Module):
    """
    Ensemble of vectorfields modeled by neural networks.
    Based on the diffrax examples using the Equinox library.

    Parameters:
        layers (tuple) list of weights and biases. e.g.: (layer1size, layer2size, ..., layerN_size)
        activation (Callable) Non-linear activation function. Uses the same activation for all layers for now.
        final_activation (Callable) Non-linear activation function for final layer. If none, then no transformation is applied. 
        D_sys (int) vectorfield dimensionality
        D_control (int) dimensionality of control input
        _name (str)
    """
    ensemble_size:int
    layer_sizes:tuple
    activation: Callable
    final_activation:Callable
    D_sys:int
    D_control:int=0
    observation_model:callable= None
    inverse_observation_model:callable= None
    _name:str='Ensemble neural ODE'
    model:eqx.Module = None 

    def __init__(self, 
                 ensemble_size:int,
                 layer_sizes:tuple, 
                 D_sys:int,
                 D_control:int=0,
                 activation:callable=jnn.elu, 
                 final_activation:callable=lambda x:x, 
                 observation_model:callable = None,
                 inverse_observation_model:callable = None,
                 *, 
                 key:jr.PRNGKey,
                ):
        keys = jr.split(key, ensemble_size)
        self.ensemble_size = ensemble_size
        self.layer_sizes = layer_sizes
        self.D_sys = D_sys
        self.D_control = D_control
        self.activation = activation
        self.final_activation = final_activation
        self.observation_model = observation_model
        self.inverse_observation_model = inverse_observation_model
        self.model = self.make_ensemble(keys)

    def __call__(self, t:float, xs:Array, args):
        """
        Forward function of f(x,u) for each ensemble member.

        Args:
            t (E,) floats - time point of evaluation
            x (E, D_sys)  Array  input state
            args (ts,us) additional arguments of ODE.
                where ts (E,T)           all timepoints of integration required for interpolation
                      us (E,T,D_control) all control inputs of integration required for interpolation

            where E is the ensemble_size.
        
        Return:
            y (Array) predicted next state
        """
        ts, us = args
        return self.evaluate_ensemble(self.model, t, xs, ts, us)
    
    def pmp_forward(self, t:float, xs:Array, us:Array):
        """
        Forward function of f(x,u) for each ensemble member.

        Args:
            t (E,) floats - time point of evaluation
            x (E, D_sys)  Array  input state
            us (E, D_control)  Control inputs.
                
            where E is the ensemble_size.
        
        Return:
            y (Array) predicted next state
        """
        return self.evaluate_ensemble_pmp(self.model, t, xs, us)
    
    def forward_shared_us(self, t:float, xs:Array, args):
        """
        Forward function of f(x,u) for each ensemble member.

        Args:
            t (E,) floats - time point of evaluation
            x (E, D_sys)  Array  input state
            args additional arguments of ODE.
                where u (dfx.AbstractGlobalInterpolation)
            where E is the ensemble_size.
        
        Return:
            y (Array) predicted next state
        """
        u = args
        return self.evaluate_ensemble_pmp_shared_u(self.model, t, xs, u)

    @eqx.filter_vmap(in_axes=(None, eqx.if_array(0), None, 0, None))
    def evaluate_ensemble_pmp_shared_u(self, model, t, x, u): 
        """ 
            Evaluate each member of the ensemble on different data for the PMP model. 
            Model, t, x, and u are all vmapped over the ensemble size dimension.
        """
        return model(t=t, x=x, u=u)
    
    def backward_shared_us(self, t:float, lambdas_:Array, args):
        """
        Compute the costate equation 
            \dot{lambda} = \nabla_x H(x,lambda, u), for each ensemble member.


        Args:
            t (E,) floats - time point of evaluation
            lambdas (E, D_sys)  Array  input co-states
            args additional arguments of ODE.
                where u (dfx.AbstractGlobalInterpolation)
            where E is the ensemble_size.
        
        Return:
            y (Array) predicted next state
        """
        ts, xs, u = args
        return self.evaluate_ensemble_backward_shared_u(self.model, t, ts, lambdas_, xs, u)

    @eqx.filter_vmap(in_axes=(None, eqx.if_array(0), None, None, 0, 0, None))
    def evaluate_ensemble_backward_shared_u(self, model, t, ts, lambda_, xs, u): 
        """ 
            Evaluate each member of the ensemble on different data for the PMP model. 
            Model, t, x, and u are all vmapped over the ensemble size dimension.
        """
        coeffs = dfx.backward_hermite_coefficients(ts=ts, ys=xs)
        x_t = dfx.CubicInterpolation(ts=ts, coeffs=coeffs)
        # x_t = dfx.LinearInterpolation(ts=ts, ys=xs)
        x = x_t.evaluate(t)
        return -jax.grad(lambda x_: jnp.dot(lambda_, model(t=t, x=x_, u=u)))(x)
    
    @eqx.filter_vmap(in_axes=(None, eqx.if_array(0), None, 0, 0, 0))
    def evaluate_ensemble(self, model, t, x, ts, us): 
        """ 
            Evaluate each member of the ensemble on different data. 
            Model, t, x, and u are all vmapped over the ensemble size dimension.
        """
        u_t = dfx.LinearInterpolation(ts, us)
        u = u_t.evaluate(t)
        return model(t=t, x=x, u=u)
    
    @eqx.filter_vmap(in_axes=(None, eqx.if_array(0), None, 0, 0))
    def evaluate_ensemble_pmp(self, model, t, x, u): 
        """ 
            Evaluate each member of the ensemble on different data for the PMP model. 
            Model, t, x, and u are all vmapped over the ensemble size dimension.
        """
        return model(t=t, x=x, u=u)
    
    def evaluate_ensemble_dH_dx(self, t:float, xs:Array, us:Array, lambdas_:Array):
        """
        Forward function of f(x,u) for each ensemble member.

        Args:
            t (E,) floats - time point of evaluation
            x (E, D_sys)  Array  input state
            us (E, D_control)  Control inputs.
                
            where E is the ensemble_size.
        
        Return:
            y (Array) predicted next state
        """
        return self.evaluate_dH_dx(self.model, t, xs, us, lambdas_)
    
    @eqx.filter_vmap(in_axes=(None, eqx.if_array(0), None, 0, 0, 0))
    def evaluate_dH_dx(self, model, t, x, u, lambda_): 
        """ 
            Evaluate each member of the ensemble on different data for the PMP model. 
            Model, t, x, and u are all vmapped over the ensemble size dimension.
        """
        return jax.grad(lambda x_: jnp.dot(lambda_, model(t=t, x=x_, u=u)))(x)
    
    @eqx.filter_vmap(in_axes=(None, eqx.if_array(0), None, 0, None, 0))
    def evaluate_dH_dx_shared_u(self, model, t, x, u, lambda_): 
        """ 
            Evaluate each member of the ensemble on different data for the PMP model. 
            Model, t, x, and u are all vmapped over the ensemble size dimension.
        """
        return jax.grad(lambda x_: jnp.dot(lambda_, model(t=t, x=x_, u=u)))(x)
    
    def evaluate_ensemble_dH_du(self, t:float, xs:Array, us:Array, lambdas_:Array):
        """
        Forward function of f(x,u) for each ensemble member.

        Args:
            t (E,) floats - time point of evaluation
            x (E, D_sys)  Array  input state
            us (E, D_control)  Control inputs.
                
            where E is the ensemble_size.
        
        Return:
            y (Array) predicted next state
        """
        return self.evaluate_dH_du(self.model, t, xs, us, lambdas_)
    
    @eqx.filter_vmap(in_axes=(None, eqx.if_array(0), None, 0, 0, 0))
    def evaluate_dH_du(self, model, t, x, u, lambda_): 
        """ 
            Evaluate each member of the ensemble on different data for the PMP model. 
            Model, t, x, and u are all vmapped over the ensemble size dimension.
        """
        return jax.grad(lambda u_: jnp.dot(lambda_, model(t=t, x=x, u=u_)))(u)
    
    def evaluate_ensemble_dH_du_shared(self, t:float, xs:Array, us:Array, lambdas_:Array):
        """
        Forward function of f(x,u) for each ensemble member.

        Args:
            t (E,) floats - time point of evaluation
            x (E, D_sys)  Array  input state
            us (D_control)  Control inputs.
                
            where E is the ensemble_size.
        
        Return:
            y (Array) predicted next state
        """
        return self.evaluate_dH_du_shared(self.model, t, xs, us, lambdas_)
    
    @eqx.filter_vmap(in_axes=(None, eqx.if_array(0), None, 0, None, 0))
    def evaluate_dH_du_shared(self, model, t, x, u, lambda_): 
        """ 
            Evaluate each member of the ensemble on different data for the PMP model. 
            Model, t, x, and u are all vmapped over the ensemble size dimension.
        """
        return jax.grad(lambda u_: jnp.dot(lambda_, model(t=t, x=x, u=u_)))(u)
    
    @eqx.filter_vmap
    def make_ensemble(self, key:jr.PRNGKey):
        return NeuralVectorField(D_sys=self.D_sys,
                                D_control=self.D_control,
                                layer_sizes=self.layer_sizes,
                                activation=self.activation,
                                final_activation=self.final_activation, 
                                observation_model=self.observation_model,
                                inverse_observation_model=self.inverse_observation_model,
                                key=key)
