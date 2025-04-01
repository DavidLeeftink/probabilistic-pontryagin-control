"""
Copyright (c) 2025 David Leeftink
 
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""
from dataclasses import dataclass, field
from diffrax import AbstractGlobalInterpolation
from beartype.typing import Optional, Any
import jax
import jax.numpy as jnp
import diffrax as dfx
from jax import tree_util
from jaxtyping import Num
from jaxtyping import Array, Float
# from simple_pytree import Pytree
from beartype.typing import Callable
import functools

@functools.partial(jax.tree_util.register_dataclass, 
                   data_fields=['ts', 'ys','us', 'ts_dense','indices'], 
                   meta_fields=['control_interpolator', '_original_data_size','standardize_at_initialisation', 
                                'T_scalar', '_original_ys_mean', '_original_ys_std', 
                                '_original_us_mean', '_original_us_std'])
@dataclass
class DiffEqDataset:
    ts: Optional[jnp.ndarray] = None
    ys: Optional[jnp.ndarray] = None
    us: Optional[jnp.ndarray] = None
    ts_dense: Optional[jnp.ndarray] = None
    indices: Optional[jnp.ndarray] = None

    control_interpolator : dfx.AbstractGlobalInterpolation = None
    _original_data_size :int = None
    indices:Array = None
    standardize_at_initialisation:bool = False 
    T_scalar:float = None
    _original_ys_mean = None
    _original_ys_std = None
    _original_us_mean = None
    _original_us_std = None

    def __init__(self, 
                 ts=None, ys=None, us=None, ts_dense=None, indices=None,
                 control_interpolator=None, _original_data_size=None,
                 standardize_at_initialisation=False, T_scalar=None,
                 _original_ys_mean=None, _original_ys_std=None,
                 _original_us_mean=None, _original_us_std=None):
        # Initialize data fields
        self.ts = ts
        self.ys = ys
        self.us = us
        self.ts_dense = ts_dense
        self.indices = indices

        # Initialize static/meta fields
        self.control_interpolator = control_interpolator
        self._original_data_size = _original_data_size
        self.standardize_at_initialisation = standardize_at_initialisation
        self.T_scalar = T_scalar
        self._original_ys_mean = _original_ys_mean
        self._original_ys_std = _original_ys_std
        self._original_us_mean = _original_us_mean
        self._original_us_std = _original_us_std
        # self.__post_init__()
   
    def __post_init__(self) -> None:
        _check_shape(self.ts, self.ys, self.us)

        if self.indices is None:
            self.indices = jnp.arange(self.ys.shape[0])

        if self.ys is not None and self.standardize_at_initialisation:
            self._original_ys_mean = self.mean_ys
            self._original_ys_std = self.std_ys
            self.ys = self.standardize(self.ys)

        if self.ts_dense is not None:
            assert self.ts_dense.shape[0] == self.ys.shape[0], f"Number of sequences of ts_dense ({self.ts_dense}) is not equal to the number of sequences of ys ({self.ys.shape})"
            assert self.ts_dense.shape[1] == self.us.shape[1], f"The dense time points of ts_dense ({self.ts_dense.shape[1]}) do not match the dense timepoints of us ({self.us.shape[1]})"

        if self.ts is not None and self.T_scalar is not None:
            self.ts = self.scale_timepoints(self.ts)

        if self.us is not None and self.standardize_at_initialisation:
            self._original_us_mean = self.mean_us
            self._original_us_std = self.std_us
            self.us = self.standardize_us(self.us)

    def __add__(self, other: "DiffEqDataset") -> "DiffEqDataset":
        r"""Combine two datasets. """
        ts = None
        ys = None
        us = None
        ts_dense = None
        indices = None

        if self.ts is not None and other.ts is not None:            
            ts = jnp.concatenate((self.ts, other.ts))

        if self.ys is not None and other.ys is not None:
            if other._original_ys_mean is not None: #
                other_ys = other.inverse_standardize(other.ys)
                ys = self.inverse_standardize(self.ys)
            else:
                other_ys = other.ys
                ys = self.ys
            ys = jnp.concatenate((ys, other_ys))

        if self.us is not None and other.us is not None:
            if other._original_us_mean is not None: #
                other_us = other.inverse_standardize_us(other.us)
                us= self.inverse_standardize_us(self.us)
            else:
                other_us= other.us
                us = self.us
            us = jnp.concatenate((us, other_us))

        if self.ts_dense is not None and other.ts_dense is not None:
            ts_dense = jnp.concatenate((self.ts_dense, other.ts_dense))

        if self.indices is not None and other.indices is not None:
            indices = jnp.concatenate((self.indices, other.indices+jnp.max(self.indices)))

        return DiffEqDataset(ts=ts, ys=ys, us=us, ts_dense=ts_dense, indices=indices, standardize_at_initialisation=False, control_interpolator=self.control_interpolator)

    @property
    def n(self) -> int:
        r"""Number of sequences/batches."""
        return self.ys.shape[0]

    @property
    def T(self) -> int:
        r"""Number of timepoints per sequence/batch."""
        return self.ys.shape[1]

    @property
    def d_out(self) -> int:
        r"""Dimension of the observations, $`Y`$."""
        return self.ys.shape[-1]
    
    @property
    def d_control(self) -> int:
        r"""Dimensions of the control input, $u(t)$"""
        return self.us.shape[-1]
    
    @property
    def full_data_size(self) ->int:
        r""" The number of datapoints of the full dataset, in case the  """
        if self._original_data_size is not None:
            return self._original_data_size
        else:
            return self.n 

    @property
    def mean_ys(self) ->float:
        r""" Mean of the observed states of all trials"""
        ys = self.ys.reshape(-1, self.ys.shape[-1])
        return jnp.mean(ys, axis=0)
        
    @property
    def std_ys(self) -> float:
        r""" Standard deviation of the observed states of all trials"""
        ys = self.ys.reshape(-1, self.ys.shape[-1])
        return jnp.std(ys, axis=0)
    
    @property
    def mean_us(self) ->float:
        r""" Mean of the control inputs of all trials"""
        us = self.us.reshape(-1, self.us.shape[-1])
        return jnp.mean(us, axis=0)
        
    @property
    def std_us(self) -> float:
        r""" Standard deviation of the control inputs of all trials"""
        us = self.us.reshape(-1, self.us.shape[-1])
        return jnp.std(us, axis=0)

    def standardize(self, ys:Array):
        """
        Apply standardized transformation to the data set: y = (y- E[y])/ \sigma(y)

        Args:
            input_signal (Array) observations of the input signal

        Return:
            standardized_ys (Float[Array]) 
        """
        assert self._original_ys_mean is not None, "No standardization mean is set for this class instance."
        assert self._original_ys_std is not None, "No standardization standard deviation is set for this class instance."
        assert ys.shape[-1] == self._original_ys_mean.shape[-1], f"Given system dimension ({ys.shape[-1]}) is not equal to original mean shape ({self._original_ys_mean.shape[-1]})"
        return (ys-self._original_ys_mean)/self._original_ys_std

    def inverse_standardize(self, ys:Array):
        """
        Apply inverse standardization transformation to a given input signal.

        Args:
            ys (Array) standardized observations or signal

        Return:
            original_signal (Array) input signal destandardized.
        """
        assert self._original_ys_mean is not None, "No standardization mean is set for this class instance."
        assert self._original_ys_std is not None, "No standardization standard deviation is set for this class instance."
        return ys*self._original_ys_std+self._original_ys_mean
    
    def standardize_us(self, us:Array):
        """
        Apply standardized transformation to the data set: y = (y- E[y])/ \sigma(y)

        Args:
            input_signal (Array) observations of the input signal

        Return:
            standardized_ys (Float[Array]) 
        """
        assert self._original_us_mean is not None, "No standardization mean is set for this class instance."
        assert self._original_us_std is not None, "No standardization standard deviation is set for this class instance."
        assert us.shape[-1] == self._original_us_mean.shape[-1], f"Given system dimension ({us.shape[-1]}) is not equal to original mean shape ({self._original_us_mean.shape[-1]})"
        return (us-self._original_us_mean)/self._original_us_std

    def inverse_standardize_us(self, us:Array):
        """
        Apply inverse standardization transformation to a given control input signal.

        Args:
            input_signal (Array) standardized observations or signal

        Return:
            original_signal (Array) input signal destandardized.
        """
        assert self._original_us_mean is not None, "No standardization mean is set for this class instance."
        assert self._original_us_std is not None, "No standardization standard deviation is set for this class instance."
        return us*self._original_us_std+self._original_us_mean
    
    def scale_timepoints(self, ts:Array):
        """
        Scale the observation timepoints between 0 and 1.

        Args:
            ts (Array) original measurement timepoints
        
        Return: 
            ts_standardized (Array) timepoints scaled between 0 and 1.
        """
        assert self.T_scalar is not None, "No time standardization constant is set for this class instance."
        return ts/self.T_scalar
    
    def inverse_scale_timepoints(self, ts_standardized:Array):
        """
        Inverse transform the scaled down observation timepoints to the original observation times.

        Args:
            ts_standardized (Array) timepoints scaled between 0 and 1.    
        
        Return: 
            ts (Array) original measurement timepoints 
        """
        assert self.T_scalar is not None, "No time standardization constant is set for this class instance."
        return ts_standardized*self.T_scalar

    def tree_flatten(self):
        """
        Flatten the data fields and return them along with `None` for static/meta fields.
        The static/meta fields are not included in the flattened structure.
        """
        # Flatten only the data fields (ignore meta fields)
        children = (self.ts, self.ys, self.us, self.ts_dense, self.indices)
        aux_data = None  # Static/meta fields are not needed for flattening
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the data fields and return the object.
        The static/meta fields are set after unflattening.
        """
        # Unpack the children into the data fields
        ts, ys, us, ts_dense, indices = children
        # Create the object with the given data fields and default values for static/meta fields
        return cls(
            ts=ts,
            ys=ys,
            us=us,
            ts_dense=ts_dense,
            indices=indices,
            control_interpolator=None,  # Default value
            _original_data_size=None,  # Default value
            standardize_at_initialisation=False,  # Default value
            T_scalar=None,  # Default value
            _original_ys_mean=None,  # Default value
            _original_ys_std=None,  # Default value
            _original_us_mean=None,  # Default value
            _original_us_std=None,  # Default value
        )


def _check_shape(
        ts: Optional[Num[Array, "..."]], 
        ys: Optional[Num[Array, "..."]],
        us: Optional[Num[Array, "..."]],
    ) -> None:
    r"""Checks that the shapes of $`X`$ and $`y`$ are compatible."""
    if ys is not None and len(ys.shape) != 3:
        raise ValueError(
            f"Expected format for ys is: (b, n, d), while given format is: {ys.shape}"
        )
    
    if ts is not None and len(ts.shape) != 2:
        raise ValueError(
            f"Expected format for ts is: (b, n), while given format is: {ts.shape}"
        )
    
    if ys is not None and ts is not None and ts.shape[1]!=ys.shape[1]:
        raise ValueError(
            f"Time points are not formatted correctly between ts ({ts.shape[1]}) and ys ({ys.shape[1]})."
        )
    
    if ys is not None and ts is not None and ts.shape[0]!=ys.shape[0]:
        raise ValueError(
            f"Number of trials differs between ts ({ts.shape[0]}) and ys ({ys.shape[0]})"
        )
    
    if us is not None and ts is not None and ts.shape[0]!=us.shape[0]:
        raise ValueError(
            f"Number of trials differs between ts ({ts.shape[0]}) and us ({us.shape[0]})"
        )
    

    
__all__ = [
    "DiffEqDataset",
]
