from beartype.typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import equinox as eqx
import jax
import jax.numpy as jnp
from jax._src.random import _check_prng_key
import jax.random as jr
import optax as ox
import ppc
from tqdm import tqdm
from ppc.dataset import DiffEqDataset
from ppc.nn.node import NeuralODE, NeuralVectorField
from jaxtyping import Array

def fit_node(
    *,
    model: eqx.Module,
    objective,
    train_data: DiffEqDataset,
    optim: ox.GradientTransformation,
    key: jr.PRNGKey,
    num_iters: Optional[int] = 100,
    batch_size: Optional[int] = -1,
    log_rate: Optional[int] = 10,
    verbose: Optional[bool] = True,
    unroll: Optional[int] = 1,
    safe: Optional[bool] = True,
    identical_keys:Optional[bool]=False,
):
    r"""Train a Module model with respect to a supplied Objective function.
    Optimisers used here should originate from Optax.

    Code based on gpjax.fit(), and adapted for DiffEqDataset type.

    Args:
        model (eqx.Module): The model Module to be optimised.
        objective (Objective): The objective function that we are optimising with
            respect to.
        train_data (Dataset): The training data to be used for the optimisation.
        optim (GradientTransformation): The Optax optimiser that is to be used for
            learning a parameter set.
        num_iters (Optional[int]): The number of optimisation steps to run. Defaults
            to 100.
        batch_size (Optional[int]): The size of the mini-batch to use. Defaults to -1
            (i.e. full batch).
        key (Optional[KeyArray]): The random key to use for the optimisation batch
            selection. Defaults to jr.PRNGKey(42).
        log_rate (Optional[int]): How frequently the objective function's value should
            be printed. Defaults to 10.
        verbose (Optional[bool]): Whether to print the training loading bar. Defaults
            to True.
        unroll (int): The number of unrolled steps to use for the optimisation.
            Defaults to 1.
        identical_keys (Optional[bool]) if true, use the same jr random key at each training iteration. Default is False.

    Returns
    -------
        Tuple[Module, Array]: A Tuple comprising the optimised model and training
            history respectively.
    """
    if safe:
        # Check inputs.
        _check_model(model)
        _check_train_data(train_data)
        _check_optim(optim)
        _check_num_iters(num_iters)
        _check_batch_size(batch_size)
        _check_prng_key('prng key1', key)
        _check_log_rate(log_rate)
        _check_verbose(verbose)

    # Initialise optimiser state.
    init_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    
    # Unconstrained space loss function with stop-gradient rule for non-trainable params.
    @eqx.filter_value_and_grad
    def loss_func(model, batch: DiffEqDataset, key:jr.PRNGKey):
        return objective(model, batch, key)

    # Mini-batch random keys to scan over.
    iter_keys = jr.split(key, num_iters) if not identical_keys else jnp.tile(key, (num_iters, 1))

    # Optimisation step.
    @eqx.filter_jit
    def step(carry, key):
        model, opt_state = carry
        key1, key2 = jr.split(key)

        if batch_size != -1:
            batch = get_batch(train_data, batch_size, key1)
        else:
            batch = train_data

        loss_val, loss_gradient = loss_func(model, batch, key2)
        updates, opt_state = optim.update(loss_gradient, opt_state, model)
        model = eqx.apply_updates(model, updates)
    
        carry = model, opt_state
        return carry, loss_val

    # Optimisation.
    carry = model, init_state
    history = jnp.zeros((num_iters))
    
    progress_bar = tqdm(range(num_iters))
    for i in range(num_iters):
        carry, loss = step(carry, iter_keys[i])
        history =  history.at[i].set(loss)
        if i%log_rate==0:
            progress_bar.set_postfix({"loss": f"{loss:.4f}"}, refresh=False)
            progress_bar.update(log_rate if i + log_rate <= num_iters else num_iters - i)

    model, _ = carry
    return model, history

def get_batch(train_data: DiffEqDataset, batch_size: int, key: jr.PRNGKey, replace=False) -> DiffEqDataset:
    """Batch the data into mini-batches. Sampling is done with replacement.
        Function from gpjax, adapted for DiffEqdata type.

    Args:
        train_data (DiffEqDataset): The training dataset.
        batch_size (int): The batch size.
        key (KeyArray): The random key to use for the batch selection.

    Returns
    -------
        Dataset: The batched dataset.
    """
    ys, ts, us, ts_dense = train_data.ys, train_data.ts, train_data.us, train_data.ts_dense
    # if train_data._original_ys_mean is not None:
    #     ys = train_data.inverse_standardize(ys)
    # if train_data._original_us_mean is not None:
    #     us = train_data.inverse_standardize_us(us)        
    n = train_data.n

    # Subsample mini-batch indices with/without replacement.
    indices = jr.choice(key, n, (batch_size,), replace=replace)

    return DiffEqDataset(ts=ts[indices], 
                         ys=ys[indices], 
                         us=us[indices] if us is not None else us,
                         ts_dense=ts_dense[indices] if ts_dense is not None else ts_dense,
                         standardize_at_initialisation=False,
                         _original_data_size=n,
                         indices=indices)


def _check_model(model: Any) -> None:
    """Check that the model is of type Module. Check trainables and bijectors tree structure."""
    pass


def _check_train_data(train_data: Any) -> None:
    """Check that the train_data is of type DiffEqDataset."""
    if not isinstance(train_data, DiffEqDataset):
        raise TypeError(f"train_data must be of type gpdx.DiffEqDataset but is: {type(train_data)}")


def _check_optim(optim: Any) -> None:
    """Check that the optimiser is of type GradientTransformation."""
    if not isinstance(optim, ox.GradientTransformation):
        raise TypeError("optax_optim must be of type optax.GradientTransformation")


def _check_num_iters(num_iters: Any) -> None:
    """Check that the number of iterations is of type int and positive."""
    if not isinstance(num_iters, int):
        raise TypeError("num_iters must be of type int")

    if not num_iters > 0:
        raise ValueError("num_iters must be positive")
    

def _check_log_rate(log_rate: Any) -> None:
    """Check that the log rate is of type int and positive."""
    if not isinstance(log_rate, int):
        raise TypeError("log_rate must be of type int")

    if not log_rate > 0:
        raise ValueError("log_rate must be positive")


def _check_verbose(verbose: Any) -> None:
    """Check that the verbose is of type bool."""
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be of type bool")


def _check_batch_size(batch_size: Any) -> None:
    """Check that the batch size is of type int and positive if not minus 1."""
    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be of type int")

    if not batch_size == -1 and not batch_size > 0:
        raise ValueError("batch_size must be positive")


__all__ = [
    "fit",
    "get_batch",
]
