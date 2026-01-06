import dataclasses
import warnings
from collections.abc import Callable
from typing import Any

import math
from typing import cast

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx


def field(
    *,
    converter: Callable[[Any], Any] | None = None,
    static: bool = False,
    **kwargs: Any,
) -> Any:
    """Equinox supports extra functionality on top of the default dataclasses.

    **Arguments:**

    - `converter`: a function to call on this field when the model is initialised. For
        example, `field(converter=jax.numpy.asarray)` to convert
        `bool`/`int`/`float`/`complex` values to JAX arrays. This is ran after the
        `__init__` method (i.e. when using a user-provided `__init__`), and after
        `__post_init__` (i.e. when using the default dataclass initialisation).
        If `converter` is `None`, then no converter is registered.
    - `static`: whether the field should not interact with any JAX transform at all (by
        making it part of the PyTree structure rather than a leaf).
    - `**kwargs`: All other keyword arguments are passed on to `dataclass.field`.

    !!! example "Example for `converter`"

        ```python
        class MyModule(eqx.Module):
            foo: Array = eqx.field(converter=jax.numpy.asarray)

        mymodule = MyModule(1.0)
        assert isinstance(mymodule.foo, jax.Array)
        ```

    !!! example "Example for `static`"

        ```python
        class MyModule(eqx.Module):
            normal_field: int
            static_field: int = eqx.field(static=True)

        mymodule = MyModule("normal", "static")
        leaves, treedef = jax.tree_util.tree_flatten(mymodule)
        assert leaves == ["normal"]
        assert "static" in str(treedef)
        ```

    `static=True` means that this field is not a node of the PyTree, so it does not
    interact with any JAX transforms, like JIT or grad. This means that it is usually a
    bug to make JAX arrays be static fields. `static=True` should very rarely be used.
    It is preferred to just filter out each field with `eqx.partition` whenever you need
    to select only some fields.
    """
    try:
        metadata = dict(kwargs.pop("metadata"))  # safety copy
    except KeyError:
        metadata = {}
    if "converter" in metadata:
        raise ValueError("Cannot use metadata with `converter` already set.")
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    # We don't just use `lambda x: x` as the default, so that this works:
    # ```
    # class Abstract(eqx.Module):
    #     x: int = eqx.field()
    #
    # class Concrete(Abstract):
    #    @property
    #    def x(self):
    #        pass
    # ```
    # otherwise we try to call the default converter on a property without a setter,
    # and an error is raised.
    # Oddities like the above are to be discouraged, of course, but in particular
    # `field(init=False)` was sometimes used to denote an abstract field (prior to the
    # introduction of `AbstractVar`), so we do want to support this.
    if converter is not None:
        metadata["converter"] = converter
    if static:
        metadata["static"] = True
    return dataclasses.field(metadata=metadata, **kwargs)



def default_init(
    key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any, lim: float
) -> jax.Array:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        real_dtype = jnp.finfo(dtype).dtype
        rkey, ikey = jrandom.split(key, 2)
        real = jrandom.uniform(rkey, shape, real_dtype, minval=-lim, maxval=lim)
        imag = jrandom.uniform(ikey, shape, real_dtype, minval=-lim, maxval=lim)
        return real.astype(dtype) + 1j * imag.astype(dtype)
    else:
        return jrandom.uniform(key, shape, dtype, minval=-lim, maxval=lim)

def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


class EideticGRUCell(eqx.Module):
    """A single step of an Eidetic Gated Recurrent Unit (GRU).
    
    This variant removes the reset gate, allowing the cell to retain all 
    information from the previous step ('eidetic' memory). This reduces 
    the parameter count by approximately 1/3 compared to a standard GRU.

    !!! example

        This is often used by wrapping it into a `jax.lax.scan`. For example:

        ```python
        class Model(Module):
            cell: EideticGRUCell

            def __init__(self, **kwargs):
                self.cell = EideticGRUCell(**kwargs)

            def __call__(self, xs):
                scan_fn = lambda state, input: (self.cell(input, state), None)
                init_state = jnp.zeros(self.cell.hidden_size)
                final_state, _ = jax.lax.scan(scan_fn, init_state, xs)
                return final_state
        ```
    """

    weight_ih: Array
    weight_hh: Array
    bias: Array | None
    bias_n: Array | None
    input_size: int = field(static=True)
    hidden_size: int = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `input_size`: The dimensionality of the input vector at each time step.
        - `hidden_size`: The dimensionality of the hidden state passed along between
            time steps.
        - `use_bias`: Whether to add on a bias after each update.
        - `dtype`: The dtype to use for all weights and biases in this GRU cell.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending on
            whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        ihkey, hhkey, bkey, bkey2 = jrandom.split(key, 4)
        lim = math.sqrt(1 / hidden_size)

        # Factor is 2 (Update, New) instead of 3 (Reset, Update, New)
        ihshape = (2 * hidden_size, input_size)
        self.weight_ih = default_init(ihkey, ihshape, dtype, lim)
        hhshape = (2 * hidden_size, hidden_size)
        self.weight_hh = default_init(hhkey, hhshape, dtype, lim)
        if use_bias:
            self.bias = default_init(bkey, (2 * hidden_size,), dtype, lim)
            self.bias_n = default_init(bkey2, (hidden_size,), dtype, lim)
        else:
            self.bias = None
            self.bias_n = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

    def __call__(self, input: Array, hidden: Array, *, key: PRNGKeyArray | None = None):
        """**Arguments:**

        - `input`: The input, which should be a JAX array of shape `(input_size,)`.
        - `hidden`: The hidden state, which should be a JAX array of shape
            `(hidden_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The updated hidden state, which is a JAX array of shape `(hidden_size,)`.
        """
        if self.use_bias:
            bias = cast(Array, self.bias)
            bias_n = cast(Array, self.bias_n)
        else:
            bias = 0
            bias_n = 0
        
        # Split into 2 gates: Update (inp) and New (candidate)
        igates = jnp.split(self.weight_ih @ input + bias, 2)
        hgates = jnp.split(self.weight_hh @ hidden, 2)
        
        # Update gate (z_t)
        inp = jnn.sigmoid(igates[0] + hgates[0])
        
        # Candidate state (n_t) - No reset gate applied here
        # bias_n is applied to the hidden component, maintaining standard GRU bias structure
        new = jnn.tanh(igates[1] + (hgates[1] + bias_n))
        
        return new + inp * (hidden - new)
