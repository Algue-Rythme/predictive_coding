import jax
import jax.numpy as jnp


def in_jit():
  return isinstance(jnp.array(0), jax.core.Tracer)