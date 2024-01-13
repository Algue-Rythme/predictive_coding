from typing import Any, Callable, List

import jax
import flax.linen as nn
import jax.numpy as jnp


class ActivationShapeInferencer(nn.Module):
  """A flax.linen.Module that infers the shape of the activations of a network.

  Attributes:
    layers: a list of flax.linen.Module

  Returns:
    a list of shapes for each activation (without the batch dimension).
  """
  layers: List[Callable]

  def setup(self):
    pass
  
  def __call__(self, x):
    shapes = []
    for layer in self.layers:
      x = layer(x)
      shapes.append(x.shape[1:])
    return shapes

def infer_activations_shape(network: nn.Module,
                            x0: jnp.ndarray
                            ) -> List[Any]:
  """Infer the shape of the activations of a network given an input.
  
  Args:
    network: a flax.linen.Module
    x0: an input to the network without the batch dimension.
  
  Returns:
    a list of shapes for each activation.
  """
  x0 = jnp.expand_dims(x0, axis=0)  # dummy batch dimension
  model = ActivationShapeInferencer(network.layers)
  key = jax.random.PRNGKey(0)
  params = model.init(key, x0)
  shapes = model.apply(params, x0)
  return shapes


class FeedforwardPCNetwork(nn.Module):
  """A predictive coding network.

  Attributes:
    layers: a list of flax.linen.Module

  Returns:
    a list of activations for each layer.
  """
  layers: List[Callable]

  @nn.compact
  def __call__(self, xs):
    ys = []
    for x, layer in zip(xs, self.layers):
      y = layer(x)
      ys.append(y)
    return ys


def make_global_energy(
    energy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    target_energy: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Create a global energy function for a network.

  Args:
    energy_fn: a function that takes a pair of activations ad returns a scalar.
    target_energy: a function that takes a pair of activations and returns a scalar.

  Returns:
    a function that takes a list of activations and returns a scalar.
  """

  def global_energy(activations: List[jnp.ndarray],
                    targets: List[jnp.ndarray]) -> jnp.ndarray:
    """Compute the global energy of a network.

    Args:
      activations: a list of activations for each layer.
      targets: a list of targets for each layer.

    Returns:
      a scalar.
    """
    total_energy = target_energy(activations[-1], targets[-1])
    for activation, target in zip(activations[:-1], targets[:-1]):
      if activation.shape != target.shape:
        raise ValueError('activation and target shapes do not match.')
      energy = energy_fn(activation, target)
      total_energy = total_energy + energy
    return total_energy

  return global_energy


def make_ps_loss(network: nn.Module,
                 global_energy: Callable[[List[jnp.ndarray], List[jnp.ndarray]],
                                        jnp.ndarray],
                 ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  """Create a loss function for a network.

  Args:
    network: a flax.linen.Module
    global_energy: a function that takes a list of activations and returns a scalar.

  Returns:
    a function that takes a batch of inputs and targets and returns a scalar.
  """

  batched_energy = jax.vmap(global_energy, in_axes=(0, 0))

  def loss_fn(params: Any,
              inputs: jnp.ndarray,
              targets: jnp.ndarray) -> jnp.ndarray:
    """Compute the loss of a network.

    Args:
      params: the parameters of the network.
      inputs: a batch of inputs.
      targets: a batch of targets.

    Returns:
      a scalar.
    """
    activations = network.apply(params, inputs)
    energies = batched_energy(activations, targets)
    return jnp.mean(energies)

  return loss_fn
