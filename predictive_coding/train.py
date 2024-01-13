from typing import List

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from ml_collections import ConfigDict

from flax.training import TrainState

from .predictive_coder import PredictiveCoder
from .predictive_coder import make_global_energy, make_loss, infer_activations_shape
from .utils import get_optimizer, get_loss_fn, in_jit


class Trainer:

  def __init__(self,
               config: ConfigDict,
               model: PredictiveCoder):
    self.cfg = config
    self.optimizer = get_optimizer(self.cfg)
    self.energy_fn = get_loss_fn(self.cfg.energy_fn)
    self.target_energy_fn = get_loss_fn(self.cfg.energy_fn)
    self.global_energy = make_global_energy(self.energy_fn, self.loss_fn)
    self.loss = make_loss(self.global_energy)

    self.model = model

    self.key = jax.random.PRNGKey(self.cfg.key)
    self.batch_size = self.cfg.batch_size

  def stateful_subkey(self):
    assert in_jit(), "The key generator should not be traced."
    self.key, subkey = jax.random.split(self.key)
    return subkey

  def create_activations(self) -> List[jnp.ndarray]:
    return [jnp.zeros((self.batch_size,)+shape) for shape in self.shapes]

  def train_step(self,
                 train_state: TrainState,
                 train_batch):
      """Train a predictive coding model for one step."""
      x, y = train_batch
      activations = self.create_activations()
      train_state, loss = self.loss(train_state, x, y, activations)
      return train_state, loss

  def train_epoch(self,
                  train_state: TrainState,
                  ds_train):
    """Train a predictive coding model for one epoch."""
    for train_batch in ds_train:
      train_state, loss = self.train_step(train_state, train_batch)
    return train_state

  def train(self,
            ds_train: tfds.data.Dataset,
            params: flax.core.FrozenDict = None):
    """Train a predictive coding model.

    Args:
      ds_train: a tf.data.Dataset with (x, y) pairs as NumPy iterables.
      params: a flax.core.FrozenDict (default: None).
    """
    key = self.stateful_subkey()

    x0 = next(iter(ds_train))[0]

    self.shapes = infer_activations_shape(self.model, x0)
    activations = self.create_activations()

    if params is None:
      params = self.model.init(key, activations)

    train_state = TrainState.create(
        apply_fn=self.model.apply,
        params=params,
        tx=self.optimizer,
    )
    
    for epoch in range(self.cfg.num_epochs):
      train_state = self.train_epoch(train_state, ds_train)
