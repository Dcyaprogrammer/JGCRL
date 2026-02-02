import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.jax_utils import nonpytree_field, Transition

class BaseAgent(flax.struct.PyTreeNode):
  rng: Any
  config: Any = nonpytree_field()

  @jax.jit
  def total_loss(self, batch, grad_params, rng=None):
    pass

  @jax.jit
  def update(self, batch, fast=True, info=None):
      return self, {}

  @jax.jit
  def sample_actions(self, observations, rng=None,temperature=1.0):
    pass
  
  def __call__(self, observations, rng=None,temperature=1.0):
      return self.sample_actions(observations, rng,temperature) 
      
  @classmethod
  def create(cls, rng, env, env_params, config):
    pass

def get_config():
  config = ml_collections.ConfigDict(
      dict(
          agent_name='base',
          batch_size=256,
          )
  )
  return config
