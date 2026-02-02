import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax


import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from utils.jax_utils import TrainState, nonpytree_field
from agents.base import BaseAgent

class RandomAgent(BaseAgent):
    
    rng: Any
    config: Any = nonpytree_field()
    env: Any = nonpytree_field()
    env_params: Any = nonpytree_field()
    
    @jax.jit
    def sample_actions(self, observations, rng=None, temperature=1.0):
        #assume no batch, if needed vmap outside
        if rng is  None:
            rng = self.rng
        
        return  self.env.action_space(self.env_params).sample(rng), {}
    

    @classmethod
    def create(cls, seed, env,env_params, config):
        
        rng = jax.random.PRNGKey(seed)
        return cls(
            rng=rng,
            env=env,
            env_params=env_params,
            config=config
        )

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='random',
            num_slow_updates=0,
            target_entropy_multiplier=0.03,
        )
    )
    return config