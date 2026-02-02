from gymnax.experimental import RolloutWrapper

import functools
import gymnax
from typing import Union,Optional,Any
import abc

import jax
import jax.numpy as jnp

from gymnax.environments.environment import Environment
from agents.base import BaseAgent


import jax
import jax.numpy as jnp


from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper
from brax import envs
from brax.envs import _envs as brax_envs
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper



class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional", **env_kwargs):
        # parse params
        task_params = self._parse_task_params(env_name, env_kwargs)
        
        env = envs.get_environment(env_name=task_params['env_name'], backend=backend, **task_params['kwargs'])
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self._task_name = env_name  

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )
    def get_obs(self, state, params):
        return self._env._get_obs(state.pipeline_state)
    
    def _parse_task_params(self, env_name, env_kwargs):

        task_kwargs = {}

        if env_name.startswith('walker'):
            if env_name == 'walker':
                task_kwargs = {'move_speed': 1.0, 'forward': True, 'flip': False}
            elif env_name == 'walker-stand':
                task_kwargs = {'move_speed': 0.0, 'forward': True, 'flip': False}
            elif env_name == 'walker-walk':
                task_kwargs = {'move_speed': 1.0, 'forward': True, 'flip': False}
            elif env_name == 'walker-run':
                task_kwargs = {'move_speed': 8.0, 'forward': True, 'flip': False}
            elif env_name == 'walker-flip':
                task_kwargs = {'move_speed': 0.0, 'forward': True, 'flip': True}
            elif env_name == 'walker-backward':
                task_kwargs = {'move_speed': 1.0, 'forward': False, 'flip': False}
            
            task_kwargs.update(env_kwargs)
            return {'env_name': 'walker', 'kwargs': task_kwargs}
        
        return {'env_name': env_name, 'kwargs': env_kwargs}



class CustomRolloutWrapper:
    """Wrapper to define batch evaluation for generation parameters."""

    def __init__(
        self,
        env_or_name: Union[str,Environment] = "Pendulum-v1",
        num_env_steps: Optional[int] = None,
        env_kwargs: Any | None = None,
        env_params: Any | None = None,
    ):
        """Wrapper to define batch evaluation for generation parameters."""
        # Define the RL environment & network forward function
        if env_kwargs is None:
            env_kwargs = {}
        if env_params is None:
            env_params = {}
        if isinstance(env_or_name,Environment):
            self.env = env_or_name
            self.env_params = env_or_name.default_params
        # brax walker multi-task support
        elif env_or_name.startswith('walker') or env_or_name.startswith('ant'):
            self.env = BraxGymnaxWrapper(env_or_name, backend='positional', **env_kwargs)
            self.env_params = gymnax.environments.environment.EnvParams(max_steps_in_episode=1000)
        # mjx environment support - check if it's a registry name or custom format
        elif (MJX_AVAILABLE and 
              (env_or_name.startswith('mjx-') or env_or_name.startswith('mjx_') or 
               env_or_name in registry.ALL_ENVS)):
            self.env = MjxGymnaxWrapper(env_or_name, **env_kwargs)
            self.env_params = MjxEnvParams(max_steps_in_episode=1000)
        else:    
            self.env, self.env_params = gymnax.make(env_or_name, **env_kwargs)
        if env_params is not None:
            self.env_params = self.env_params.replace(**env_params)

        if num_env_steps is None:
            self.num_env_steps = self.env_params.max_steps_in_episode
        else:
            self.num_env_steps = num_env_steps

    def batch_reset(self,rng_input):
        batch_reset = jax.vmap(self.single_reset_state)
        return batch_reset(rng_input)

    def single_reset_state(self,rng_input):
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)
        return state

    @functools.partial(jax.jit, static_argnums=(0, 4,5))
    def batch_rollout(self, rng_eval, model:BaseAgent,
                      env_state=None,num_steps=None,temperature=1.0):
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0, None,0,None,None))
        return batch_rollout(rng_eval, model, env_state,num_steps,temperature)  

    @functools.partial(jax.jit, static_argnums=(0,4,5))
    def single_rollout(self, rng_input, model:BaseAgent,
                       env_state=None,num_steps=None,temperature=1.0):
        """Rollout a pendulum episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)

        if env_state is None:
            obs, env_state = self.env.reset(rng_reset, self.env_params)
        else:
            obs = self.env.get_obs(env_state, self.env_params)

        def policy_step(state_input, _=None):
            """lax.scan compatible step transition in jax env."""
            obs, state,  rng, cum_reward, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            if model is not None:
                action,info = model(obs, rng_net,temperature=temperature)
            else:
                action = self.env.action_space(self.env_params).sample(rng_net)
                info = {}
            next_obs, next_state, reward, done, step_info = self.env.step(
                rng_step, state, action, self.env_params
            )
            info.update(step_info)
            # not used for training
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_obs,
                next_state,
                rng,
                new_cum_reward,
                new_valid_mask,
            ]
            y = [obs, action, reward, next_obs, done, state, info]
            return carry, y
            
        if num_steps is not None:
            # Scan over episode step loop
            carry_out, scan_out = jax.lax.scan(
                policy_step,
                [
                    obs,
                    env_state,
                    rng_episode,
                    jnp.array([0.0]),
                    jnp.array([1.0]),
                ],
                (),
                num_steps,
            )
            # Return the sum of rewards accumulated by agent in episode rollout
            obs, action, reward, next_obs, done, state, info = scan_out
            cum_return = carry_out[-2]
            return obs, action, reward, next_obs, done,state, info, cum_return,  carry_out[1]
        else:
            def while_policy_step(state_input):
                """lax.scan compatible step transition in jax env."""
                obs, state, rng, cum_reward, valid_mask = state_input
                rng, rng_step, rng_net = jax.random.split(rng, 3)
                if model is not None:
                    action,info = model(obs, rng_net,temperature=0.001)
                else:
                    action = self.env.action_space(self.env_params).sample(rng_net)
                    info = {}
                next_obs, next_state, reward, done, step_info = self.env.step(
                    rng_step, state, action, self.env_params
                )
                info.update(step_info)
                new_cum_reward = cum_reward + reward * valid_mask
                new_valid_mask = valid_mask * (1 - done)
                carry = [
                    next_obs,
                    next_state,
                    rng,
                    new_cum_reward,
                    new_valid_mask,
                ]
                return carry
            # Scan over episode step loop
            carry_out = jax.lax.while_loop(
                lambda state: jnp.logical_not(jnp.all(state[-1])),
                while_policy_step,
                [
                    obs,
                    env_state,
                    rng_episode,
                    jnp.array([0.0]),
                    jnp.array([1.0]),
                ],  
            )
            # Return the sum of rewards accumulated by agent in episode rollout
            next_obs, next_state,rng,  cum_return,  new_valid_mask = carry_out
            return cum_return