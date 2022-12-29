import jax
import jax.numpy as jnp
from typing import Tuple
from flax import linen

from brax.training import networks


def make_mlp_policy_network(policy_params_size: int,
                            obs_size: int,
                            hidden_layer_sizes: Tuple[int, ...] = (256, 256),
                            ) -> networks.FeedForwardModel:
  """Creates a policy for BC."""
  policy_module = networks.MLP(
      layer_sizes=hidden_layer_sizes + (policy_params_size,),
      activation=linen.relu,
      kernel_init=jax.nn.initializers.lecun_uniform())
  dummy_obs = jnp.zeros((1, obs_size))
  policy = networks.FeedForwardModel(
      init=lambda key: policy_module.init(key, dummy_obs),
      apply=policy_module.apply)
  return policy
