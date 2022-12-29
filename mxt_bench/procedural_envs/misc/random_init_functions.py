from typing import Tuple

import tensorflow_probability as tfp
import jax
import jax.numpy as jnp

from procedural_envs.misc.quaternion import eular2quat
from procedural_envs.misc.quaternion import quat2eular
from procedural_envs.misc.quaternion import sample_quat_uniform

tfp = tfp.substrates.jax
tfd = tfp.distributions


def annulus_xy_sampler(rng: jnp.ndarray, r_min: float, r_max: float, init_z: float = 0.0):
  low = jnp.ones(1)*(r_min**2)
  high =  jnp.ones(1)*(r_max**2)
  r_sample_fn = lambda: tfd.Uniform(low=low, high=high)
  theta_sample_fn = lambda: tfd.Uniform(low=jnp.zeros(1), high=jnp.ones(1))

  rng1, rng2 = jax.random.split(rng, num=2)
  init_r = r_sample_fn().sample(seed=rng1)
  init_theta = theta_sample_fn().sample(seed=rng2) * 2 * jnp.pi
  init_x = jnp.sqrt(init_r) * jnp.cos(init_theta)
  init_y = jnp.sqrt(init_r) * jnp.sin(init_theta)
  init_pos = jnp.concatenate([init_x, init_y])
  # xy -> xyz
  init_pos = jax.ops.index_update(jnp.zeros(3), jnp.index_exp[0:2], init_pos)
  init_pos = jax.ops.index_update(init_pos, jnp.index_exp[2], jnp.ones(())*init_z)
  return init_pos


def rectangle_xy_sampler(rng: jnp.ndarray,
                         x_range: Tuple[float, float],
                         y_range: Tuple[float, float],
                         init_z: float = 0.0,
                         const=None):
  x_sample_fn = lambda: tfd.Uniform(
    low=jnp.ones(1)*x_range[0], high=jnp.ones(1)*x_range[1])
  y_sample_fn = lambda: tfd.Uniform(
    low=jnp.ones(1)*y_range[0], high=jnp.ones(1)*y_range[1])

  rng1, rng2 = jax.random.split(rng, num=2)
  if const == 'x':
    init_x = jnp.ones(1) * x_range[0]
  else:
    init_x = x_sample_fn().sample(seed=rng1)
  if const == 'y':
    init_y = jnp.ones(1) * y_range[0]
  else:
    init_y = y_sample_fn().sample(seed=rng2)
  init_pos = jnp.concatenate([init_x, init_y])
  # xy -> xyz
  init_pos = jax.ops.index_update(jnp.zeros(3), jnp.index_exp[0:2], init_pos)
  init_pos = jax.ops.index_update(init_pos, jnp.index_exp[2], jnp.ones(())*init_z)
  return init_pos


def circular_sector_xy_sampler(rng: jnp.ndarray,
                               r_min: float,
                               r_max: float,
                               theta_min: float,
                               theta_max: float,
                               ref_theta: jnp.ndarray,
                               init_z: float = 0.0):
  low = jnp.ones(1)*(r_min**2)
  high = jnp.ones(1)*(r_max**2)
  r_sample_fn = lambda: tfd.Uniform(low=low, high=high)
  t_low = jnp.ones(1)*theta_min
  t_high = jnp.ones(1)*theta_max
  theta_sample_fn = lambda: tfd.Uniform(low=t_low, high=t_high)

  rng1, rng2 = jax.random.split(rng, num=2)
  init_r = r_sample_fn().sample(seed=rng1)
  init_theta = theta_sample_fn().sample(seed=rng2) * 2 * jnp.pi + ref_theta
  init_x = jnp.sqrt(init_r) * jnp.cos(init_theta)
  init_y = jnp.sqrt(init_r) * jnp.sin(init_theta)
  init_pos = jnp.concatenate([init_x, init_y])
  # xy -> xyz
  init_pos = jax.ops.index_update(jnp.zeros(3), jnp.index_exp[0:2], init_pos)
  init_pos = jax.ops.index_update(init_pos, jnp.index_exp[2], jnp.ones(())*init_z)
  return init_pos


def roll_pitch_yaw_uniform_sampler(rng: jnp.ndarray):
  init_eular = quat2eular(sample_quat_uniform(key=rng))
  init_rot = eular2quat(init_eular)
  return init_rot


def pitch_yaw_uniform_sampler():
  init_eular = quat2eular(sample_quat_uniform(key=rng))
  # roll -> 0.
  init_eular = jax.ops.index_update(init_eular, jnp.index_exp[0:1], jnp.zeros(1))
  init_rot = eular2quat(init_eular)
  return init_rot


def yaw_uniform_sampler():
  init_eular = quat2eular(sample_quat_uniform(key=rng))
  # roll, pitch -> 0.
  init_eular = jax.ops.index_update(init_eular, jnp.index_exp[0:2], jnp.zeros(2))
  init_rot = eular2quat(init_eular)
  return init_rot


def uniform_z_sampler(rng: jnp.ndarray, z_min: float, z_max: float, const: bool = False):
  low = jnp.ones(())*z_min
  high =  jnp.ones(())*z_max
  if not const:
    z_sample_fn = lambda: tfd.Uniform(low=low, high=high)
    init_z = z_sample_fn().sample(seed=rng)
  else:
    init_z = low
  # z -> xyz
  init_pos = jnp.zeros(3).at[jnp.index_exp[2]].set(init_z)
  return init_pos
