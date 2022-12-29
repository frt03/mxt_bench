import brax
from brax import math
from procedural_envs.misc import sim_utils
from jax import numpy as jnp


def upright_term_fn(done, sys, qp: brax.QP, info: brax.Info, component):
  """Terminate when it falls."""
  del info
  # upright termination
  index = sim_utils.names2indices(sys.config, component['root'], 'body')[0][0]
  rot = qp.rot[index]
  up = jnp.array([0., 0., 1.])
  torso_up = math.rotate(up, rot)
  torso_is_up = jnp.dot(torso_up, up)
  done = jnp.where(torso_is_up < 0.0, x=1.0, y=done)
  return done


def height_term_fn(done,
                   sys,
                   qp: brax.QP,
                   info: brax.Info,
                   component,
                   max_height: float = 1.0,
                   min_height: float = 0.2):
  """Terminate when it flips or jumps too high."""
  del info
  # height termination
  z_offset = component.get('term_params', {}).get('z_offset', 0.0)
  index = sim_utils.names2indices(sys.config, component['root'], 'body')[0][0]
  z = qp.pos[index][2]
  done = jnp.where(z < min_height + z_offset, x=1.0, y=done)
  done = jnp.where(z > max_height + z_offset, x=1.0, y=done)
  return done
