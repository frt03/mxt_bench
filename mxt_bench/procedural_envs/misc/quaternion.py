import jax

from brax.training.types import PRNGKey
from brax import jumpy as jp
from jax import numpy as jnp


def quat2expmap(quat: jp.ndarray) -> jp.ndarray:
  """Converts a quaternion to an exponential map
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
  Args:
    quat: 4-dim quaternion
  Returns:
    r: 3-dim exponential map
  Raises:
    ValueError if the l2 norm of the quaternion is not close to 1
  """
  # assert jnp.abs(jnp.linalg.norm(quat) - 1) <= 1e-3, 'quat2expmap: input quaternion is not norm 1'

  sinhalftheta = jnp.linalg.norm(quat[1:])
  coshalftheta = quat[0]
  r0 = jnp.divide(quat[1:], (jnp.linalg.norm(quat[1:]) + jnp.finfo(jnp.float32).eps))
  theta = 2 * jnp.arctan2(sinhalftheta, coshalftheta)
  theta = jnp.mod(theta + 2 * jp.pi, 2 * jp.pi)
  r = jax.lax.cond(
    theta > jp.pi,
    lambda x: -r0 * (2 * jp.pi - x),
    lambda x: r0 * x,
    theta)
  return r


def sample_quat_uniform(key: PRNGKey) -> jnp.ndarray:
  # from https://github.com/brentyi/jaxlie/blob/master/jaxlie/_so3.py
  # Uniformly sample over S^3.
  # > Reference: http://planning.cs.uiuc.edu/node198.html
  u1, u2, u3 = jax.random.uniform(
      key=key,
      shape=(3, ),
      minval=jnp.zeros(3),
      maxval=jnp.array([1.0, 2.0 * jnp.pi, 2.0 * jnp.pi]),
  )
  a = jnp.sqrt(1.0 - u1)
  b = jnp.sqrt(u1)

  return jnp.array(
    [
      a * jnp.sin(u2),
      a * jnp.cos(u2),
      b * jnp.sin(u3),
      b * jnp.cos(u3),
      ]
    )


def quat_multiply(quat1: jnp.ndarray, quat2: jnp.ndarray) -> jnp.ndarray:
  # from https://github.com/brentyi/jaxlie/blob/master/jaxlie/_so3.py
  w0, x0, y0, z0 = quat1
  w1, x1, y1, z1 = quat2
  return jnp.array(
    [
      -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
      x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
      -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
      x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
      ]
    )


def quat_inverse(quat: jnp.ndarray) -> jnp.ndarray:
  # return inverse quaternion q^-1
  conjugate = quat * jnp.array([1, -1, -1, -1])
  norm = jnp.linalg.norm(quat)
  return conjugate / (norm**2 + 1e-8)


def quat2eular(quat: jnp.ndarray) ->jnp.ndarray:
  """Computes roll, pitch, and yaw angles.
  Returns:
      Euler angles in radians.
  """
  # from https://github.com/brentyi/jaxlie/blob/master/jaxlie/_so3.py
  # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
  q0, q1, q2, q3 = quat
  roll = jnp.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
  # clipping to avoid nan
  pitch = jnp.arcsin(jnp.clip(2 * (q0 * q2 - q3 * q1), -1.0 + 1e-8, 1.0 - 1e-8))
  yaw = jnp.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
  return jnp.array([roll, pitch, yaw])


def eular2quat(eular: jnp.ndarray) -> jnp.ndarray:
  """Computes quaternions from roll, pitch, and yaw angles in radians.
  Returns:
      Quaternions.
  """
  roll, pitch, yaw = eular

  cr, sr = jnp.cos(roll * 0.5), jnp.sin(roll * 0.5)
  cp, sp = jnp.cos(pitch * 0.5), jnp.sin(pitch * 0.5)
  cy, sy = jnp.cos(yaw * 0.5), jnp.sin(yaw * 0.5)

  w = cr * cp * cy + sr * sp * sy
  x = sr * cp * cy - cr * sp * sy
  y = cr * sp * cy + sr * cp * sy
  z = cr * cp * sy - sr * sp * cy

  return jp.array([w, x, y, z])


def quat2angle_diff(quat1: jnp.ndarray, quat2: jnp.ndarray) -> jnp.ndarray:
  """Compute the angle difference between two quaternion orientations
     as Euler angles in radians.
  """
  inv_quat2 = quat_inverse(quat2)
  quat_diff = quat_multiply(quat1, inv_quat2)
  return quat2eular(quat_diff)
