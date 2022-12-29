import jax
import jax.numpy as jnp
from flax import linen
from typing import Any, Callable


class PositionalEncoding(linen.Module):
  """PositionalEncoding module."""
  d_model: int
  seq_len: int
  dropout_rate: float = 0.1
  kernel_init: Callable[..., Any] = jax.nn.initializers.normal(stddev=1.0)
  deterministic: bool = False if dropout_rate > 0.0 else True

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    # (B, L, O)
    pe = self.param(
        'pe',
        self.kernel_init,
        (data.shape[1], self.d_model),
        jnp.float32)
    output = data + pe
    output = linen.Dropout(
        rate=self.dropout_rate,
        deterministic=self.deterministic)(output)
    return output


class PositionalEncoding1D(linen.Module):
  """PositionalEncoding1D module."""
  d_model: int
  seq_len: int
  dropout_rate: float = 0.1
  deterministic: bool = False if dropout_rate > 0.0 else True

  def setup(self):
    assert self.d_model % 2 == 0
    position = jnp.arange(self.seq_len).reshape(self.seq_len, 1)
    div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model))
    self.pe = jnp.zeros((self.seq_len, self.d_model))
    self.pe = self.pe.at[jnp.index_exp[:, 0::2]].set(jnp.sin(position * div_term))
    self.pe = self.pe.at[jnp.index_exp[:, 1::2]].set(jnp.cos(position * div_term))
    self.pe = jax.lax.stop_gradient(self.pe)

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    # (B, L, O)
    output = data + self.pe
    output = linen.Dropout(
        rate=self.dropout_rate,
        deterministic=self.deterministic)(output)
    return output
