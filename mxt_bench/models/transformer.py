"""Transformer Encoder
Reference: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
"""
from typing import Any, Callable, Optional, Tuple
from flax import linen
from flax.linen.initializers import lecun_normal, zeros
import jax
import jax.numpy as jnp

from brax.training.networks import FeedForwardModel
from models.attention import MultiHeadDotProductAttentionWithWeight
from models.positional_encoding import PositionalEncoding


class TransformerEncoderLayer(linen.Module):
  """TransformerEncoderLayer module."""
  d_model: int
  num_heads: int
  dim_feedforward: int
  dropout_rate: float = 0.1
  dtype: Any = jnp.float32
  qkv_features: Optional[int] = None
  activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
  kernel_init: Callable[..., Any] = lecun_normal()
  bias_init: Callable[..., Any] = zeros
  deterministic: bool = False if dropout_rate > 0.0 else True

  @linen.compact
  def __call__(
      self,
      src: jnp.ndarray,
      src_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    src2, attn_weights = MultiHeadDotProductAttentionWithWeight(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_features,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=True,
        broadcast_dropout=False,
        dropout_rate=self.dropout_rate)(src, src, mask=src_mask)
    src = src + linen.Dropout(
        rate=self.dropout_rate,
        deterministic=self.deterministic)(src2)
    src = linen.LayerNorm(dtype=self.dtype)(src)
    src2 = linen.Dense(
        self.dim_feedforward,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(src)
    src2 = self.activation(src2)
    src2 = linen.Dropout(
        rate=self.dropout_rate,
        deterministic=self.deterministic)(src2)
    src2 = linen.Dense(
        self.d_model,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(src2)
    src = src + linen.Dropout(
        rate=self.dropout_rate,
        deterministic=self.deterministic)(src2)
    src = linen.LayerNorm(dtype=self.dtype)(src)
    return src, attn_weights


class TransformerEncoder(linen.Module):
  """TransformerEncoder module."""
  num_layers: int
  d_model: int
  num_heads: int
  dim_feedforward: int
  norm: Optional[Callable[..., Any]] = None
  dropout_rate: float = 0.1
  dtype: Any = jnp.float32
  qkv_features: Optional[int] = None
  activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
  kernel_init: Callable[..., Any] = lecun_normal()
  bias_init: Callable[..., Any] = zeros

  @linen.compact
  def __call__(
      self,
      src: jnp.ndarray,
      src_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
      # NOTE: Shape of attn_weights: Batch Size x MAX_JOINTS x MAX_JOINTS
      attn_weights = []
      output = src
      for _ in range(self.num_layers):
          output, attn_weight = TransformerEncoderLayer(
              d_model=self.d_model,
              num_heads=self.num_heads,
              dim_feedforward=self.dim_feedforward,
              dropout_rate=self.dropout_rate,
              dtype=self.dtype,
              qkv_features=self.qkv_features,
              activation=self.activation,
              kernel_init=self.kernel_init,
              bias_init=self.bias_init)(output, src_mask)
          attn_weights.append(attn_weight)
      if self.norm is not None:
          output = self.norm(dtype=self.dtype)(output)
      return output, attn_weights


class TransformerModel(linen.Module):
  """Transformer Policy/Critic"""
  num_layers: int
  d_model: int
  num_heads: int
  dim_feedforward: int
  output_size: int
  dropout_rate: float = 0.5
  transformer_norm: bool = False
  condition_decoder: bool = False

  @linen.compact
  def __call__(self, data: jnp.ndarray, src_mask: jnp.ndarray = None):
    input_size = data.shape[-1]
    # encoder
    output = linen.Dense(
      self.d_model,
      kernel_init=jax.nn.initializers.uniform(scale=0.1),
      bias_init=linen.initializers.zeros)(
        data) * jnp.sqrt(input_size)
    output, attn_weights = TransformerEncoder(
      num_layers=self.num_layers,
      norm=linen.LayerNorm if self.transformer_norm else None,
      d_model=self.d_model,
      num_heads=self.num_heads,
      dim_feedforward=self.dim_feedforward,
      dropout_rate=self.dropout_rate)(output, src_mask)
    if self.condition_decoder:
      output = jnp.concatenate([output, data], axis=-1)
    # decoder
    output = linen.Dense(
      self.output_size,
      kernel_init=jax.nn.initializers.uniform(scale=0.1),
      bias_init=linen.initializers.zeros)(output)
    return output, attn_weights


class TransformerPEModel(linen.Module):
  """Transformer Policy/Critic"""
  num_layers: int
  d_model: int
  num_heads: int
  dim_feedforward: int
  output_size: int
  dropout_rate: float = 0.5
  transformer_norm: bool = False
  condition_decoder: bool = False

  @linen.compact
  def __call__(self, data: jnp.ndarray, src_mask: jnp.ndarray = None):
    # (B, L, O)
    input_size = data.shape[-1]
    seq_len = data.shape[1]
    # encoder
    output = linen.Dense(
      self.d_model,
      kernel_init=jax.nn.initializers.uniform(scale=0.1),
      bias_init=linen.initializers.zeros)(
        data) * jnp.sqrt(input_size)
    output = PositionalEncoding(
      d_model=self.d_model, seq_len=seq_len, dropout_rate=self.dropout_rate)(output)
    output, attn_weights = TransformerEncoder(
      num_layers=self.num_layers,
      norm=linen.LayerNorm if self.transformer_norm else None,
      d_model=self.d_model,
      num_heads=self.num_heads,
      dim_feedforward=self.dim_feedforward,
      dropout_rate=self.dropout_rate)(output, src_mask)
    if self.condition_decoder:
      output = jnp.concatenate([output, data], axis=-1)
    # decoder
    output = linen.Dense(
      self.output_size,
      kernel_init=jax.nn.initializers.uniform(scale=0.1),
      bias_init=linen.initializers.zeros)(output)
    return output, attn_weights


def make_transformer(output_size: int,
                     num_layers: int = 3,
                     d_model: int = 128,
                     num_heads: int = 2,
                     dim_feedforward: int = 256,
                     dropout_rate: float = 0.0,
                     transformer_norm: bool = True,
                     condition_decoder: bool = True,
                     positional_encoding: bool = False) -> TransformerModel:
  """Creates a transformer model (https://arxiv.org/abs/2010.01856).
  Args:
    layer_sizes: layers
    obs_size: size of an observation
    output_size: size of an output (for policy)
    num_layers: number of layers in TransformerEncoder
    d_model: size of an input for TransformerEncoder
    num_heads: number of heads in the multiheadattention
    dim_feedforward: the dimension of the feedforward network model
    dropout_rate: the dropout value
    transformer_norm: whether to use a layer normalization
    condition_decoder: whether to concat the features of the joint
  Returns:
    a model
  """
  if not positional_encoding:
    module = TransformerModel(
      num_layers=num_layers,
      d_model=d_model,
      num_heads=num_heads,
      dim_feedforward=dim_feedforward,
      output_size=output_size,
      dropout_rate=dropout_rate,
      transformer_norm=transformer_norm,
      condition_decoder=condition_decoder)
  else:
    module = TransformerPEModel(
      num_layers=num_layers,
      d_model=d_model,
      num_heads=num_heads,
      dim_feedforward=dim_feedforward,
      output_size=output_size,
      dropout_rate=dropout_rate,
      transformer_norm=transformer_norm,
      condition_decoder=condition_decoder)
  return module


def make_transformers(policy_params_size: int,
                      obs_size: int,
                      action_size: int,
                      max_num_limb: int,
                      num_layers: int = 3,
                      d_model: int = 128,
                      num_heads: int = 2,
                      dim_feedforward: int = 256,
                      dropout_rate: float = 0.0,
                      transformer_norm: bool = True,
                      condition_decoder: bool = True,
                      positional_encoding: bool = False,
                      ) -> Tuple[FeedForwardModel, FeedForwardModel]:
  """Creates transformer models for policy and value functions,
     following https://arxiv.org/abs/2010.01856.
  Args:
    policy_params_size: number of params that a policy network should generate
    obs_size: size of an observation
  Returns:
    a model for policy and a model for value function
  """
  dummy_obs = jnp.zeros((1, 1, obs_size))
  dummy_action = jnp.zeros((1, 1, action_size))
  if positional_encoding:
    dummy_obs = jnp.zeros((1, max_num_limb, obs_size))
    dummy_action = jnp.zeros((1, max_num_limb, 1))

  def policy_model_fn():
    class PolicyModule(linen.Module):
      @linen.compact
      def __call__(self, data: jnp.ndarray, mask: jnp.ndarray = None):
        # (B, L, O)
        output, attn_weights = make_transformer(
          output_size=policy_params_size,
          num_layers=num_layers,
          d_model=d_model,
          num_heads=num_heads,
          dim_feedforward=dim_feedforward,
          dropout_rate=dropout_rate,
          transformer_norm=transformer_norm,
          condition_decoder=condition_decoder,
          positional_encoding=positional_encoding,
        )(data, mask) # (B, 1, P) P: number of distribution parameters
        output = output.reshape((data.shape[0], -1))
        return output, attn_weights

    module = PolicyModule()
    model = FeedForwardModel(
          init=lambda rng: module.init(rng, dummy_obs), apply=module.apply)
    return model

  def value_model_fn():
    class ValueModule(linen.Module):
      """Q Module."""
      n_critics: int = 2

      @linen.compact
      def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray, mask: jnp.ndarray = None):
        data = jnp.concatenate([obs, actions], axis=-1)
        # (B, L, O+1)
        res = []
        attns = []
        for _ in range(self.n_critics):
          output, attn_weights = make_transformer(
            output_size=policy_params_size,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            transformer_norm=transformer_norm,
            condition_decoder=condition_decoder,
            positional_encoding=positional_encoding,
          )(data, mask) # (B, 1, P) P: number of distribution parameters
          output = output.reshape((data.shape[0], -1, 1))
          res.append(output)
          attns.append(attn_weights)
        return jnp.concatenate(res, axis=-1), attns

    module = ValueModule()
    model = FeedForwardModel(
          init=lambda rng: module.init(rng, dummy_obs, dummy_action),
          apply=module.apply)
    return model

  return policy_model_fn(), value_model_fn()
