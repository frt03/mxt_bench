"""Architecture config for behavioral cloning"""
import functools

from models.transformer import make_transformers
from models.mlp import make_mlp_policy_network


ARCHITECTURE_CONFIG = {
  'transformer': functools.partial(
    make_transformers,
    policy_params_size=1,
    num_layers=3,
    d_model=128*2,
    num_heads=2,
    dim_feedforward=256*2,
    transformer_norm=True,
    condition_decoder=True),
  'transformer_pe': functools.partial(
    make_transformers,
    policy_params_size=1,
    num_layers=3,
    d_model=128*2,
    num_heads=2,
    dim_feedforward=256*2,
    transformer_norm=True,
    condition_decoder=True,
    positional_encoding=True),
  'mlp_1024_1024': functools.partial(
    make_mlp_policy_network,
    hidden_layer_sizes=(1024, 1024),
  ),
}
