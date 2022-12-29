from procedural_envs.misc.unimal_utils import get_all_bodies, get_config


def get_specs(config_name: str):

  _SYSTEM_CONFIG = get_config(config_name)
  collides = get_all_bodies(config_name)
  return dict(
      message_str=_SYSTEM_CONFIG,
      collides=collides,
      root='torso_0',
      term_fn=None)
