from procedural_envs.tasks.observation_config import amorpheus
from procedural_envs.tasks.observation_config import base_m
from procedural_envs.tasks.observation_config import mtg_v1_base
from procedural_envs.tasks.observation_config import mtg_v2_base
from procedural_envs.tasks.observation_config import mtg_v1_base_m
from procedural_envs.tasks.observation_config import mtg_v2_base_m
from procedural_envs.tasks.observation_config import mtg_v1_base_id
from procedural_envs.tasks.observation_config import mtg_v2_base_id
from procedural_envs.tasks.observation_config import mtg_v1_base_rp
from procedural_envs.tasks.observation_config import mtg_v2_base_rp
from procedural_envs.tasks.observation_config import mtg_v1_base_rr
from procedural_envs.tasks.observation_config import mtg_v2_base_rr
from procedural_envs.tasks.observation_config import mtg_v1_base_rp_rr
from procedural_envs.tasks.observation_config import mtg_v2_base_rp_rr
from procedural_envs.tasks.observation_config import mtg_v1_base_jv_rp_rr
from procedural_envs.tasks.observation_config import mtg_v2_base_jv_rp_rr
from procedural_envs.tasks.observation_config import mtg_v1_base_jv_rp_rr_m
from procedural_envs.tasks.observation_config import mtg_v2_base_jv_rp_rr_m


RELATIONAL_GOAL_FEATURES = 2 * 3

# load pre-defined config dictionaries
obs_config_dict = {
  'amorpheus': amorpheus.OBSERVATION_CONFIG,
  'base_m': base_m.OBSERVATION_CONFIG,
  'mtg_v1_base': mtg_v1_base.OBSERVATION_CONFIG,
  'mtg_v1_base_id': mtg_v1_base_id.OBSERVATION_CONFIG,
  'mtg_v1_base_rr': mtg_v1_base_rr.OBSERVATION_CONFIG,
  'mtg_v1_base_rp': mtg_v1_base_rp.OBSERVATION_CONFIG,
  'mtg_v1_base_rp_rr': mtg_v1_base_rp_rr.OBSERVATION_CONFIG,
  'mtg_v1_base_jv_rp_rr': mtg_v1_base_jv_rp_rr.OBSERVATION_CONFIG,
  'mtg_v1_base_jv_rp_rr_m': mtg_v1_base_jv_rp_rr_m.OBSERVATION_CONFIG,
  'mtg_v1_base_m': mtg_v1_base_m.OBSERVATION_CONFIG,
  'mtg_v2_base': mtg_v2_base.OBSERVATION_CONFIG,
  'mtg_v2_base_id': mtg_v2_base_id.OBSERVATION_CONFIG,
  'mtg_v2_base_rr': mtg_v2_base_rr.OBSERVATION_CONFIG,
  'mtg_v2_base_rp': mtg_v2_base_rp.OBSERVATION_CONFIG,
  'mtg_v2_base_rp_rr': mtg_v2_base_rp_rr.OBSERVATION_CONFIG,
  'mtg_v2_base_jv_rp_rr': mtg_v2_base_jv_rp_rr.OBSERVATION_CONFIG,
  'mtg_v2_base_jv_rp_rr_m': mtg_v2_base_jv_rp_rr_m.OBSERVATION_CONFIG,
  'mtg_v2_base_m': mtg_v2_base_m.OBSERVATION_CONFIG,
}

obs_size_dict = {
  'amorpheus': amorpheus.OBSERVATION_SIZE,
  'base_m': base_m.OBSERVATION_SIZE,
  'mtg_v1_base_id': mtg_v1_base_id.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v1_base': mtg_v1_base.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v1_base_rr': mtg_v1_base_rr.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v1_base_rp': mtg_v1_base_rp.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v1_base_rp_rr': mtg_v1_base_rp_rr.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v1_base_jv_rp_rr': mtg_v1_base_jv_rp_rr.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v1_base_jv_rp_rr_m': mtg_v1_base_jv_rp_rr_m.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v1_base_m': mtg_v1_base_m.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v2_base_id': mtg_v2_base_id.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v2_base': mtg_v2_base.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v2_base_rr': mtg_v2_base_rr.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v2_base_rp': mtg_v2_base_rp.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v2_base_rp_rr': mtg_v2_base_rp_rr.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v2_base_jv_rp_rr': mtg_v2_base_jv_rp_rr.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v2_base_jv_rp_rr_m': mtg_v2_base_jv_rp_rr_m.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
  'mtg_v2_base_m': mtg_v2_base_m.OBSERVATION_SIZE + RELATIONAL_GOAL_FEATURES,
}
