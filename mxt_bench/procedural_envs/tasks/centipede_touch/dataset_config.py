num_goal = 1
num_limb_dict = {i: 5*i+num_goal for i in range(2, 8, 1)}
action_size_dict = {i: 5*i - 1 for i in range(2, 8, 1)}

mtg_v2_base_m_obs_size = 26 + 6

# TODO: this is an example of format.
DATASET_CONFIG = {
  'centipede_touch_3': {
    'qp_path': '../data/centipede_touch_3_qp.pkl',
    'mtg_v2_base_m': {
      'dataset_path': '../data/centipede_touch_3_amorpheus.pkl',
      'observation_size': mtg_v2_base_m_obs_size*num_limb_dict[3], 'action_size': action_size_dict[3], 'num_limb': num_limb_dict[3]},
  },
  'centipede_touch_4': {
    'qp_path': '../data/centipede_touch_4_qp.pkl',
    'mtg_v2_base_m': {
      'dataset_path': '../data/centipede_touch_4_amorpheus.pkl',
      'observation_size': mtg_v2_base_m_obs_size*num_limb_dict[4], 'action_size': action_size_dict[4], 'num_limb': num_limb_dict[4]},
  },
  'centipede_touch_5': {
    'qp_path': '../data/centipede_touch_5_qp.pkl',
    'mtg_v2_base_m': {
      'dataset_path': '../data/centipede_touch_5_amorpheus.pkl',
      'observation_size': mtg_v2_base_m_obs_size*num_limb_dict[5], 'action_size': action_size_dict[5], 'num_limb': num_limb_dict[5]},
  },
  'centipede_touch_6': {
    'qp_path': '../data/centipede_touch_6_qp.pkl',
    'mtg_v2_base_m': {
      'dataset_path': '../data/centipede_touch_6_amorpheus.pkl',
      'observation_size': mtg_v2_base_m_obs_size*num_limb_dict[6], 'action_size': action_size_dict[6], 'num_limb': num_limb_dict[6]},
  },
  'centipede_touch_7': {
    'qp_path': '../data/centipede_touch_7_qp.pkl',
    'mtg_v2_base_m': {
      'dataset_path': '../data/centipede_touch_7_amorpheus.pkl',
      'observation_size': mtg_v2_base_m_obs_size*num_limb_dict[7], 'action_size': action_size_dict[7], 'num_limb': num_limb_dict[7]},
  },
}
