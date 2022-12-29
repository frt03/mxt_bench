num_goal = 1
num_limb_dict = {i: 1+2*i+num_goal for i in range(2, 7, 1)}
action_size_dict = {i: 2*i for i in range(2, 7, 1)}

mtg_v2_base_m_obs_size = 26 + 6

# TODO: this is an example of format.
DATASET_CONFIG = {
  'ant_reach_2': {
    'qp_path': '../data/ant_reach_2_qp.pkl',
    'mtg_v2_base_m': {
      'dataset_path': '../data/ant_reach_2_mtg_v2_base_m.pkl',
      'observation_size': mtg_v2_base_m_obs_size*num_limb_dict[2], 'action_size': action_size_dict[2], 'num_limb': num_limb_dict[2]},
  },
  'ant_reach_3': {
    'qp_path': '../data/ant_reach_3_qp.pkl',
    'mtg_v2_base_m': {
      'dataset_path': '../data/ant_reach_3_mtg_v2_base_m.pkl',
      'observation_size': mtg_v2_base_m_obs_size*num_limb_dict[3], 'action_size': action_size_dict[3], 'num_limb': num_limb_dict[3]},
  },
  'ant_reach_4': {
    'qp_path': '../data/ant_reach_4_qp.pkl',
    'mtg_v2_base_m': {
      'dataset_path': '../data/ant_reach_4_mtg_v2_base_m.pkl',
      'observation_size': mtg_v2_base_m_obs_size*num_limb_dict[4], 'action_size': action_size_dict[4], 'num_limb': num_limb_dict[4]},
  },
  'ant_reach_5': {
    'qp_path': '../data/ant_reach_5_qp.pkl',
    'mtg_v2_base_m': {
      'dataset_path': '../data/ant_reach_5_mtg_v2_base_m.pkl',
      'observation_size': mtg_v2_base_m_obs_size*num_limb_dict[5], 'action_size': action_size_dict[5], 'num_limb': num_limb_dict[5]},
  },
  'ant_reach_6': {
    'qp_path': '../data/ant_reach_6_qp.pkl',
    'mtg_v2_base_m': {
      'dataset_path': '../data/ant_reach_6_mtg_v2_base_m.pkl',
      'observation_size': mtg_v2_base_m_obs_size*num_limb_dict[6], 'action_size': action_size_dict[6], 'num_limb': num_limb_dict[6]},
  },
}
