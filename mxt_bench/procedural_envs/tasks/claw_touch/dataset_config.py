num_goal = 1
num_limb_dict = {i: 1+4*i+num_goal for i in range(2, 7, 1)}
action_size_dict = {i: 4*i for i in range(2, 7, 1)}

mtg_v2_base_m_obs_size = 26 + 6

DATASET_CONFIG = {}
