num_goal = 2
num_limb_dict = {i: 5*i+num_goal for i in range(2, 8, 1)}
action_size_dict = {i: 5*i - 1 for i in range(2, 8, 1)}

mtg_v2_base_m_obs_size = 26 + 6

DATASET_CONFIG = {}
