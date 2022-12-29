num_goal = 1
num_limb_dict = {i: i+num_goal for i in range(2, 8, 1)}
action_size_dict = {i: i - 1 for i in range(2, 8, 1)}

mtg_v2_base_m_obs_size = 26 + 6

DATASET_CONFIG = {}
