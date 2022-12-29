import importlib


unimal_id = [
  '5506-2-16-01-10-58-23', '5506-12-12-01-11-30-00', '5506-12-14-01-15-22-01', '5506-9-12-01-10-32-52', '5506-9-7-01-13-40-02', '5506-4-14-01-14-32-47',
  '5506-12-12-01-15-33-01', '5506-8-16-01-13-07-43',  '5506-3-10-01-14-19-06', '5506-13-5-02-21-35-41', '5506-14-2-02-15-14-46', '5506-13-4-02-21-40-07',
  '5506-10-12-02-12-35-19', '5506-15-16-02-22-21-06', '5506-13-10-17-12-25-45', '5506-12-6-17-12-20-06', '5506-3-15-17-12-18-03', '5506-1-13-17-12-03-16',
  '5506-1-12-17-11-10-12', '5506-2-17-17-10-16-02', '5506-10-14-17-10-38-34', '5506-7-6-17-12-20-01', '5506-12-11-17-05-56-16', '5506-10-0-01-15-43-53',
  '5506-10-3-01-15-22-34', '5506-0-7-01-15-34-13', '5506-6-2-01-09-16-44', '5506-11-2-01-14-11-40', '5506-15-11-01-10-04-14', '5506-8-11-01-15-28-53',
  '5506-15-16-01-14-17-18', '5506-3-15-01-14-36-50', '5506-14-15-01-15-20-33', '5506-8-12-01-13-32-46', '5506-10-13-01-15-03-41', '5506-6-11-01-14-16-09',
  '5506-9-9-01-13-15-48', '5506-6-3-01-15-20-20', '5506-4-12-01-15-10-52', '5506-4-3-01-09-35-18', '5506-12-8-01-14-50-41', '5506-0-5-01-12-45-36',
  '5506-2-0-01-11-27-44', '5506-6-5-01-14-20-42', '5506-14-12-01-12-02-42', '5506-0-2-01-15-36-43', '5506-8-6-01-15-22-56', '5506-14-11-01-13-58-37',
  '5506-5-12-01-15-05-55', '5506-14-5-01-15-59-52', '5506-15-11-01-12-54-35', '5506-13-17-01-16-09-18', '5506-9-2-01-14-19-00', '5506-9-3-01-14-23-39',
  '5506-5-3-02-18-52-53', '5506-1-5-02-19-23-33', '5506-8-16-02-14-47-12', '5506-1-2-02-20-28-11', '5506-13-3-02-21-34-38', '5506-8-5-02-21-39-20',
  '5506-5-16-02-21-15-42', '5506-11-4-17-12-33-10', '5506-1-15-17-07-32-47', '5506-6-8-17-09-59-06', '5506-11-6-17-12-43-05', '5506-2-9-17-11-10-42',
  '5506-15-11-17-12-12-28', '5506-0-13-17-12-26-41', '5506-4-16-17-05-46-47', '5506-10-3-17-12-09-26', '5506-8-17-17-09-38-29', '5506-12-6-17-08-36-18'
]


# TODO: these are examples of format. please register your own datasets.
TASK_CONFIG = {
  'ant_reach': {
    f'ant_reach_{i}': importlib.import_module('procedural_envs.tasks.ant_reach.dataset_config').DATASET_CONFIG[f'ant_reach_{i}'] for i in range(2, 7, 1)},
  'centipede_touch': {
    f'centipede_touch_{i}': importlib.import_module('procedural_envs.tasks.centipede_touch.dataset_config').DATASET_CONFIG[f'centipede_touch_{i}'] for i in range(3, 8, 1)},
}
TASK_CONFIG['example'] = {**TASK_CONFIG['ant_reach'], **TASK_CONFIG['centipede_touch']}


ZERO_SHOT_TASK_CONFIG = {
  'ant_reach_zs_5': {
    'all_envs': TASK_CONFIG['ant_reach'],
    'train_envs': {f'ant_reach_{i}': importlib.import_module('procedural_envs.tasks.ant_reach.dataset_config').DATASET_CONFIG[f'ant_reach_{i}'] for i in [2, 3, 4, 6]},
    'test_envs': {'ant_reach_5': importlib.import_module('procedural_envs.tasks.ant_reach.dataset_config').DATASET_CONFIG[f'ant_reach_5']},
  },
  'centipede_touch_zs_4': {
    'all_envs': TASK_CONFIG['ant_reach'],
    'train_envs': {f'centipede_touch_{i}': importlib.import_module('procedural_envs.tasks.centipede_touch.dataset_config').DATASET_CONFIG[f'centipede_touch_{i}'] for i in [3, 5, 6, 7]},
    'test_envs': {'centipede_touch_4': importlib.import_module('procedural_envs.tasks.centipede_touch.dataset_config').DATASET_CONFIG[f'centipede_touch_4']},
  },
}

ZERO_SHOT_TASK_CONFIG['example'] = {
  'all_envs': {
    **ZERO_SHOT_TASK_CONFIG['ant_reach_zs_5']['all_envs'],
    **ZERO_SHOT_TASK_CONFIG['centipede_touch_zs_4']['all_envs'],
  },
  'train_envs': {
    **ZERO_SHOT_TASK_CONFIG['ant_reach_zs_5']['train_envs'],
    **ZERO_SHOT_TASK_CONFIG['centipede_touch_zs_4']['train_envs'],
  },
  'test_envs': {
    **ZERO_SHOT_TASK_CONFIG['ant_reach_zs_5']['test_envs'],
    **ZERO_SHOT_TASK_CONFIG['centipede_touch_zs_4']['test_envs'],
  },
}

ZERO_SHOT_TASK_CONFIG['fs_example'] = {
  'all_envs': {
    **ZERO_SHOT_TASK_CONFIG['ant_reach_zs_5']['all_envs'],
    **ZERO_SHOT_TASK_CONFIG['centipede_touch_zs_4']['all_envs'],
  },
  'train_envs': {
    **ZERO_SHOT_TASK_CONFIG['ant_reach_zs_5']['train_envs'],
    **ZERO_SHOT_TASK_CONFIG['centipede_touch_zs_4']['train_envs'],
  },
  'test_envs': {
    **ZERO_SHOT_TASK_CONFIG['ant_reach_zs_5']['test_envs'],
    **ZERO_SHOT_TASK_CONFIG['centipede_touch_zs_4']['test_envs'],
  },
}
