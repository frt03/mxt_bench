import functools
from procedural_envs.misc.observers import SimObserver
from procedural_envs.misc.random_init_functions import annulus_xy_sampler
from procedural_envs.misc.reward_functions import nearest_distance_reward


def load_desc(
    num_legs: int = 4,
    radius: float = 0.1,
    r_min: float = 7.5,
    r_max: float = 10.0,
    agent: str = 'ant',
    broken_id: int = 0,
    mass_values: list = [],
    size_scales: list = []):
  # define random_init_fn for 'ball' component
  random_init_fn = functools.partial(
    annulus_xy_sampler, r_min=r_min, r_max=r_max)
  component_params = dict(num_legs=num_legs)
  leg_indices = [i for i in range(num_legs)]
  if agent == 'broken_ant':
    component_params['broken_id'] = broken_id
    if broken_id in leg_indices:
      leg_indices.remove(broken_id)
  elif agent == 'mass_rand_ant':
    component_params['mass_values'] = mass_values
  elif agent == 'size_rand_ant':
    component_params['size_scales'] = size_scales
  return dict(
      components=dict(
          agent1=dict(
              component=agent,
              component_params=component_params,
              pos=(0, 0, 0),
              reward_fns=dict(
                  distance=dict(
                      reward_type=nearest_distance_reward,
                      target=SimObserver(comp_name='cap1', sdname='Ball', indices=(0, 1)),
                      obs=[
                          SimObserver(comp_name='agent1', sdname=f'$ Body 4_{i}', indices=(0, 1)) for i in leg_indices],
                      min_dist=radius,
                      done_bonus=0.0)
              ),
          ),
          cap1=dict(
              component='ball',
              component_params=dict(
                  radius=radius,
                  frozen=True,
                  name="Ball"
                  ),
              pos=(0, 0, 0),
              random_init='pos',
              random_init_fn=random_init_fn,
          ),
        ),
      global_options=dict(dt=0.05, substeps=10),
      goal_based_task=True,
      task_edge=[
        ['cap1___Ball']+[f'agent1___$ Body 4_{i}' for i in leg_indices],
        [],
        [],
        ]
      )

ENV_DESCS = dict()

# add environments
for i in range(2, 7, 1):
  ENV_DESCS[f'ant_reach_{i}'] = functools.partial(load_desc, num_legs=i)
  ENV_DESCS[f'ant_reach_hard_{i}'] = functools.partial(load_desc, num_legs=i, r_min=10.5, r_max=11.5)

# missing
for i in range(3, 7, 1):
  for j in range(i):
    ENV_DESCS[f'ant_reach_{i}_b_{j}'] = functools.partial(load_desc, agent='broken_ant', num_legs=i, broken_id=j)
    ENV_DESCS[f'ant_reach_hard_{i}_b_{j}'] = functools.partial(load_desc, agent='broken_ant', num_legs=i, broken_id=j, r_min=10.5, r_max=11.5)

# size/mass randomization
for i in range(2, 7, 1):
  for mass_values in [[0.5, 1.0, 3.0], [0.5, 1.0, 1.0], [1.0, 3.0, 3.0]]:
    ENV_DESCS['ant_reach_{}_mass_{}'.format(i, "_".join([str(float(v)) for v in mass_values]))] = functools.partial(load_desc, agent='mass_rand_ant', num_legs=i, mass_values=mass_values)
    ENV_DESCS['ant_reach_hard_{}_mass_{}'.format(i, "_".join([str(float(v)) for v in mass_values]))] = functools.partial(load_desc, agent='mass_rand_ant', num_legs=i, mass_values=mass_values, r_min=10.5, r_max=11.5)
  for size_scales in [[0.9, 1.0, 1.1], [0.9, 1.0, 1.0], [1.0, 1.1, 1.1]]:
    ENV_DESCS['ant_reach_{}_size_{}'.format(i, "_".join([str(float(v)) for v in size_scales]))] = functools.partial(load_desc, agent='size_rand_ant', num_legs=i, size_scales=size_scales)
    ENV_DESCS['ant_reach_hard_{}_size_{}'.format(i, "_".join([str(float(v)) for v in size_scales]))] = functools.partial(load_desc, agent='size_rand_ant', num_legs=i, size_scales=size_scales, r_min=10.5, r_max=11.5)
  for size_scales in [[0.8, 1.0, 1.5],]:
    ENV_DESCS['ant_reach_hard_{}_size_{}'.format(i, "_".join([str(float(v)) for v in size_scales]))] = functools.partial(load_desc, agent='size_rand_ant', num_legs=i, size_scales=size_scales, r_min=10.5, r_max=11.5)
