import functools
import itertools
from procedural_envs.misc.observers import SimObserver
from procedural_envs.misc.random_init_functions import annulus_xy_sampler
from procedural_envs.misc.reward_functions import nearest_distance_reward


def load_desc(
    num_body: int = 4,
    radius: float = 0.1,
    r_min: float = 5.0,
    r_max: float = 6.0,
    agent: str = 'centipede',
    broken_ids: tuple = (4, 0),
    size_scales: list = [],
    mass_values: list = []):
  # define random_init_fn for 'ball' component
  random_init_fn = functools.partial(
    annulus_xy_sampler, r_min=r_min, r_max=r_max)
  leg_indices = list(itertools.product([4,5], list(range(num_body))))
  component_params = dict(n=num_body)
  if agent == 'broken_centipede':
    for idx in leg_indices:
      if idx == broken_ids[0:2]:
        leg_indices.remove(idx)
    component_params['broken_ids'] = broken_ids
  elif agent == 'broken_centipede2':
    for idx in leg_indices:
      for j in range(len(broken_ids)):
        if idx == broken_ids[j][0:2]:
          leg_indices.remove(idx)
    component_params['broken_ids'] = broken_ids
  elif agent == 'size_rand_centipede':
    component_params['size_scales'] = size_scales
  elif agent == 'mass_rand_centipede':
    component_params['mass_values'] = mass_values
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
                          SimObserver(comp_name='agent1', sdname=f'$ Body {i}_{j}', indices=(0, 1)) for i, j in leg_indices],
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
        ['cap1___Ball']+[f'agent1___$ Body {i}_{j}' for i, j in leg_indices],
        [],
        [],
        ]
      )

ENV_DESCS = dict()

# add environments
for i in range(2, 8, 1):
  ENV_DESCS[f'centipede_reach_{i}'] = functools.partial(load_desc, num_body=i)

# missing
for i in range(2, 8, 1):
  for j in range(i):
      for k in (4, 5):  # left or right
        ENV_DESCS[f'centipede_reach_{i}_b_{k}_{j}'] = functools.partial(load_desc, agent='broken_centipede', num_body=i, broken_ids=(k, j))
        # missing entire leg
        ENV_DESCS[f'centipede_reach_{i}_b_{k}_{j}_all'] = functools.partial(load_desc, agent='broken_centipede', num_body=i, broken_ids=(k, j, -1))

# size/mass
for i in range(2, 8, 1):
  for size_scales in [[0.9, 1.0, 1.1], [0.9, 1.0, 1.0], [1.0, 1.1, 1.1]]:
    ENV_DESCS['centipede_reach_{}_size_{}'.format(i, "_".join([str(float(v)) for v in size_scales]))] = functools.partial(load_desc, agent='size_rand_centipede', num_body=i, size_scales=size_scales)
  for mass_values in [[0.5, 1.0, 3.0], [0.5, 1.0, 1.0], [1.0, 3.0, 3.0]]:
    ENV_DESCS['centipede_reach_{}_mass_{}'.format(i, "_".join([str(float(v)) for v in mass_values]))] = functools.partial(load_desc, agent='mass_rand_centipede', num_body=i, mass_values=mass_values)
