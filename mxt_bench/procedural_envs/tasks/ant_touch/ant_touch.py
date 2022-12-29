import functools
from procedural_envs.misc.observers import SimObserver
from procedural_envs.misc.random_init_functions import annulus_xy_sampler
from procedural_envs.misc.reward_functions import distance_reward


def load_desc(
    num_legs: int = 4,
    radius: float = 0.25,
    mass: float = 100.,
    r_min: float = 2.0,
    r_max: float = 5.0,
    agent: str = 'ant',
    broken_id: int = 0,
    mass_values: list = [],
    size_scales: list = []):
  # define random_init_fn for 'ball' component
  random_init_fn = functools.partial(
    annulus_xy_sampler, r_min=r_min, r_max=r_max, init_z=radius)
  component_params = dict(num_legs=num_legs)
  if agent == 'broken_ant':
    component_params['broken_id'] = broken_id
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
                      reward_type=distance_reward,
                      obs1=SimObserver(comp_name='agent1', sdname='torso', indices=(0, 1)),
                      obs2=SimObserver(comp_name='cap1', sdname='Ball', indices=(0, 1)),
                      min_dist=radius*2,
                      done_bonus=0.0)
              ),
          ),
          cap1=dict(
              component='ball',
              component_params=dict(
                  radius=radius,
                  mass=mass,
                  name="Ball"
                  ),
              pos=(0, 0, radius),
              random_init='pos',
              random_init_fn=random_init_fn,
          ),
        ),
      global_options=dict(dt=0.05, substeps=10),
      goal_based_task=True,
      task_edge=[
        ['cap1___Ball', 'agent1___torso'],
        [],
        [],
        ]
      )

ENV_DESCS = dict()

# add environments
for i in range(2, 7, 1):
  ENV_DESCS[f'ant_touch_{i}'] = functools.partial(load_desc, num_legs=i)

# missing
for i in range(3, 7, 1):
  for j in range(i):
    ENV_DESCS[f'ant_touch_{i}_b_{j}'] = functools.partial(load_desc, agent='broken_ant', num_legs=i, broken_id=j)

# size/mass
for i in range(2, 7, 1):
  for mass_values in [[0.5, 1.0, 3.0], [0.5, 1.0, 1.0], [1.0, 3.0, 3.0]]:
    ENV_DESCS['ant_touch_{}_mass_{}'.format(i, "_".join([str(float(v)) for v in mass_values]))] = functools.partial(load_desc, agent='mass_rand_ant', num_legs=i, mass_values=mass_values)
  for size_scales in [[0.9, 1.0, 1.1], [0.9, 1.0, 1.0], [1.0, 1.1, 1.1]]:
    ENV_DESCS['ant_touch_{}_size_{}'.format(i, "_".join([str(float(v)) for v in size_scales]))] = functools.partial(load_desc, agent='size_rand_ant', num_legs=i, size_scales=size_scales)
