import functools
from typing import Tuple

from procedural_envs.misc.observers import SimObserver
from procedural_envs.misc.random_init_functions import annulus_xy_sampler
from procedural_envs.misc.random_init_functions import circular_sector_xy_sampler
from procedural_envs.misc.reward_functions import distance_reward, moving_reward, fraction_reward


def load_desc(
    num_legs: int = 4,
    radius: float = 1.0,
    mass: float = 1.,
    r_min_b: float = 3.5,
    r_max_b: float = 4.5,
    r_min_g: float = 7.5,
    r_max_g: float = 8.5,
    theta_min: float = -1/15,
    theta_max: float = 1/15,
    done_bonus: float = 20.,
    halfsize: Tuple[float, float, float] = (1., 1., 0.75),
    moving_to_target_scale: float = 3.0,
    agent: str = 'ant',
    broken_id: int = 0,
    size_scales: list = [],
    mass_values: list = []):
  # define random_init_fn for 'ball' component
  random_box_init_fn = functools.partial(
    annulus_xy_sampler, r_min=r_min_b, r_max=r_max_b, init_z=halfsize[2])
  random_goal_init_fn = functools.partial(
    circular_sector_xy_sampler,
    r_min=r_min_g, r_max=r_max_g, theta_min=theta_min, theta_max=theta_max)
  dt = 0.05
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
                      obs1=SimObserver(comp_name='cap1', sdname='Box', indices=(0, 1)),
                      obs2=SimObserver(comp_name='cap2', sdname='Target', indices=(0, 1)),
                      min_dist=radius,
                      done_bonus=done_bonus,
                      scale=0.0,
                      zero_scale_score=True),
                  moving_to_object=dict(
                      reward_type=moving_reward,
                      vel0=SimObserver(comp_name='agent1', sdname='torso', sdcomp='vel', indices=(0, 1, 2)),
                      pos0=SimObserver(comp_name='cap1', sdname='Box', indices=(0, 1, 2)),
                      pos1=SimObserver(comp_name='agent1', sdname='torso', indices=(0, 1, 2)),
                      scale=0.1*dt),
                  close_to_object=dict(
                      reward_type=fraction_reward,
                      obs1=SimObserver(comp_name='cap1', sdname='Box', indices=(0, 1)),
                      obs2=SimObserver(comp_name='agent1', sdname='torso', indices=(0, 1)),
                      min_dist=-1.,
                      frac_epsilon=1e-6,
                      scale=0.1*dt),
                  moving_to_target=dict(
                      reward_type=moving_reward,
                      vel0=SimObserver(comp_name='cap1', sdname='Box', sdcomp='vel', indices=(0, 1, 2)),
                      pos0=SimObserver(comp_name='cap2', sdname='Target', indices=(0, 1, 2)),
                      pos1=SimObserver(comp_name='cap1', sdname='Box', indices=(0, 1, 2)),
                      scale=moving_to_target_scale*dt),
              ),
          ),
          cap1=dict(
              component='box',
              component_params=dict(
                  halfsize=halfsize,
                  mass=mass,
                  name="Box"
                  ),
              pos=(0, 0, halfsize[2]),
              random_init='pos',
              random_init_fn=random_box_init_fn,
          ),
          cap2=dict(
              component='ball',
              component_params=dict(
                  radius=radius,
                  frozen=True,
                  mass=mass,
                  name="Target"
                  ),
              pos=(0, 0, 0),
              random_init='pos',
              random_init_fn=random_goal_init_fn,
              reference='cap1___Box',
          ),
        ),
      global_options=dict(dt=dt, substeps=10),
      goal_based_task=True,
      task_edge=[
        ['cap2___Target', 'cap1___Box'],
        [],
        [],
        ]
      )

ENV_DESCS = dict()

# add environments
for i in range(2, 7, 1):
  ENV_DESCS[f'ant_push_{i}'] = functools.partial(load_desc, num_legs=i)

for i in range(3, 7, 1):
  for j in range(i):
    ENV_DESCS[f'ant_push_{i}_b_{j}'] = functools.partial(load_desc, agent='broken_ant', num_legs=i, broken_id=j)

for i in range(2, 7, 1):
  for mass_values in [[0.5, 1.0, 3.0], [0.5, 1.0, 1.0], [1.0, 3.0, 3.0]]:
    ENV_DESCS['ant_push_{}_mass_{}'.format(i, "_".join([str(float(v)) for v in mass_values]))] = functools.partial(load_desc, agent='mass_rand_ant', num_legs=i, mass_values=mass_values)
  for size_scales in [[0.9, 1.0, 1.1], [0.9, 1.0, 1.0], [1.0, 1.1, 1.1]]:
    ENV_DESCS['ant_push_{}_size_{}'.format(i, "_".join([str(float(v)) for v in size_scales]))] = functools.partial(load_desc, agent='size_rand_ant', num_legs=i, size_scales=size_scales)
