import functools
from typing import Tuple

from procedural_envs.misc.observers import SimObserver
from procedural_envs.misc.random_init_functions import annulus_xy_sampler
from procedural_envs.misc.random_init_functions import circular_sector_xy_sampler
from procedural_envs.misc.reward_functions import distance_reward, moving_reward, fraction_reward


def load_desc(
    num_body: int = 4,
    radius: float = 1.5,
    mass: float = 1.,
    r_min_b: float = 2.5,
    r_max_b: float = 3.5,
    r_min_g: float = 6.0,
    r_max_g: float = 6.5,
    theta_min: float = -1/15,
    theta_max: float = 1/15,
    done_bonus: float = 25.,
    halfsize: Tuple[float, float, float] = (1., 1., 0.75),
    moving_to_target_scale: float = 4.0):
  # define random_init_fn for 'ball' component
  random_box_init_fn = functools.partial(
    annulus_xy_sampler, r_min=r_min_b, r_max=r_max_b, init_z=halfsize[2])
  random_goal_init_fn = functools.partial(
    circular_sector_xy_sampler,
    r_min=r_min_g, r_max=r_max_g, theta_min=theta_min, theta_max=theta_max)
  dt = 0.05
  return dict(
      components=dict(
          agent1=dict(
              component='worm',
              component_params=dict(n=num_body),
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
                      vel0=SimObserver(comp_name='agent1', sdname='torso_0', sdcomp='vel', indices=(0, 1, 2)),
                      pos0=SimObserver(comp_name='cap1', sdname='Box', indices=(0, 1, 2)),
                      pos1=SimObserver(comp_name='agent1', sdname='torso_0', indices=(0, 1, 2)),
                      scale=0.1*dt),
                  close_to_object=dict(
                      reward_type=fraction_reward,
                      obs1=SimObserver(comp_name='cap1', sdname='Box', indices=(0, 1)),
                      obs2=SimObserver(comp_name='agent1', sdname='torso_0', indices=(0, 1)),
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
for i in range(2, 8, 1):
  ENV_DESCS[f'worm_push_{i}'] = functools.partial(load_desc, num_body=i)
