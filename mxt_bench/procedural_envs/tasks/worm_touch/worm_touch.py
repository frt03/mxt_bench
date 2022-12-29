import functools
from procedural_envs.misc.observers import SimObserver
from procedural_envs.misc.random_init_functions import annulus_xy_sampler
from procedural_envs.misc.reward_functions import distance_reward


def load_desc(
    num_body: int = 4,
    radius: float = 0.25,
    mass: float = 100.,
    r_min: float = 2.0,
    r_max: float = 5.0):
  # define random_init_fn for 'ball' component
  random_init_fn = functools.partial(
    annulus_xy_sampler, r_min=r_min, r_max=r_max, init_z=radius)
  return dict(
      components=dict(
          agent1=dict(
              component='worm',
              component_params=dict(n=num_body),
              pos=(0, 0, 0),
              reward_fns=dict(
                  distance=dict(
                      reward_type=distance_reward,
                      obs1=SimObserver(comp_name='agent1', sdname='torso_0', indices=(0, 1)),
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
        ['cap1___Ball', 'agent1___torso_0'],
        [],
        [],
        ]
      )

ENV_DESCS = dict()

# add environments
for i in range(2, 8, 1):
  ENV_DESCS[f'worm_touch_{i}'] = functools.partial(load_desc, num_body=i)
