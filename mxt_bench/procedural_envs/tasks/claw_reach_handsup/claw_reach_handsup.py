import functools
from procedural_envs.misc.observers import SimObserver
from procedural_envs.misc.random_init_functions import annulus_xy_sampler, uniform_z_sampler
from procedural_envs.misc.reward_functions import nearest_distance_reward, distance_reward


def load_desc(
    num_legs: int = 4,
    radius: float = 0.1,
    r_min: float = 1.25,
    r_max: float = 1.75,
    z_min: float = 0.55,
    z_max: float = 0.7,
    constant_z: bool = False,
    min_dist_z: float = 0.1,
    z_done_bonus: float = 0.0,
    agent: str = 'claw'):
  # define random_init_fn for 'ball' component
  random_target_init_fn = functools.partial(
    annulus_xy_sampler, r_min=r_min, r_max=r_max)
  random_z_target_init_fn = functools.partial(
    uniform_z_sampler, z_min=z_min, z_max=z_max, const=constant_z)

  height_index = 0
  leg_indices = []
  for i in range(num_legs):
    if i != height_index:
      leg_indices.append(i)

  component_params = dict(num_legs=num_legs)

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
                      done_bonus=0.0),
                  leg_height=dict(
                      reward_type=distance_reward,
                      obs1=SimObserver(comp_name='cap2', sdname='Z_Target', indices=(2,)),
                      obs2=SimObserver(comp_name='agent1', sdname=f'$ Body 4_{height_index}', indices=(2,)),
                      min_dist=min_dist_z,
                      done_bonus=z_done_bonus)
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
              random_init_fn=random_target_init_fn,
          ),
          cap2=dict(
              component='ball',
              component_params=dict(
                  radius=min_dist_z,
                  frozen=True,
                  name="Z_Target"
                  ),
              pos=(0, 0, 0),
              random_init='pos',
              random_init_fn=random_z_target_init_fn,
          ),
        ),
      global_options=dict(dt=0.05, substeps=10),
      goal_based_task=True,
      satisfy_all_cond=True,
      task_edge=[
        ['cap1___Ball']+[f'agent1___$ Body 4_{i}' for i in leg_indices],
        ['cap2___Z_Target', f'agent1___$ Body 4_{height_index}'],
        [],
        ]
      )

ENV_DESCS = dict()

# add environments
for i in range(2, 7, 1):
  ENV_DESCS[f'claw_reach_handsup_{i}'] = functools.partial(load_desc, num_legs=i)
  ENV_DESCS[f'claw_reach_hard_handsup_{i}'] = functools.partial(load_desc, num_legs=i, r_min=3.5, r_max=5.5, z_min=0.5, z_max=0.6)
