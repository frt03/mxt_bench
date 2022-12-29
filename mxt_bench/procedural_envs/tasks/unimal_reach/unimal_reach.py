import functools
from procedural_envs.misc.observers import SimObserver
from procedural_envs.misc.random_init_functions import annulus_xy_sampler
from procedural_envs.misc.reward_functions import nearest_distance_reward
from procedural_envs.misc.unimal_utils import get_end_effectors, get_agent_names


def load_desc(
    radius: float = 0.1,
    r_min: float = 7.5,
    r_max: float = 10.0,
    agent: str = 'unimal',
    config_name: str=""):
  # define random_init_fn for 'ball' component
  random_init_fn = functools.partial(
    annulus_xy_sampler, r_min=r_min, r_max=r_max)
  component_params = dict(config_name=config_name)

  # detect end nodes for reach tasks
  end_effectors = get_end_effectors(config_name)

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
                          SimObserver(comp_name='agent1', sdname=e, indices=(0, 1)) for e in end_effectors],
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
      global_options=dict(dt=0.05, substeps=10, friction=0.6, baumgarte_erp=0.1),
      goal_based_task=True,
      task_edge=[
        ['cap1___Ball']+[e for e in end_effectors],
        [],
        [],
        ]
      )

ENV_DESCS = dict()

# add environments
agent_names = get_agent_names()
for config_name in agent_names:
  ENV_DESCS[f'unimal_reach_{config_name}'] = functools.partial(load_desc, config_name=config_name)
