"""Composer for environments.

ComponentEnv composes a scene from descriptions of the form below:

   composer = Composer(
    components=dict(
        agent1=dict(component='ant', pos=(0, 1, 0)),
        agent2=dict(component='ant', pos=(0, -1, 0)),
    ),
    edges=dict(ant1__ant2=dict(collide_type='full'),),
   )
   env = ComposeEnv(composer=composer)

During loading an env through create(), it:
- loads an env_descs, a dictionary containing all args to Composer/ComposerEnv
  - pre-defined envs are defined in envs/
  - new envs can be registered through register_env() or register_lib()
  - example env_descs are in envs/ant_descs.py
  - example multi-agent RL envs (through agent_utils.py) are in envs/ma_descs.py
- creates components: loads and pieces together brax.Config()
    components defined in components/
    such as ant.py or ground.py
  - new components can be registered through register_component()
  - support multiple instances of the same component through comp_name's
  - each component requires: ROOT=root body, SYS_CONFIG=config in string form,
      TERM_FN=termination function of this component, COLLIDES=bodies that
      are allowed to collide, DEFAULT_OBSERVERS=a list of observers (
      see observers.py for references)
  - optionally, each component can specify a dictionary of reward functions
      as `reward_fns`. See reward_functions.py.
- creates edges: automatically create necessary edge information
    among 2+ components, such as collide_include's in brax.Config()
  - optionally edge information can be supplied,
      e.g. `collide_type`={'full', 'root', None} specifying full collisons,
      collision only between roots, or no collision between two components
  - optionally, each edge can specify a dictionary of reward functions
      as `reward_fns`. See reward_functions.py.
- sets reward as sum of all `reward_fns` defined in `components` and `edges`
- sets termination as any(termination_fn of each component)
- sets observation to concatenation of observations of each component defined
    by each component's `observers` argument
"""

import copy
import functools
import itertools
# import pprint
from collections import namedtuple, OrderedDict
from typing import Dict, Any, Callable, Tuple, Optional, Union

import brax
import jax
from brax import envs
from brax.envs import Env
from brax.envs import State
from brax.envs import Wrapper
from brax.envs import wrappers
from brax.experimental.composer import agent_utils
from brax.experimental.composer import composer_utils
from brax.experimental.composer import data_utils
from brax.training.types import PRNGKey
from procedural_envs import tasks
from procedural_envs.components import load_component
from procedural_envs.components import list_components
from procedural_envs.components import register_default_components
from procedural_envs.misc import component_editor
from procedural_envs.misc import observers
from procedural_envs.misc import reward_functions
from procedural_envs.misc import sim_utils
from procedural_envs.misc.env_wrapper import GoalEvalWrapper, CustomState
from jax import numpy as jnp

assert_env_params = tasks.assert_env_params
inspect_env = tasks.inspect_env
list_env = tasks.list_env
register_env = tasks.register_env
register_lib = tasks.register_lib

register_default_components()
# print('Registered default components')
# pprint.pprint(list_components())
# print('Registered default tasks')
tasks.register_default_libs()
# pprint.pprint(tasks.ENV_DESCS)

# metadata of comporser's env
MetaData = namedtuple('MetaData', [
    'components',
    'edges',
    'global_options',
    'config_str',
    'config_json',
    'extra_observers',
    'reward_features',
    'reward_fns',
    'agent_groups',
    'goal_based_task',
    'satisfy_all_cond',
    'task_edge',
])


class CustomComposer(object):
  """Compose a brax system."""

  def __init__(self,
               components: Dict[str, Dict[str, Any]],
               edges: Dict[str, Dict[str, Any]] = None,
               extra_observers: Tuple[observers.Observer] = (),
               add_ground: bool = True,
               agent_groups: Dict[str, Any] = None,
               global_options: Dict[str, Any] = None,
               goal_based_task: bool = False,
               satisfy_all_cond: bool = False,
               task_edge: list = [[], [], []]):
    components = copy.deepcopy(components)
    edges = copy.deepcopy(edges or {})
    global_options = copy.deepcopy(global_options or {})
    extra_observers = copy.deepcopy(extra_observers)
    reward_features = []
    reward_fns = OrderedDict()
    agent_groups = agent_groups or {}

    # load components
    if add_ground:
      components['ground'] = dict(component='ground')
    components = {
        name: load_component(**value) for name, value in components.items()
    }
    component_keys = sorted(components)
    components = OrderedDict([(k, components[k]) for k in component_keys])

    # set global parameters (e.g. friction, gravity, dt, substeps)
    global_options = dict(
        json=component_editor.json_global_options(**global_options))
    global_options['message_str'] = component_editor.json2message_str(
        global_options['json'])

    # edit component_lib
    components_ = OrderedDict()
    for k in component_keys:
      v, new_v = components[k], {}
      # [start config editing] convert to json format for easy editing
      new_v['json'] = component_editor.message_str2json(v.pop('message_str'))

      # add comp_name's
      comp_name = k
      rename_fn = functools.partial(
          component_editor.json_concat_name, comp_name=comp_name)
      new_v['json'] = rename_fn(new_v['json'])
      new_v['collides'] = rename_fn(v.pop('collides'), force_add=True)
      new_v['root'] = rename_fn(v.pop('root'), force_add=True)

      # add frozen bodies option
      if v.pop('frozen', False):
        for b in new_v['json'].get('bodies', []):
          b['frozen'] = {'all': True}
        new_v['frozen'] = True

      # cache config properties
      new_v['bodies'] = [b['name'] for b in new_v['json'].get('bodies', [])]
      new_v['joints'] = [b['name'] for b in new_v['json'].get('joints', [])]
      new_v['actuators'] = [
          b['name'] for b in new_v['json'].get('actuators', [])
      ]
      new_v['comp_name'] = comp_name

      # [end config editing] convert back to str
      new_v['message_str'] = component_editor.json2message_str(new_v['json'])

      # set transform or not
      if 'pos' in v or 'quat' in v:
        new_v['transform'] = True
        new_v['pos'] = jnp.array(v.pop('pos', [0, 0, 0]), dtype=jnp.float32)
        new_v['quat_origin'] = jnp.array(
            v.pop('quat_origin', [0, 0, 0]), dtype=jnp.float32)
        new_v['quat'] = jnp.array(
            v.pop('quat', [1., 0., 0., 0.]), dtype=jnp.float32)
      else:
        new_v['transform'] = False

      # set randomized initial pos/rot function for goals
      new_v['random_init'] = v.pop('random_init', None)
      new_v['random_init_fn'] = v.pop('random_init_fn', None)
      assert new_v['random_init'] in ('pos', 'quat', None), new_v['random_init']
      new_v['reference'] = v.pop('reference', None)

      # add reward functions
      component_reward_fns = v.pop('reward_fns', {})
      for name, reward_kwargs in sorted(component_reward_fns.items()):
        name = component_editor.concat_name(name, comp_name)
        assert name not in reward_fns, f'duplicate reward_fns {name}'
        reward_fn, unwrapped_reward_fn = reward_functions.get_reward_fns(
            new_v, **reward_kwargs)
        reward_fns[name] = reward_fn
        reward_features += reward_functions.get_observers_from_reward_fns(
            unwrapped_reward_fn)

      # add extra observers
      component_observers = v.pop('extra_observers', ())
      for observer_kwargs in component_observers:
        extra_observers += (observers.get_component_observers(
            new_v, **observer_kwargs),)

      assert all(kk in ('term_fn', 'observers', 'collide')
                 for kk in v), f'unused kwargs in components[{k}]: {v}'
      new_v.update(v)
      components_[k] = new_v
    del components

    edges_ = {}
    for k1, k2 in itertools.combinations(component_keys, 2):
      if k1 == k2:
        continue
      k1, k2 = sorted([k1, k2])  # ensure the name is always sorted in order
      edge_name = component_editor.concat_comps(k1, k2)
      v, new_v = edges.pop(edge_name, {}), {}
      v1, v2 = [components_[k] for k in [k1, k2]]

      # add reward functions
      edge_reward_fns = v.pop('reward_fns', {})
      for name, reward_kwargs in sorted(edge_reward_fns.items()):
        name = component_editor.concat_name(name, edge_name)
        assert name not in reward_fns, f'duplicate reward_fns {name}'
        reward_fn, unwrapped_reward_fn = reward_functions.get_reward_fns(
            v1, v2, **reward_kwargs)
        reward_fns[name] = reward_fn
        reward_features += reward_functions.get_observers_from_reward_fns(
            unwrapped_reward_fn)

      # add observers
      edge_observers = v.pop('extra_observers', ())
      for observer_kwargs in edge_observers:
        extra_observers += (observers.get_edge_observers(
            v1, v2, **observer_kwargs),)

      # [start config editing]
      collide_type = v.pop('collide_type', 'full')
      v_json = {}
      # add colliders
      if not all([vv.get('collide', True) for vv in [v1, v2]]):
        pass
      elif collide_type == 'full':
        v_json.update(
            component_editor.json_collides(v1['collides'], v2['collides']))
      elif collide_type == 'root':
        v_json.update(
            component_editor.json_collides([v1['root']], [v2['root']]))
      else:
        assert not collide_type, collide_type
      # [end config editing]
      if v_json:
        # convert back to str
        new_v['message_str'] = component_editor.json2message_str(v_json)
      else:
        new_v['message_str'] = ''
      new_v['json'] = v_json
      assert not v, f'unused kwargs in edges[{edge_name}]: {v}'
      edges_[edge_name] = new_v
    assert not edges, f'unused edges: {edges}'
    edge_keys = sorted(edges_.keys())
    edges_ = OrderedDict([(k, edges_[k]) for k in edge_keys])

    # merge all message strs
    message_str = ''
    for _, v in sorted(components_.items()):
      message_str += v.get('message_str', '')
    for _, v in sorted(edges_.items()):
      message_str += v.get('message_str', '')
    message_str += global_options.get('message_str', '')
    config_str = message_str
    config_json = component_editor.message_str2json(message_str)
    metadata = MetaData(
        components=components_,
        edges=edges_,
        global_options=global_options,
        config_str=config_str,
        config_json=config_json,
        extra_observers=extra_observers,
        reward_features=reward_features,
        reward_fns=reward_fns,
        agent_groups=agent_groups,
        goal_based_task=goal_based_task,
        satisfy_all_cond=satisfy_all_cond,
        task_edge=task_edge,
    )
    config = component_editor.message_str2message(message_str)
    self.config, self.metadata = config, metadata

  def reset_fn(self, sys, qp: brax.QP, rng: jnp.ndarray):
    """Reset state."""
    # apply translations and rotations
    for _, v in sorted(self.metadata.components.items()):
      rng, _ = jax.random.split(rng)
      if v['transform']:
        _, _, mask = sim_utils.names2indices(sys.config, v['bodies'], 'body')
        qp = sim_utils.transform_qp(qp, mask[..., None], v['quat'],
                                    v['quat_origin'], v['pos'])
      if v['random_init']:
        _, _, mask = sim_utils.names2indices(sys.config, v['bodies'], 'body')
        random_init_fn = functools.partial(v['random_init_fn'], rng=rng)
        if v['reference']:
          # get reference direction in xy space
          indices, _, _ = sim_utils.names2indices(sys.config, v['reference'], 'body')
          ref_theta = jnp.arctan2(qp.pos[indices[0]][1], qp.pos[indices[0]][0])
          random_init_fn = functools.partial(random_init_fn, ref_theta=ref_theta)
        qp = sim_utils.sample_init_qp(qp, v['random_init'],
                                      random_init_fn, mask[..., None])

    return qp

  def term_fn(self, done: jnp.ndarray, sys, qp: brax.QP, info: brax.Info):
    """Termination."""
    for _, v in self.metadata.components.items():
      term_fn = v['term_fn']
      if term_fn:
        done = term_fn(done, sys, qp, info, v)
    return done

  def obs_fn(self, sys, qp: brax.QP, info: brax.Info):
    """Return observation as OrderedDict."""
    cached_obs_dict = {}
    obs_dict = OrderedDict()
    reward_features = OrderedDict()
    for observer in self.metadata.extra_observers:
      obs_dict_ = observers.get_obs_dict(sys, qp, info, observer,
                                         cached_obs_dict, None)
      obs_dict = OrderedDict(list(obs_dict.items()) + list(obs_dict_.items()))
    for observer in self.metadata.reward_features:
      obs_dict_ = observers.get_obs_dict(sys, qp, info, observer,
                                         cached_obs_dict, None)
      reward_features = OrderedDict(
          list(reward_features.items()) + list(obs_dict_.items()))
    return obs_dict, reward_features


class CustomComponentEnv(Env):
  """Make a brax Env from config/metadata for training and inference."""

  def __init__(self,
               composer: CustomComposer,
               env_desc: Dict[str, Any],
               observer: Any,
               observer2: Any = None):
    self.observer_shapes = None
    self.observer_shapes2 = None
    self.composer = composer
    self.env_desc = env_desc
    self.observer = observer
    self.observer2 = observer2 if observer2 is not None else observer
    self.metadata = composer.metadata
    super().__init__(config=self.composer.metadata.config_str)
    self.action_shapes = get_action_shapes(self.sys)
    self.reward_shape = (len(
        self.metadata.agent_groups),) if self.metadata.agent_groups else ()
    self.observer.initialize(self.sys, self.metadata.components['agent1']['root'], env_desc['task_edge'])
    self.observer2.initialize(self.sys, self.metadata.components['agent1']['root'], env_desc['task_edge'])
    self.num_node = len(sim_utils.get_names(self.sys.config, 'body')[:-1]) + self.observer.num_duplicate_node
    self.num_node2 = len(sim_utils.get_names(self.sys.config, 'body')[:-1]) + self.observer.num_duplicate_node
    if self.observer.direct_goal_edge:
      self.num_node -= sum([1 for v in env_desc['task_edge'] if len(v) > 0])
    if self.observer2.direct_goal_edge:
      self.num_node2 -= sum([1 for v in env_desc['task_edge'] if len(v) > 0])
    assert self.observation_size  # ensure self.observer_shapes is set
    self.group_action_shapes = agent_utils.set_agent_groups(
        self.metadata, self.action_shapes, self.observer_shapes)
    self.group_action_shapes2 = agent_utils.set_agent_groups(
        self.metadata, self.action_shapes, self.observer_shapes2)

  @property
  def is_multiagent(self):
    return bool(self.metadata.agent_groups)

  @property
  def agent_names(self):
    if not self.is_multiagent:
      return ()
    else:
      return tuple(sorted(self.metadata.agent_groups))

  def reset(self, rng: jnp.ndarray) -> CustomState:
    """Resets the environment to an initial state."""
    qpos = self.sys.default_angle()
    qvel = jnp.zeros((self.sys.num_joint_dof,))
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    qp = self.composer.reset_fn(self.sys, qp, rng)
    info = self.sys.info(qp)
    obs_dict, _ = self._get_obs(qp, info)
    obs = data_utils.concat_array(obs_dict, self.observer_shapes)
    # get obs from observer2
    obs_dict2, _ = self._get_obs2(qp, info)
    obs2 = data_utils.concat_array(obs_dict2, self.observer_shapes2)
    reward, done, score = jnp.zeros((3,) + self.reward_shape)
    reward_names = tuple(self.composer.metadata.reward_fns)
    if self.is_multiagent:  # multi-agent
      done = jnp.any(done, axis=-1)  # ensure done is a scalar
      reward_names += self.agent_names
    metrics = {'distance': jnp.zeros(()), 'score': score}
    for k in reward_names:
      metrics[f'reward_{k}'] = jnp.zeros(())
      metrics[f'score_{k}'] = jnp.zeros(())
    return CustomState(
        qp=qp,
        obs=obs,
        obs2=obs2,
        reward=reward,
        done=done.astype(jnp.float32),
        metrics=metrics)

  def replay_from_qp_a(self, qp: brax.QP, action: jnp.ndarray) -> jnp.ndarray:
    next_qp, next_info = self.sys.step(qp, action)
    obs_dict2, _ = self._get_obs2(next_qp, next_info)
    obs2 = data_utils.concat_array(obs_dict2, self.observer_shapes2)
    return obs2

  def step(self,
           state: State,
           action: jnp.ndarray,
           normalizer_params: Dict[str, jnp.ndarray] = None,
           extra_params: Dict[str, Dict[str, jnp.ndarray]] = None) -> CustomState:
    """Run one timestep of the environment's dynamics."""
    del normalizer_params, extra_params
    qp, info = self.sys.step(state.qp, action)
    obs_dict, reward_features = self._get_obs(qp, info)
    obs = data_utils.concat_array(obs_dict, self.observer_shapes)
    # get obs from observer2
    obs_dict2, _ = self._get_obs2(qp, info)
    obs2 = data_utils.concat_array(obs_dict2, self.observer_shapes2)
    reward_tuple_dict = OrderedDict([
        (k, fn(action, reward_features))
        for k, fn in self.composer.metadata.reward_fns.items()
    ])
    for k, v in reward_tuple_dict.items():
      state.metrics[f'reward_{k}'] = v[0]
      state.metrics[f'score_{k}'] = v[1]
      if 'agent1___distance' in k:
        state.metrics['distance'] = jnp.abs(v[1])
    if self.is_multiagent:  # multi-agent
      reward, score, done = agent_utils.process_agent_rewards(
          self.metadata, reward_tuple_dict)
      for k, v in zip(self.agent_names, reward):
        state.metrics[f'reward_{k}'] = v[0]
        state.metrics[f'score_{k}'] = v[1]
    else:
      reward, done, score = jnp.zeros((3,))
      if self.metadata.satisfy_all_cond:
        done = jnp.ones(())
      for r, s, d in reward_tuple_dict.values():
        reward += r
        score += s
        if self.metadata.satisfy_all_cond:
          done = jnp.logical_and(done, d)
        else:
          done = jnp.logical_or(done, d)
    done = self.composer.term_fn(done, self.sys, qp, info)
    state.metrics['score'] = score
    return state.replace(
        qp=qp, obs=obs, obs2=obs2, reward=reward, done=done.astype(jnp.float32))

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jnp.ndarray:
    """Observe for policy."""
    obs_dict = observers.get_obs_dict(
        self.sys, qp, info, self.observer, {}, self.metadata.components['agent1'])
    obs_dict2, reward_features = self.composer.obs_fn(self.sys, qp, info)
    obs_dict = OrderedDict(list(obs_dict.items()) + list(obs_dict2.items()))
    if self.observer_shapes is None:
      self.observer_shapes = data_utils.get_array_shapes(
          obs_dict, batch_shape=())
    return obs_dict, reward_features

  def _get_obs2(self, qp: brax.QP, info: brax.Info) -> jnp.ndarray:
    """Observe for policy."""
    obs_dict = observers.get_obs_dict(
        self.sys, qp, info, self.observer2, {}, self.metadata.components['agent1'])
    obs_dict2, reward_features = self.composer.obs_fn(self.sys, qp, info)
    obs_dict = OrderedDict(list(obs_dict.items()) + list(obs_dict2.items()))
    if self.observer_shapes2 is None:
      self.observer_shapes2 = data_utils.get_array_shapes(
          obs_dict, batch_shape=())
    return obs_dict, reward_features

  @property
  def observation_size2(self) -> int:
    """The size of the observation vector returned in step and reset."""
    rng = jax.random.PRNGKey(0)
    reset_obs = self.reset(rng).obs2
    return reset_obs.shape[-1]


def get_action_shapes(sys):
  """Get action shapes."""
  names = sim_utils.get_names(sys.config, 'actuator')
  action_shapes = sim_utils.names2indices(
      sys.config, names=names, datatype='actuator')[1]
  action_shapes = OrderedDict([
      (k, dict(start=v[0], end=v[-1] + 1, size=len(v), shape=(len(v),)))
      for k, v in action_shapes.items()
  ])
  return action_shapes


def unwrap(env: Env):
  """Unwrap all Env wrappers."""
  while isinstance(env, Wrapper):  # unwrap wrappers
    env = env.unwrapped
  return env


def is_multiagent(env: Env):
  """If it supports multiagent RL."""
  env = unwrap(env)
  return env.is_multiagent if hasattr(env, 'is_multiagent') else False


def get_obs_dict_shape(env: Env):
  """Get observation (dictionary) shape."""
  env = unwrap(env)
  if isinstance(env, CustomComponentEnv):
    assert env.observation_size  # ensure env.observer_shapes is set
    return env.observer_shapes
  else:
    return (env.observation_size,)


def create(env_name: str = None,
           env_desc: Union[Dict[str, Any], Callable[..., Dict[str,
                                                              Any]]] = None,
           desc_edits: Dict[str, Any] = None,
           episode_length: int = 1000,
           action_repeat: int = 1,
           auto_reset: bool = True,
           batch_size: Optional[int] = None,
           eval_metrics: bool = True,
           observer: Any = None,
           observer2: Any = None,
           **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  assert env_name or env_desc, 'env_name or env_desc must be supplied'
  env_desc = env_desc or {}
  desc_edits = desc_edits or {}
  if env_name in tasks.ENV_DESCS:
    desc = tasks.ENV_DESCS[env_name]
    if callable(desc):
      desc = desc(**kwargs)
    else:
      assert not kwargs, f'unused kwargs: {kwargs}'
    env_desc = dict(**env_desc, **desc)
    env_desc = composer_utils.edit_desc(env_desc, desc_edits)
    composer = CustomComposer(**env_desc)
    env = CustomComponentEnv(composer=composer,
                             env_desc=env_desc,
                             observer=observer,
                             observer2=observer2)
  elif env_desc:
    if callable(env_desc):
      env_desc = env_desc(**kwargs)
    else:
      assert not kwargs, f'unused kwargs: {kwargs}'
    env_desc = composer_utils.edit_desc(env_desc, desc_edits)
    composer = CustomComposer(**env_desc)
    env = CustomComponentEnv(composer=composer,
                             env_desc=env_desc,
                             observer=observer,
                             observer2=observer2)
  else:
    env = envs.create(
        env_name,
        episode_length=episode_length,
        action_repeat=action_repeat,
        auto_reset=auto_reset,
        batch_size=batch_size,
        **kwargs)
    return env  # type: ignore

  if episode_length is not None:
    env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
  if batch_size:
    env = wrappers.VectorWrapper(env, batch_size)
  if auto_reset:
    env = wrappers.AutoResetWrapper(env)
  if eval_metrics and composer.metadata.goal_based_task:
    env = GoalEvalWrapper(env)
  elif eval_metrics:
    env = wrappers.EvalWrapper(env)

  return env  # type: ignore


def create_fn(env_name: str = None,
              env_desc: Union[Dict[str, Any], Callable[..., Dict[str,
                                                                 Any]]] = None,
              observer: Any = None,
              observer2: Any = None,
              desc_edits: Dict[str, Any] = None,
              episode_length: int = 1000,
              action_repeat: int = 1,
              auto_reset: bool = True,
              batch_size: Optional[int] = None,
              **kwargs) -> Callable[..., Env]:
  """Returns a function that when called, creates an Env."""
  return functools.partial(
      create,
      env_name=env_name,
      env_desc=env_desc,
      observer=observer,
      observer2=observer2,
      desc_edits=desc_edits,
      episode_length=episode_length,
      action_repeat=action_repeat,
      auto_reset=auto_reset,
      batch_size=batch_size,
      **kwargs)
