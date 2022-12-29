"""Observation functions.

get_obs_dict() supports modular observation specs, with observer=
  an Observer(): get the specified observation
    e.g.1. SimObserver(sim_datatype='body', sim_datacomp='vel',
      sim_dataname='root')
    e.g.2. LambdaObserver(observers=[obs1, obs2], fn=lambda x, y: x-y)
  'qp': includes all body info (pos, rot, vel, ang) of the component
  'root_joints': includes root body info and all joint info (pos, vel) of the
    component
  'root_z_joints': the same as 'root_joints' but remove root body's pos[:2]

index_preprocess() converts special specs, e.g. ('key1', 2), into int indices
  for index_obs()

index_obs() allows indexing with a list of indices
"""
import abc
import collections
import copy
import json
from typing import Any, Dict, List, Tuple, Union

import brax
from brax.envs import Env
from brax.experimental.braxlines.common import sim_utils
from brax.experimental.composer import component_editor
from brax.experimental.composer import composer
from procedural_envs.misc.quaternion import quat_inverse
from procedural_envs.misc.quaternion import quat_multiply
from procedural_envs.misc.quaternion import quat2expmap
from procedural_envs.tasks.observation_config.one_hot_limb_id import AMORPHEUS, AMORPHEUS_GOAL

from google.protobuf.json_format import MessageToJson
from jax import numpy as jnp


EXTRA_ROOT_NODE_DICT = {
  'handsup': {'Z_Target': 'agent1___$ Body 4_0'},
  'unimal_handsup': {'Z_Target': 'agent1___limb_end_0'},
  'ant_handsup2': {'Z_Target_1': 'agent1___$ Body 4_0', 'Z_Target_2': 'agent1___$ Body 4_2'},
  'centipede_handsup2': {'Z_Target_1': 'agent1___$ Body 4_0', 'Z_Target_2': 'agent1___$ Body 5_2'}
}


class Observer(abc.ABC):
  """Observer."""

  def __init__(self, name: str = None, indices: Tuple[int] = None):
    assert name
    self.name = name
    if isinstance(indices, int):
      indices = (indices,)
    self.indices = indices
    self.initialized = False

  def initialize(self, sys):
    del sys
    self.initialized = True

  def index_obs(self, obs: jnp.ndarray):
    if self.indices is not None:
      obs = obs[..., self.indices]
    return obs

  def get_obs(self, sys, qp: brax.QP, info: brax.Info,
              cached_obs_dict: Dict[str, jnp.ndarray], component: Dict[str,
                                                                       Any]):
    if not self.initialized:
      self.initialize(sys)
    if self.name in cached_obs_dict:
      return cached_obs_dict[self.name].copy()
    obs = self._get_obs(sys, qp, info, cached_obs_dict, component)
    obs = self.index_obs(obs)
    return obs

  @abc.abstractmethod
  def _get_obs(self, sys, qp: brax.QP, info: brax.Info,
               cached_obs_dict: Dict[str, jnp.ndarray], component: Dict[str,
                                                                        Any]):
    raise NotImplementedError

  def __str__(self) -> str:
    return self.name

  def __repr__(self) -> str:
    return self.name


LAMBDA_FN_MAPPING = {
    '-': lambda x, y: x - y,
}


class LambdaObserver(Observer):
  """LambdaObserver."""

  def __init__(self, observers: List[Observer], fn, **kwargs):
    self.observers = observers
    if isinstance(fn, str):
      fn = LAMBDA_FN_MAPPING[fn]
    self.fn = fn
    super().__init__(**kwargs)

  def initialize(self, sys):
    for o in self.observers:
      o.initialize(sys)
    super().initialize(sys)

  def _get_obs(self, sys, qp: brax.QP, info: brax.Info,
               cached_obs_dict: Dict[str, jnp.ndarray], component: Dict[str,
                                                                        Any]):
    obses = [
        o.get_obs(sys, qp, info, cached_obs_dict, component)
        for o in self.observers
    ]
    return self.fn(*obses)


class SimObserver(Observer):
  """SimObserver."""

  def __init__(self,
               sdtype: str = 'body',
               sdcomp: str = 'pos',
               sdname: str = '',
               comp_name: str = '',
               name: str = None,
               indices: Tuple[int] = None,
               **kwargs):
    sdname = component_editor.concat_name(sdname, comp_name)
    if not name:
      name = f'{sdtype}_{sdcomp}:{sdname}'
      if indices:
        name = f'{name}[{indices}]'
    self.sdtype = sdtype
    self.sdcomp = sdcomp
    self.sdname = sdname
    super().__init__(name=name, indices=indices, **kwargs)

  def initialize(self, sys):
    self.sim_indices, self.sim_info, self.sim_mask = sim_utils.names2indices(
        sys.config, names=[self.sdname], datatype=self.sdtype)
    self.sim_indices = self.sim_indices[0]  # list -> int
    super().initialize(sys)

  def _get_obs(self,
               sys,
               qp: brax.QP,
               info: brax.Info,
               cached_obs_dict: Dict[str, jnp.ndarray],
               component: Dict[str, Any] = None):
    """Get observation."""
    if self.sdtype == 'body':
      assert self.sdcomp in ('pos', 'rot', 'ang', 'vel'), self.sdcomp
      obs = getattr(qp, self.sdcomp)[self.sim_indices]
    elif self.sdtype == 'joint':
      joint_obs_dict = sim_utils.get_joint_value(sys, qp, self.sim_info)
      obs = list(joint_obs_dict.values())[0]
    elif self.sdtype == 'contact':
      assert self.sdcomp in ('vel', 'ang'), self.sdcomp
      v = getattr(info.contact, self.sdcomp)[self.sim_indices]
      v = jnp.clip(v, -1, 1)
      obs = jnp.reshape(v, v.shape[:-2] + (-1,))
    else:
      raise NotImplementedError(self.sdtype)
    return obs


class GraphObserver(Observer):
  """GraphObserver."""

  def __init__(self,
               name: str = None,
               one_hot_limb_id: bool = True,
               one_hot_limb_id_dict: dict = AMORPHEUS,
               vel_clipping: bool = True,
               vel_clipping_range: Tuple[float] = (-10., 10.),
               angle_limit: float = 0.,
               mass_clipping_range: float = 100.,
               inertia_clip_range: float = 10.,
               scale_stiffness: float = 0.001,
               scale_angular_damping: float = 0.1,
               scale_actuator_strength: float = 0.01,
               rel_obs: Dict[str, bool] = {
                 'pos': False, 'rot': False, 'vel': False, 'ang': False},
               is_rel_obs_only: Dict[str, bool] = {
                 'pos': False, 'rot': False, 'vel': False, 'ang': False},
               add_joint_angle: bool = True,
               add_joint_range: bool = True,
               add_joint_vel: bool = False,
               add_egocentric_pos: bool = True,
               morphological_parameters: Dict[str, bool] = {
                 'shape': False,  # (radius, radius, length) or (x, y, z)
                 'mass': False,  # should be clipped?
                 'inertia': False,  # most of bodies have (1, 1, 1)
                 'friction': False,
                 'frozen': False,  # 0 or 1 (frozen)
                 'stiffness': False,  # from joint; 5000
                 'angular_damping': False ,  # from joint; 35
                 'actuator_strength': False,  # from actuator; 35
                 'actuator_dof_idx': False,  # from actuator; one-hot vector
                },
                root: str = 'torso',
                extra_root_node: Dict[str, str] = {},
                num_goal: int = 3,
                one_hot_goal_edge: bool = False,
                direct_goal_edge: bool = False,
               ):
    assert name
    self.name = name
    self.one_hot_limb_id = one_hot_limb_id
    self.one_hot_limb_id_dict = one_hot_limb_id_dict
    self.vel_clipping = vel_clipping
    self.vel_clipping_range = vel_clipping_range
    self.angle_limit = angle_limit
    self.mass_clipping_range = mass_clipping_range
    self.inertia_clip_range = inertia_clip_range
    self.scale_stiffness = scale_stiffness
    self.scale_angular_damping = scale_angular_damping
    self.scale_actuator_strength = scale_actuator_strength
    # original + relative_observations from parent to children
    self.rel_obs = rel_obs
    # *only* relative_observations from parent to children
    self.is_rel_obs_only = is_rel_obs_only
    self.add_joint_vel = add_joint_vel
    self.add_joint_angle = add_joint_angle
    self.add_joint_range = add_joint_range
    self.add_egocentric_pos = add_egocentric_pos
    self.morphological_parameters = morphological_parameters
    self.extra_root_node = extra_root_node
    self.num_goal = num_goal
    self.one_hot_goal_edge = one_hot_goal_edge
    self.direct_goal_edge = direct_goal_edge
    self.initialized = False

  def initialize(self, sys, root, task_edge=[[],[],[]]):
    self.root = root
    # get properties
    self.node_names = sim_utils.get_names(sys.config, 'body')[:-1]  # removing Ground
    # self.joint_names = sim_utils.get_names(sys.config, 'joint')
    self.actuator_names = sim_utils.get_names(sys.config, 'actuator')
    # get maximum dof to determine graph shape
    self.actuator_config = sim_utils.names2indices(
        sys.config, names=self.actuator_names, datatype='actuator')[1]
    self.max_dof = max([len(v) for v in self.actuator_config.values()])
    self.num_duplicate_node = sum([len(v) - 1 for v in self.actuator_config.values()])
    # assert self.max_dof <= 2  # assuming dof is at most 2
    assert self.max_dof <= 3
    self.action_shapes = collections.OrderedDict(
      [
        (k, dict(start=v[0], end=v[-1] + 1, size=len(v), shape=(len(v),))) for k, v in self.actuator_config.items()
        ]
      )
    # get joint_idx_dict for angle & vel
    self.joint_idx_dict = {}
    dof_counter = [0, 0, 0]
    for name in self.actuator_names:
      dof = self.action_shapes[name]['size']
      dof_counter[dof-1] += 1
      self.joint_idx_dict[name] = {'dof': dof, 'index': dof_counter[dof-1] - 1}
    for v in self.extra_root_node.values():  # e.g. extra_root_node = {'children': 'parent'}
      assert v in self.node_names, f'{v} is not in {self.node_names}'
    # task edge (assuming each goal dim is at most 2)
    self.task_edge = jnp.zeros((len(self.node_names), self.num_goal*2))
    self.goal_node_names = []
    for i, task in enumerate(task_edge):
      if len(task) > 0:
        # assumption
        self.goal_node_names.append(task[0])
      for node_idx, node in enumerate(self.node_names):
        if node in task:
          self.task_edge = self.task_edge.at[jnp.index_exp[node_idx, i*2:(i+1)*2]].set(jnp.ones(2,))
    assert len(self.goal_node_names) > 0
    self.initialized = True

  def get_obs(self, sys, qp: brax.QP, info: brax.Info,
              cached_obs_dict: Dict[str, jnp.ndarray],  # dummy
              component: Dict[str, Any]):
    if not self.initialized:
      self.initialize(sys, component['root'])
    obs = self._get_obs(sys, qp, info, cached_obs_dict, component)
    return obs

  def _get_obs(self,
               sys,
               qp: brax.QP,
               info: brax.Info,
               cached_obs_dict: Dict[str, jnp.ndarray],
               component: Dict[str, Any] = None):
    # initialize observation
    obs = []
    root_pos = qp.pos[sys.body.index[self.root]]
    joint_angle = {d: sys.joints[d].angle_vel(qp)[0] for d in range(self.max_dof)}
    joint_vel = {d: sys.joints[d].angle_vel(qp)[1] for d in range(self.max_dof)}

    # index for joints.angle_vel per dof, joint name, parents node
    child_dict = {j.child: (i, j.name, j.parent) for i, j in enumerate(sys.config.joints)}
    actuator_dict = {a.joint: (i, a.name, a.strength) for i, a in enumerate(sys.config.actuators)}

    _node_names = copy.deepcopy(self.node_names)
    if self.direct_goal_edge:
      _node_names = [v for v in _node_names if not v in self.goal_node_names]
      self.direct_task_edge = jnp.array(copy.deepcopy(self.task_edge))
      for i, goal_node in enumerate(self.goal_node_names):
        goal_pos = qp.pos[sys.body.index[goal_node]]
        if self.add_egocentric_pos:
          goal_pos -= root_pos
        if 'Z_Target' in goal_node:
          goal_pos = jnp.zeros((2,)).at[0].set(goal_pos[2])  # (Z, 0)
        else:
          goal_pos = goal_pos[:2]
        self.direct_task_edge = self.direct_task_edge.at[jnp.index_exp[:, i*2:(i+1)*2]].set(self.direct_task_edge[:, i*2:(i+1)*2] * goal_pos)

    for node_idx, node in enumerate(_node_names):
      local_obs = []
      qp_idx = sys.body.index[node]
      _node = component_editor.split_name(node)[0]
      if node in child_dict.keys():
        parent_idx = sys.body.index[child_dict[node][2]]
      elif _node in self.extra_root_node.keys():
        parent_idx = sys.body.index[self.extra_root_node[_node]]
      else:
        parent_idx = sys.body.index[self.root]
      disjoint = (node != self.root) and not (node in child_dict.keys())

      # get joint info
      if (node == self.root) or disjoint:
        angle = jnp.array([0.0])
        j_vel = jnp.array([0.0])
        # dummy
        angle2 = None
        j_vel2 = None
        angle3 = None
        j_vel3 = None
        j_dof = 0
      else:
        j_name = child_dict[node][1]
        j_dof = self.joint_idx_dict[j_name]['dof']
        j_idx = self.joint_idx_dict[j_name]['index']
        angle = jnp.degrees(
            jnp.array([joint_angle[j_dof-1][0][j_idx]])
            )
        j_vel = jnp.array([joint_vel[j_dof-1][0][j_idx]])
        # dummy
        angle2 = None
        j_vel2 = None
        angle3 = None
        j_vel3 = None
        if j_dof == 2:
          local_obs2 = []
          angle2 = jnp.degrees(
            jnp.array([joint_angle[j_dof-1][1][j_idx]])
            )
          j_vel2 = jnp.array([joint_vel[j_dof-1][1][j_idx]])
        elif j_dof == 3:
          local_obs2 = []
          angle2 = jnp.degrees(
            jnp.array([joint_angle[j_dof-1][1][j_idx]])
            )
          j_vel2 = jnp.array([joint_vel[j_dof-1][1][j_idx]])
          local_obs3 = []
          angle3 = jnp.degrees(
            jnp.array([joint_angle[j_dof-1][2][j_idx]])
            )
          j_vel3 = jnp.array([joint_vel[j_dof-1][2][j_idx]])

      # wrap append operation
      def local_obs_append(vec, vec2=None, vec3=None):
        local_obs.append(vec)
        if j_dof >= 2:
          if vec2 is None:
            local_obs2.append(vec)
          else:
            local_obs2.append(vec2)
        if j_dof == 3:
          if vec3 is None:
            local_obs3.append(vec)
          else:
            local_obs3.append(vec3)

      # add one_hot_limb_id
      if self.one_hot_limb_id:
        local_obs_append(self.one_hot_limb_id_dict[_node])

      # add relative position
      if self.rel_obs['pos']:
        if node == self.root:
          if self.add_egocentric_pos:
            p_pos = root_pos
          else:
            p_pos = jnp.array([0, 0, 0])
        else:
          p_pos = qp.pos[parent_idx]
        local_obs_append(qp.pos[qp_idx] - p_pos)
      # add original (egocentric) position
      if not self.is_rel_obs_only['pos']:
        if self.add_egocentric_pos:
          local_obs_append(qp.pos[qp_idx] - root_pos)
        else:
          local_obs_append(qp.pos[qp_idx])

      # add relative velocity
      if self.rel_obs['vel']:
        p_vel = jnp.array([0, 0, 0]) if node == self.root else qp.vel[parent_idx]
        if self.vel_clipping:
          node_vel = jnp.clip(
              qp.vel[qp_idx] - p_vel, self.vel_clipping_range[0], self.vel_clipping_range[1])
        else:
          node_vel = qp.vel[qp_idx] - p_vel
        local_obs_append(node_vel)
      # add original velocity
      if not self.is_rel_obs_only['vel']:
        if self.vel_clipping:
          node_vel = jnp.clip(
              qp.vel[qp_idx], self.vel_clipping_range[0], self.vel_clipping_range[1])
        else:
          node_vel = qp.vel[qp_idx]
        local_obs_append(node_vel)

      # add relative angular velocity
      if self.rel_obs['ang']:
        p_ang = jnp.array([0, 0, 0]) if node == self.root else qp.ang[parent_idx]
        local_obs_append(qp.ang[qp_idx] - p_ang)
      # add original angular velocity
      if not self.is_rel_obs_only['ang']:
        local_obs_append(qp.ang[qp_idx])

      # add exponential-mapped relative quatanion
      if self.rel_obs['rot']:
        p_rot = jnp.array([1., 0., 0., 0.]) if node == self.root else qp.rot[parent_idx]
        quat_diff = quat_multiply(qp.rot[qp_idx], quat_inverse(p_rot))
        local_obs_append(quat2expmap(quat_diff))
      # add exponential-mapped original quatanion
      if not self.is_rel_obs_only['rot']:
        local_obs_append(quat2expmap(qp.rot[qp_idx]))

      # add joint angle & joint limit
      if (node == self.root) or disjoint:
        j_range = jnp.array([0.0, 0.0])
        j_range2 = None
        j_range3 = None
      else:
        j_range_dict = json.loads(
            MessageToJson(sys.config.joints[child_dict[node][0]].angle_limit[0]))
        j_range = [
            j_range_dict.get('min', self.angle_limit),
            j_range_dict.get('max', self.angle_limit),
            ]
        j_range = jnp.array(j_range)
        if len(j_range_dict) == 0:
          angle = jnp.array([0.5])
        else:
          angle = (angle - j_range[0]) / (j_range[1] - j_range[0])
        j_range = (180.0 + j_range) / 360.0
        j_range2 = None
        j_range3 = None
        if j_dof == 2:
          j_range_dict2 = json.loads(
              MessageToJson(sys.config.joints[child_dict[node][0]].angle_limit[1]))
          j_range2 = [
              j_range_dict2.get('min', self.angle_limit),
              j_range_dict2.get('max', self.angle_limit),
              ]
          j_range2 = jnp.array(j_range2)
          if len(j_range_dict2) == 0:
            angle2 = jnp.array([0.5])
          else:
            angle2 = (angle2 - j_range2[0]) / (j_range2[1] - j_range2[0])
          j_range2 = (180.0 + j_range2) / 360.0
        elif j_dof == 3:
          j_range_dict3 = json.loads(
              MessageToJson(sys.config.joints[child_dict[node][0]].angle_limit[2]))
          j_range3 = [
              j_range_dict3.get('min', self.angle_limit),
              j_range_dict3.get('max', self.angle_limit),
              ]
          j_range3 = jnp.array(j_range3)
          if len(j_range_dict3) == 0:
            angle3 = jnp.array([0.5])
          else:
            angle3 = (angle3 - j_range3[0]) / (j_range3[1] - j_range3[0])
          j_range3 = (180.0 + j_range3) / 360.0
      if self.add_joint_angle:
        local_obs_append(angle, angle2, angle3)
      if self.add_joint_range:
        local_obs_append(j_range, j_range2, j_range3)

      # add joint velocity
      if self.add_joint_vel:
        local_obs_append(j_vel, j_vel2, j_vel3)

      # add morphology infomation
      bodies_dict = json.loads(MessageToJson(sys.config.bodies[node_idx]))
      # (radius, radius, length) or (x, y, z)
      if self.morphological_parameters['shape']:
        if 'capsule' in bodies_dict['colliders'][0].keys():
          radius = bodies_dict['colliders'][0]['capsule'].get('radius', 0.0)
          length = bodies_dict['colliders'][0]['capsule'].get('length', radius)
          m_shape = jnp.array([radius, radius, length])
        elif 'sphere' in bodies_dict['colliders'][0].keys():
          radius = bodies_dict['colliders'][0]['sphere'].get('radius', 0.0)
          m_shape = jnp.array([radius, radius, radius])
        elif 'box' in bodies_dict['colliders'][0].keys():
          m_x = bodies_dict['colliders'][0]['box'].get('halfsize').get('x', 0.0)
          m_y = bodies_dict['colliders'][0]['box'].get('halfsize').get('y', 0.0)
          m_z = bodies_dict['colliders'][0]['box'].get('halfsize').get('z', 0.0)
          m_shape = jnp.array([m_x, m_y, m_z])
        local_obs_append(m_shape)
      # clipped
      if self.morphological_parameters['mass']:
        m_mass = jnp.clip(jnp.array([bodies_dict.get('mass', 0.0)]), 0.0, self.mass_clipping_range)
        local_obs_append(jnp.array(m_mass))
      # most of bodies have (1, 1, 1)
      if self.morphological_parameters['inertia']:
        if 'inertia' in bodies_dict.keys():
          m_ix = bodies_dict['inertia'].get('x', 1.0)
          m_iy = bodies_dict['inertia'].get('y', 1.0)
          m_iz = bodies_dict['inertia'].get('z', 1.0)
          m_inertia = jnp.array([m_ix, m_iy, m_iz])
        else:
          m_inertia = jnp.array([1., 1., 1.])
        m_inertia = jnp.clip(m_inertia, 0.0, self.inertia_clip_range)
        local_obs_append(m_inertia)
      if self.morphological_parameters['friction']:
        if 'material' in bodies_dict['colliders'][0].keys():
          m_friction = jnp.array([bodies_dict['colliders'][0]['material'].get('friction', 1.0)])
        else:
          m_friction = jnp.array([1.0])
        local_obs_append(m_friction)
      # 0 or 1 (frozen)
      if self.morphological_parameters['frozen']:
        if 'frozen' in bodies_dict.keys():
          m_frozen = jnp.array([float(bodies_dict['frozen'].get('all', False))])
        else:
          m_frozen = jnp.array([0.0])
        local_obs_append(m_frozen)
      # joint_info
      if (node == self.root) or disjoint:
        if self.morphological_parameters['stiffness']:
          j_stiffness = jnp.zeros(1)
          local_obs_append(j_stiffness)
        if self.morphological_parameters['angular_damping']:
          j_angular_damping = jnp.zeros(1)
          local_obs_append(j_angular_damping)
        if self.morphological_parameters['actuator_strength']:
          j_actuator_strength = jnp.zeros(1)
          local_obs_append(j_actuator_strength)
        if self.morphological_parameters['actuator_dof_idx']:
          j_actuator_dof_idx = jnp.zeros(3)
          local_obs_append(j_actuator_dof_idx)
      else:
        joints_dict = json.loads(MessageToJson(sys.config.joints[child_dict[node][0]]))
        # from joint; 5000
        if self.morphological_parameters['stiffness']:
          j_stiffness = jnp.array([joints_dict.get('stiffness', 0.0)])*self.scale_stiffness
          local_obs_append(j_stiffness)
        # from joint; 35
        if self.morphological_parameters['angular_damping']:
          j_angular_damping = jnp.array([joints_dict.get('angularDamping', 0.0)])*self.scale_angular_damping
          local_obs_append(j_angular_damping)
        if self.morphological_parameters['actuator_strength']:
          j_actuator_strength = jnp.array([actuator_dict[child_dict[node][1]][2]])*self.scale_actuator_strength
          local_obs_append(j_actuator_strength)
        if self.morphological_parameters['actuator_dof_idx']:
          j_actuator_dof_idx = jnp.array([1., 0, 0])
          j_actuator_dof_idx2 = None
          j_actuator_dof_idx3 = None
          if j_dof >= 2:
            j_actuator_dof_idx2 = jnp.array([0, 1., 0])
          if j_dof == 3:
            j_actuator_dof_idx3 = jnp.array([0, 0, 1.])
          local_obs_append(j_actuator_dof_idx, j_actuator_dof_idx2, j_actuator_dof_idx3)

      if self.one_hot_goal_edge:
        local_obs_append(self.task_edge[node_idx])
      elif self.direct_goal_edge:
        local_obs_append(self.direct_task_edge[node_idx])

      # concat all local_obs
      obs.append(jnp.concatenate(local_obs))
      if j_dof >= 2:
        obs.append(jnp.concatenate(local_obs2))
      if j_dof == 3:
        obs.append(jnp.concatenate(local_obs3))
    # return local_obs_dim * num_node
    obs = jnp.concatenate(obs).ravel()
    return obs


def index_preprocess(indices: List[Any], env: Env = None) -> List[int]:
  """Preprocess indices to a list of ints and a list of str labels."""
  if indices is None:
    return None
  int_indices = []
  labels = []
  for index in indices:
    if isinstance(index, int):
      int_indices += [index]
      labels += [f'obs[{index}]']
    elif isinstance(index, tuple):
      assert len(index) == 2, 'tuple indexing is of form: (obs_dict_key, index)'
      key, i = index
      assert isinstance(env, composer.ComponentEnv), env
      assert env.observation_size  # ensure env.observer_shapes is set
      obs_shape = env.observer_shapes
      assert key in obs_shape, f'{key} not in {tuple(obs_shape.keys())}'
      int_indices += [obs_shape[key]['start'] + i]
      labels += [f'{key}[{i}]']
    else:
      raise NotImplementedError(index)
  return int_indices, labels


def index_preprocess(indices: List[Any], env: Env = None) -> List[int]:
  """Preprocess indices to a list of ints and a list of str labels."""
  if indices is None:
    return None
  int_indices = []
  labels = []
  for index in indices:
    if isinstance(index, int):
      int_indices += [index]
      labels += [f'obs[{index}]']
    elif isinstance(index, tuple):
      assert len(index) == 2, 'tuple indexing is of form: (obs_dict_key, index)'
      key, i = index
      assert isinstance(env, composer.ComponentEnv), env
      assert env.observation_size  # ensure env.observer_shapes is set
      obs_shape = env.observer_shapes
      assert key in obs_shape, f'{key} not in {tuple(obs_shape.keys())}'
      int_indices += [obs_shape[key]['start'] + i]
      labels += [f'{key}[{i}]']
    else:
      raise NotImplementedError(index)
  return int_indices, labels


def index_obs(obs: jnp.ndarray, indices: List[Any], env: Env = None):
  """Index observation vector."""
  int_indices = index_preprocess(indices, env)
  return obs.take(int_indices, axis=-1)


def initialize_observers(observers: List[Union[Observer, str]], sys):
  """Initialize observers."""
  for o in observers:
    if isinstance(o, Observer):
      o.initialize(sys)


STRING_OBSERVERS = ('qp', 'root_joints', 'root_z_joints', 'cfrc')


def get_obs_dict(sys, qp: brax.QP, info: brax.Info, observer: Union[str,
                                                                    Observer],
                 cached_obs_dict: Dict[str, jnp.ndarray], component: Dict[str,
                                                                          Any]):
  """Observe."""
  obs_dict = collections.OrderedDict()
  if isinstance(observer, Observer):
    obs_dict[observer.name] = observer.get_obs(sys, qp, info, cached_obs_dict,
                                               component)
  elif observer == 'qp':
    # get all positions/orientations/velocities/ang velocities of all bodies
    bodies = component['bodies']
    indices = sim_utils.names2indices(sys.config, bodies, 'body')[0]
    for type_ in ('pos', 'rot', 'vel', 'ang'):
      for index, b in zip(indices, bodies):
        v = getattr(qp, type_)[index]
        key = f'body_{type_}:{b}'
        obs_dict[key] = v
  elif observer in ('root_joints', 'root_z_joints'):
    # get all positions/orientations/velocities/ang velocities of root bodies
    root = component['root']
    index = sim_utils.names2indices(sys.config, root, 'body')[0][0]
    for type_ in ('pos', 'rot', 'vel', 'ang'):
      v = getattr(qp, type_)[index]
      if observer == 'root_z_joints' and type_ == 'pos':
        # remove xy position
        v = v[2:]
      obs_dict[f'body_{type_}:{root}'] = v
    # get all joints
    joints = component['joints']
    _, joint_info, _ = sim_utils.names2indices(sys.config, joints, 'joint')
    joint_obs_dict = sim_utils.get_joint_value(sys, qp, joint_info)
    obs_dict = collections.OrderedDict(
        list(obs_dict.items()) + list(joint_obs_dict.items()))
  elif observer == 'cfrc':
    # external contact forces:
    # delta velocity (3,), delta ang (3,) * N bodies in the system
    bodies = component['bodies']
    indices = sim_utils.names2indices(sys.config, bodies, 'body')[0]
    for i, b in zip(indices, bodies):
      for type_ in ('vel', 'ang'):
        v = getattr(info.contact, type_)[i]
        v = jnp.clip(v, -1, 1)
        v = jnp.reshape(v, v.shape[:-2] + (-1,))
        key = f'contact_{type_}:{b}'
        obs_dict[key] = v
  else:
    raise NotImplementedError(observer)
  return obs_dict


def get_component_observers(component: Dict[str, Any],
                            observer_type: str = 'qp',
                            **observer_kwargs):
  """Get component-based Observers."""
  del component, observer_kwargs
  raise NotImplementedError(observer_type)


def get_edge_observers(component1: Dict[str, Any],
                       component2: Dict[str, Any],
                       observer_type: str = 'root_vec',
                       **observer_kwargs):
  """Get edge-based Observers."""
  if observer_type == 'root_vec':
    root1 = component1['root']
    root2 = component2['root']
    return LambdaObserver(
        name=f'dist__{root1}__{root2}',
        fn='-',
        observers=[
            SimObserver(sdname=component1['root'], **observer_kwargs),
            SimObserver(sdname=component2['root'], **observer_kwargs)
        ],
    )
  else:
    raise NotImplementedError(observer_type)
