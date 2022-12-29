# generate begavioral data (s, s', a, r, d, t) and the save state.qp.
import copy
import os
import pickle
import pprint
import time
from typing import Callable, Optional

from absl import app
from absl import flags
from absl import logging
from brax import envs
from brax.io import model
from brax.training import distribution
from brax.training import normalization
import flax
import jax
import jax.numpy as jnp

from algo import ppo_mlp
from procedural_envs import composer
from procedural_envs.misc.observers import GraphObserver, EXTRA_ROOT_NODE_DICT
from procedural_envs.tasks.observation_config import obs_config_dict
from procedural_envs.tasks.task_config import TASK_CONFIG

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'ant_touch_4', 'Name of environment to collect data.')
flags.DEFINE_string('task_name', 'ant_touch', 'Name of task to train.')
flags.DEFINE_string('obs_config', 'amorpheus', 'Name of observation config to train.')
flags.DEFINE_integer('total_env_steps', 97850,  # 1957 steps * 2 device * 25 parallel envs.
                     'Number of env steps to run training for.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_envs', 50, 'Number of envs to run in parallel.')
flags.DEFINE_integer('action_repeat', 1, 'Action repeat.')
flags.DEFINE_integer('episode_length', 1000, 'Episode length.')
flags.DEFINE_string('logdir', '../data/', 'Logdir to save dataset.')
flags.DEFINE_bool('normalize_observations', True,
                  'Whether to apply observation normalization.')
flags.DEFINE_integer('max_devices_per_host', None,
                     'Maximum number of devices to use per host. If None, '
                     'defaults to use as much as it can.')
flags.DEFINE_string('params_path', '', 'Path to saved params.')
flags.DEFINE_string('data_name', '', 'Filename of data.')


@flax.struct.dataclass
class ReplayBuffer:
  """Contains data related to a replay buffer."""
  data: jnp.ndarray
  current_position: jnp.ndarray
  current_size: jnp.ndarray


def collect_data(
    environment_fn: Callable[..., envs.Env],
    data_name: str,
    params_path: str,
    num_timesteps: int,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    seed=0,
    normalize_observations=True,
    local_state_size: int = 19,
    output_dir: Optional[str] = None,
    is_claw : bool = True,
):
  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
      jax.device_count(), process_count, process_id, local_device_count,
      local_devices_to_use)
  num_envs_per_device = num_envs // local_devices_to_use // process_count
  max_replay_size = (
      num_timesteps // action_repeat + num_envs) // local_devices_to_use // process_count
  num_steps_per_envs = max_replay_size // num_envs_per_device

  index = jnp.array(
      [[i * num_steps_per_envs, i] for i in range(num_envs_per_device)])

  steps_per_envs = jnp.ones((num_steps_per_envs, ), dtype=jnp.int32)

  key = jax.random.PRNGKey(seed)
  key, key_env, key_sample = jax.random.split(key, 3)
  # Make sure every process gets a different random key, otherwise they will be
  # doing identical work.
  key_env = jax.random.split(key_env, process_count)[process_id]
  key = jax.random.split(key, process_count)[process_id]
  # key_models should be the same, so that models are initialized the same way
  # for different processes

  core_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_envs_per_device,
      episode_length=episode_length)
  key_envs = jax.random.split(key_env, local_devices_to_use)
  step_fn = jax.jit(core_env.step)
  reset_fn = jax.jit(jax.vmap(core_env.reset))
  first_state = reset_fn(key_envs)

  observation_size = core_env.observation_size
  observation_size2 = core_env.observation_size2
  num_limb = observation_size // local_state_size
  action_size = core_env.action_size

  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=num_limb)

  state_qp_shape = num_limb + 1
  if is_claw:
    state_qp_shape = num_limb + 1 - int(action_size / 4)  # int(action_size / 4) == num_legs

  policy_model, _ = ppo_mlp.make_mlp_networks(parametric_action_distribution.param_size, observation_size)
  normalizer_params, policy_params = model.load_params(params_path)
  param_count = sum(x.size for x in jax.tree_leaves(policy_params))
  print(f'num_policy_param: {param_count}')

  _, _, obs_normalizer_apply_fn = (
      normalization.create_observation_normalizer(
          core_env.observation_size, normalize_observations,
          num_leading_batch_dims=2, pmap_to_devices=local_devices_to_use))

  def episodic_update_replay_buffer(carry, idx):
    replaybuffer_data, _newdata, current_position = carry
    _new_replay_data = jax.tree_multimap(
        lambda x, y: jax.lax.dynamic_update_slice_in_dim(
            x,
            y,
            current_position+idx[0],
            axis=0),
        replaybuffer_data,
        jnp.expand_dims(_newdata[idx[1]], axis=0))
    return (_new_replay_data, _newdata, current_position), ()

  def collect_and_update_buffer(carry, t):
    key, state, replay_buffer = carry
    # collect data
    key, key_sample = jax.random.split(key, 2)
    normalized_obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
    logits = policy_model.apply(policy_params, normalized_obs)
    actions = parametric_action_distribution.sample(
        logits, key_sample)
    limb_mask = jnp.ones_like(actions)
    nstate = step_fn(state, actions[:, 1:action_size+1])

    newdata = jnp.concatenate([
        state.obs2,
        nstate.obs2,
        actions,
        limb_mask,
        jnp.expand_dims(nstate.reward, axis=-1),
        jnp.expand_dims(1 - nstate.done, axis=-1),
        jnp.expand_dims(nstate.info['truncation'], axis=-1),
        state.qp.pos.reshape(-1, (state_qp_shape) * 3),
        state.qp.rot.reshape(-1, (state_qp_shape) * 4),
        state.qp.vel.reshape(-1, (state_qp_shape) * 3),
        state.qp.ang.reshape(-1, (state_qp_shape) * 3),
    ], axis=-1)

    # update buffer
    (new_replay_data, _, _), _ = jax.lax.scan(
        episodic_update_replay_buffer,
        (replay_buffer.data, newdata, replay_buffer.current_position),
        index, length=None)

    new_position = replay_buffer.current_position + t
    new_size = replay_buffer.current_size  # dummy

    return (key, nstate, ReplayBuffer(
        data=new_replay_data,
        current_position=new_position,
        current_size=new_size)), ()

  def run_collect_data(key, state, replay_buffer):
    (key, state, replay_buffer), _ = jax.lax.scan(
        collect_and_update_buffer, (key, state, replay_buffer),
        steps_per_envs, length=None)
    return key, state, replay_buffer

  run_collect_data = jax.pmap(run_collect_data, axis_name='i')

  replay_buffer = ReplayBuffer(
    data=jnp.zeros((local_devices_to_use, max_replay_size,
                    observation_size2 * 2 + num_limb * 2 + 1 + 1 + 1 + (3 + 4 + 3 + 3) * state_qp_shape)),
    current_size=jnp.zeros((local_devices_to_use,), dtype=jnp.int32),
    current_position=jnp.zeros((local_devices_to_use,), dtype=jnp.int32))

  t = time.time()
  logging.info('Starting data collection %s', t - xt)
  state = first_state
  key_sample = jnp.stack(jax.random.split(key_sample, local_devices_to_use))
  key_sample, state, replay_buffer = run_collect_data(
    key_sample, state, replay_buffer)
  logging.info('Finishing data collection %s', time.time() - t)
  # split qp from data
  data = replay_buffer.data[:, :, :observation_size2 * 2 + num_limb * 2 + 1 + 1 + 1]
  qp = replay_buffer.data[:, :, observation_size2 * 2 + num_limb * 2 + 1 + 1 + 1:]
  with open(os.path.join(output_dir, f'{data_name}.pkl'), 'wb') as f:
    pickle.dump(data, f, protocol=4)
  with open(os.path.join(output_dir, f'{FLAGS.env}_qp.pkl'), 'wb') as f:
    pickle.dump(qp, f, protocol=4)
  logging.info('Dataset is saved at: ' + os.path.join(output_dir, f'{data_name}.pkl'))
  logging.info('Dataset (qp) is saved at: ' + os.path.join(output_dir, f'{FLAGS.env}_qp.pkl'))
  logging.info(f'Dataset shape: {data.shape}')
  logging.info(f'Dataset (qp) shape: {qp.shape}')


def main(unused_argv):
  environment_params = {
      'env_name': FLAGS.env,
      'obs_config': FLAGS.obs_config,
  }
  obs_config = obs_config_dict[FLAGS.obs_config]

  if ('handsup2' in FLAGS.task_name) and ('ant' in FLAGS.task_name):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['ant_handsup2']
  elif ('handsup2' in FLAGS.task_name) and ('centipede' in FLAGS.task_name):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['centipede_handsup2']
  elif ('handsup' in FLAGS.task_name) and ('unimal' in FLAGS.task_name):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['unimal_handsup']
  elif 'handsup' in FLAGS.task_name:
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['handsup']

  env_config = copy.deepcopy(environment_params)
  observer = GraphObserver(name=FLAGS.obs_config, **obs_config)
  observer2= GraphObserver(name=FLAGS.obs_config, **obs_config)
  # create env
  env_fn = composer.create_fn(env_name=FLAGS.env, observer=observer, observer2=observer2)
  env = env_fn()
  logging.info(f'action_size: {env.action_size}')
  logging.info(f'observation_size: {env.observation_size}')
  logging.info(f'observation_size2: {env.observation_size2}')
  logging.info(f'num_node in observations: {env.num_node}')
  env_config['action_size'] = env.action_size
  env_config['observation_size'] = env.observation_size
  env_config['observation_size2'] = env.observation_size2

  is_claw = True if 'claw' in FLAGS.env else False

  logging.info('Loding qp and action: ' + FLAGS.params_path)

  data_name = FLAGS.env + '_' + FLAGS.obs_config if FLAGS.data_name == '' else FLAGS.data_name

  train_job_params = {
    'action_repeat': FLAGS.action_repeat,
    'data_name': data_name,
    'output_dir': FLAGS.logdir,
    'episode_length': FLAGS.episode_length,
    'local_state_size': env_config['observation_size'] // env.num_node,
    'normalize_observations': FLAGS.normalize_observations,
    'num_envs': FLAGS.num_envs,
    'num_timesteps': FLAGS.total_env_steps,
    'max_devices_per_host': FLAGS.max_devices_per_host,
    'params_path': FLAGS.params_path,
    'seed': FLAGS.seed,
    'is_claw': is_claw}

  config = copy.deepcopy(train_job_params)
  config['env'] = FLAGS.env
  pprint.pprint(config)

  collect_data(environment_fn=env_fn, **train_job_params)


if __name__ == '__main__':
  app.run(main)
