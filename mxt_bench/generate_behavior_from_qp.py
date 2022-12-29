# generate begavioral data (s, s', a, r, d, t) from the saved state.qp.
import copy
import os
import pickle
import time

from absl import app
from absl import flags
from absl import logging
from brax.experimental.composer import data_utils
import brax
import jax
import jax.numpy as jnp

from procedural_envs import composer
from procedural_envs.misc.observers import GraphObserver, EXTRA_ROOT_NODE_DICT
from procedural_envs.tasks.observation_config import obs_config_dict
from procedural_envs.tasks.task_config import TASK_CONFIG

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'ant_touch_4', 'Name of environment to collect data.')
flags.DEFINE_string('task_name', 'ant_touch', 'Name of task to train.')
flags.DEFINE_string('obs_config', 'amorpheus', 'Name of observation config to train.')
flags.DEFINE_string('obs_config2', 'mtg_v2_base_m', 'Name of observation config to collect data.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('action_repeat', 1, 'Action repeat.')
flags.DEFINE_integer('episode_length', 1000, 'Episode length.')
flags.DEFINE_integer('batch_size', 1958, 'Batch size of environment.')
flags.DEFINE_string('logdir', '../data/', 'Logdir to save dataset.')
flags.DEFINE_string('dataset_path', '', 'Path to saved dataset.')
flags.DEFINE_string('qp_path', '', 'Path to saved qp dataset.')
flags.DEFINE_string('data_name', '', 'Filename of data.')


def main(unused_argv):
  environment_params = {
      'env_name': FLAGS.env,
      'obs_config': FLAGS.obs_config,
      'obs_config2': FLAGS.obs_config2,
  }
  obs_config = obs_config_dict[FLAGS.obs_config]
  obs_config2 = obs_config_dict[FLAGS.obs_config2]

  if ('handsup2' in FLAGS.env) and ('ant' in FLAGS.env):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['ant_handsup2']
    obs_config2['extra_root_node'] = EXTRA_ROOT_NODE_DICT['ant_handsup2']
  elif ('handsup2' in FLAGS.env) and ('centipede' in FLAGS.env):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['centipede_handsup2']
    obs_config2['extra_root_node'] = EXTRA_ROOT_NODE_DICT['centipede_handsup2']
  elif ('handsup' in FLAGS.task_name) and ('unimal' in FLAGS.task_name):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['unimal_handsup']
    obs_config2['extra_root_node'] = EXTRA_ROOT_NODE_DICT['unimal_handsup']
  elif 'handsup' in FLAGS.env:
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['handsup']
    obs_config2['extra_root_node'] = EXTRA_ROOT_NODE_DICT['handsup']

  env_config = copy.deepcopy(environment_params)
  observer = GraphObserver(name=FLAGS.obs_config, **obs_config)
  observer2 = GraphObserver(name=FLAGS.obs_config2, **obs_config2)

  # create env
  env_fn = composer.create_fn(env_name=FLAGS.env, observer=observer, observer2=observer2)
  env = env_fn(action_repeat=FLAGS.action_repeat, episode_length=FLAGS.episode_length, batch_size=FLAGS.batch_size)
  logging.info(f'action_size: {env.action_size}')
  logging.info(f'observation_size: {env.observation_size}')
  logging.info(f'observation_size2: {env.observation_size2}')
  logging.info(f'num_node in observations: {env.num_node}')
  env_config['action_size'] = env.action_size
  env_config['observation_size'] = env.observation_size
  env_config['observation_size2'] = env.observation_size2
  observation_size = env.observation_size
  observation_size2 = env.observation_size2
  num_limb = env.num_node
  action_size = env.action_size

  is_claw = True if 'claw' in FLAGS.env else False

  state_qp_shape = num_limb + 1
  if is_claw:
    state_qp_shape = num_limb + 1 - int(action_size / 4)  # int(action_size / 4) == num_legs

  dataset_path = TASK_CONFIG[FLAGS.task_name][FLAGS.env]['amorpheus']['dataset_path'] if FLAGS.dataset_path == '' else FLAGS.dataset_path
  qp_path = TASK_CONFIG[FLAGS.task_name][FLAGS.env]['qp_path'] if FLAGS.qp_path == '' else FLAGS.qp_path
  logging.info('Loding amorpheus dataset: ' + dataset_path)
  logging.info('Loding qp dataset: ' + qp_path)

  xt = time.time()

  trans_dim = observation_size * 2 + num_limb * 2 + 3
  with open(dataset_path, 'rb') as f:
    data = pickle.load(f)
  data = data.reshape(-1, trans_dim)
  total_env_steps = data.shape[0]
  with open(qp_path, 'rb') as f:
    qp = pickle.load(f)
  qp = qp.reshape(-1, qp.shape[2])
  assert qp.shape[0] == total_env_steps

  collected_data = jnp.zeros((total_env_steps, observation_size2 * 2 + num_limb * 2 + 3))

  def get_obs2(s_qp):
    s_info = env.sys.info(s_qp)
    obs_dict, _ = env._get_obs2(s_qp, s_info)
    obs = data_utils.concat_array(obs_dict, env.observer_shapes2)
    return obs

  def qp_convert(s_qp):
    return brax.QP(
      pos=s_qp[:, 0:state_qp_shape * 3].reshape(-1, state_qp_shape, 3),
      rot=s_qp[:, state_qp_shape * 3:state_qp_shape * 7].reshape(-1, state_qp_shape, 4),
      vel=s_qp[:, state_qp_shape * 7:state_qp_shape * 10].reshape(-1, state_qp_shape, 3),
      ang=s_qp[:, state_qp_shape * 10:state_qp_shape * 13].reshape(-1, state_qp_shape, 3))

  # jit
  replay_from_qp_a_fn = jax.jit(jax.vmap(env.replay_from_qp_a))
  get_obs2_fn = jax.jit(jax.vmap(get_obs2))
  qp_convert_fn = jax.jit(qp_convert)

  assert total_env_steps % FLAGS.batch_size == 0
  num_iter = total_env_steps // FLAGS.batch_size
  t = time.time()
  logging.info('Starting data collection %s', t - xt)
  for i in range(num_iter):
    step = data[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
    step_qp = qp[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
    action = step[:, observation_size*2:observation_size*2+num_limb]
    action_and_others = step[:, observation_size*2:]
    state_qp = qp_convert_fn(step_qp)
    # get current state representation
    obs2 = get_obs2_fn(state_qp)
    # get next state representation
    nobs2 = replay_from_qp_a_fn(state_qp, action[:, 1:1+action_size])
    new_step = jnp.concatenate([obs2, nobs2, action_and_others], axis=-1)
    collected_data = collected_data.at[jnp.index_exp[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]].set(new_step)

  collected_data = collected_data.reshape(2, -1, observation_size2 * 2 + num_limb * 2 + 3)
  logging.info('Finishing data collection %s', time.time() - t)
  data_name = FLAGS.env + '_' + FLAGS.obs_config2 if FLAGS.data_name == '' else FLAGS.data_name
  with open(os.path.join(FLAGS.logdir, f'{data_name}.pkl'), 'wb') as f:
    pickle.dump(collected_data, f, protocol=4)
  logging.info('Dataset is saved at: ' + os.path.join(FLAGS.logdir, f'{data_name}.pkl'))
  logging.info(f'Dataset shape: {collected_data.shape}')

if __name__ == '__main__':
  app.run(main)
