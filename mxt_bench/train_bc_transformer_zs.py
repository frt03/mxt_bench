"""Zero-shot evaluation with pre-trained policy."""
import copy
import functools
import os
import pickle
import pprint

from absl import app
from absl import flags
from brax.io import html
from brax.io import model
from brax.experimental.braxlines import experiments
from brax.experimental.braxlines.common import logger_utils
from datetime import datetime
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from algo import bc_transformer
from procedural_envs import composer
from procedural_envs.misc.observers import GraphObserver, EXTRA_ROOT_NODE_DICT
from procedural_envs.tasks.observation_config import obs_config_dict, obs_size_dict
from procedural_envs.tasks.task_config import ZERO_SHOT_TASK_CONFIG
from models.architecture_config import ARCHITECTURE_CONFIG

FLAGS = flags.FLAGS
flags.DEFINE_string('task_name', 'example', 'Name of task to train.')
flags.DEFINE_string('obs_config', 'mtg_v2_base_m', 'Name of observation (dataset) config to train.')
flags.DEFINE_string('architecture_config', 'transformer_pe', 'Name of architecture config to train.')
flags.DEFINE_integer('total_grad_steps', 100000,
                     'Number of gradient steps to run training for.')
flags.DEFINE_integer('eval_frequency', 10000, 'How many times to run an eval.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('action_repeat', 1, 'Action repeat.')
flags.DEFINE_integer('num_episodes', 3, 'Number of episodes used for training.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('episode_length', 1000, 'Episode length.')
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
flags.DEFINE_string('logdir', '../results', 'Logdir.')
flags.DEFINE_string('identifier', '', 'Identifier for logdir.')
flags.DEFINE_bool('normalize_observations', False,
                  'Whether to apply observation normalization.')
flags.DEFINE_integer('max_devices_per_host', None,
                     'Maximum number of devices to use per host. If None, '
                     'defaults to use as much as it can.')
flags.DEFINE_float('grad_updates_per_step', 1.0,
                   'How many gradient updates to run per one step in the '
                   'environment.')
flags.DEFINE_integer('num_save_html', 10, 'Number of Videos.')


def main(unused_argv):
  # save dir
  output_dir = os.path.join(
    FLAGS.logdir,
    f'bc_transformer_zs_{FLAGS.task_name}/obs_{FLAGS.obs_config}_arch_{FLAGS.architecture_config}_ntraj{FLAGS.num_episodes}_'+FLAGS.identifier+f'/seed_{FLAGS.seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
  print(f'Saving outputs to {output_dir}')
  os.makedirs(output_dir, exist_ok=True)

  obs_config = obs_config_dict[FLAGS.obs_config]
  dataset_config = ZERO_SHOT_TASK_CONFIG[FLAGS.task_name]
  envs_list = list(dataset_config['all_envs'].keys())
  train_envs_list = list(dataset_config['train_envs'].keys())
  test_envs_list = list(dataset_config['test_envs'].keys())
  local_state_size = obs_size_dict[FLAGS.obs_config]
  architecture_fn = ARCHITECTURE_CONFIG[FLAGS.architecture_config]

  if ('handsup2' in FLAGS.task_name) and ('ant' in FLAGS.task_name):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['ant_handsup2']
  elif ('handsup2' in FLAGS.task_name) and ('centipede' in FLAGS.task_name):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['centipede_handsup2']
  elif ('handsup' in FLAGS.task_name) and ('unimal' in FLAGS.task_name):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['unimal_handsup']
  elif 'handsup' in FLAGS.task_name:
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['handsup']

  train_job_params = {
    'action_repeat': FLAGS.action_repeat,
    'batch_size': FLAGS.batch_size,
    'episode_length': FLAGS.episode_length,
    'grad_updates_per_step': FLAGS.grad_updates_per_step,
    'learning_rate': FLAGS.learning_rate,
    'local_state_size': local_state_size,
    'log_frequency': FLAGS.eval_frequency,
    'normalize_observations': FLAGS.normalize_observations,
    'num_timesteps': FLAGS.total_grad_steps,
    'max_devices_per_host': FLAGS.max_devices_per_host,
    'seed': FLAGS.seed}

  config = copy.deepcopy(train_job_params)
  config['task_name'] = FLAGS.task_name
  config['obs_config'] = FLAGS.obs_config
  config['num_episodes'] = FLAGS.num_episodes
  config['architecture_config'] = FLAGS.architecture_config
  pprint.pprint(config)

  local_device_count = jax.local_device_count()
  print(f'number of devices: {local_device_count}')

  # logging
  logger_utils.save_config(
      f'{output_dir}/config.txt', config, verbose=True)
  tab = logger_utils.Tabulator(
      output_path=f'{output_dir}/training_curves.csv', append=False)

  times = [datetime.now()]
  plotpatterns = []

  progress, _, _, _ = experiments.get_progress_fn(
      plotpatterns,
      times,
      tab=tab,
      max_ncols=5,
      xlim=[0, train_job_params['num_timesteps']],
      post_plot_fn=functools.partial(plt.savefig, f'{output_dir}/progress.png'))

  key_sample = jax.random.PRNGKey(train_job_params['seed'] + 1234)
  env_fns = []
  obs_size_list = []
  num_limb_list = []
  dataset_paths_list = []
  for env_name in envs_list:
    observer = GraphObserver(name=FLAGS.obs_config, **obs_config)
    # create env
    env_fn = composer.create_fn(env_name=env_name, observer=observer, observer2=observer)
    sample_env = env_fn(
      action_repeat=train_job_params['action_repeat'],
      episode_length=train_job_params['episode_length'])
    first_state = sample_env.reset(key_sample)
    obs_size = first_state.obs.shape[0]
    num_limb = sample_env.num_node
    if env_name in test_envs_list:
      env_fns.append(env_fn)
    assert dataset_config['all_envs'][env_name][FLAGS.obs_config]['observation_size'] == obs_size,  (env_name, f'Config: {dataset_config["all_envs"][env_name][FLAGS.obs_config]["observation_size"]}, Provided: {obs_size}')
    assert dataset_config['all_envs'][env_name][FLAGS.obs_config]['num_limb'] == num_limb, (env_name, f'Config: {dataset_config["all_envs"][env_name][FLAGS.obs_config]["num_limb"]}, Provided: {num_limb}')
    assert dataset_config['all_envs'][env_name][FLAGS.obs_config]['action_size'] == sample_env.action_size, (env_name, f'Config: {dataset_config["all_envs"][env_name][FLAGS.obs_config]["action_size"]}, Provided: {sample_env.action_size}')
    obs_size_list.append(obs_size)
    num_limb_list.append(num_limb)
    if env_name in train_envs_list:
      dataset_paths_list.append(dataset_config['all_envs'][env_name][FLAGS.obs_config]['dataset_path'])
  max_obs_size = max(obs_size_list)
  max_num_limb = max(num_limb_list)

  replay_buffer_data = []
  src_mask_data = []
  for dataset_path, env_name in zip(dataset_paths_list, train_envs_list):
    assert env_name in dataset_path, env_name
    with open(dataset_path, 'rb') as f:
      data = pickle.load(f)
    obs_size = dataset_config['train_envs'][env_name][FLAGS.obs_config]['observation_size']
    num_limb = dataset_config['train_envs'][env_name][FLAGS.obs_config]['num_limb']
    trans_dim = obs_size * 2 + num_limb * 2 + 3
    assert trans_dim == data.shape[-1], (env_name, trans_dim, data.shape)
    if local_device_count == 4:
      num_steps = 979  # magic number (num of steps per env at data-collection)
    else:
      num_steps = 1958  # magic number (num of steps per env at data-collection)
    sliced_data = data.reshape(local_device_count, -1, num_steps, trans_dim)[:, :FLAGS.num_episodes]

    # for profile
    num_dones = []
    _data = sliced_data.reshape(-1, num_steps, trans_dim)
    for idx in range(FLAGS.num_episodes):
      num_dones.append(len(jnp.where(_data[idx, :, obs_size*2+num_limb*2+1]==0)[0]))
    print(f'{env_name} | Number of Episode (approximate): {sum(num_dones)}')

    if obs_size < max_obs_size:
      assert num_limb < max_num_limb
      zero_padding_obs = jnp.zeros(
        (local_device_count, FLAGS.num_episodes, num_steps, max_obs_size - obs_size))
      zero_padding_limb = jnp.zeros(
        (local_device_count, FLAGS.num_episodes, num_steps, max_num_limb - num_limb))
      pad_sliced_data = jnp.concatenate(
        [
          sliced_data[:,:,:,:obs_size],
          zero_padding_obs,  # obs
          sliced_data[:,:,:, obs_size:obs_size*2],
          zero_padding_obs,  # next_obs
          sliced_data[:,:,:, obs_size*2:obs_size*2+num_limb],
          zero_padding_limb,  # action
          sliced_data[:,:,:, obs_size*2+num_limb:obs_size*2+num_limb*2],
          zero_padding_limb,  # limb_mask
          sliced_data[:,:,:, obs_size*2+num_limb*2:trans_dim],  # r, done, term
          ],
        axis=-1)
    else:
        pad_sliced_data = sliced_data
    assert max_obs_size * 2 + max_num_limb * 2 + 3 == pad_sliced_data.shape[-1], pad_sliced_data.shape
    src_mask = jnp.zeros(
      (pad_sliced_data.shape[0], pad_sliced_data.shape[1], num_steps, max_num_limb, max_num_limb)).at[jnp.index_exp[:, :, :, :num_limb, :num_limb]].set(jnp.ones((num_limb, num_limb)))

    replay_buffer_data.append(pad_sliced_data)
    src_mask_data.append(src_mask)
  if len(dataset_paths_list) > 1:
    replay_buffer_data = jnp.concatenate(replay_buffer_data, axis=1)
    src_mask_data = jnp.concatenate(src_mask_data, axis=1)
  else:
    replay_buffer_data = pad_sliced_data
    src_mask_data = src_mask

  inference_fn, params, _ = bc_transformer.train(
      architecture_fn=architecture_fn,
      environment_fns=env_fns,
      envs_list=test_envs_list,
      max_num_limb=max_num_limb,
      max_obs_size=max_obs_size,
      progress_fn=progress,
      replay_buffer_data=replay_buffer_data,
      src_mask_data=src_mask_data,
      **train_job_params)

  # Save to flax serialized checkpoint.
  path = os.path.join(output_dir, 'policy.pkl')
  model.save_params(path, params)

  # output an episode trajectory
  for env_fn, env_name in zip(env_fns, test_envs_list):
    env = env_fn(
      action_repeat=train_job_params['action_repeat'],
      episode_length=train_job_params['episode_length'])
    action_size = dataset_config['test_envs'][env_name][FLAGS.obs_config]['action_size']
    obs_size = dataset_config['test_envs'][env_name][FLAGS.obs_config]['observation_size']
    num_limb = dataset_config['test_envs'][env_name][FLAGS.obs_config]['num_limb']
    _, _, key_init = jax.random.split(jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)[1], 3)
    for i in range(FLAGS.num_save_html):
      key = jax.random.PRNGKey(FLAGS.seed + 666)
      qps = []
      rs = []
      jit_inference_fn = jax.jit(inference_fn)
      jit_step_fn = jax.jit(env.step)
      state = env.reset(key_init)
      while not state.done:
        key, key_sample = jax.random.split(key)
        qps.append(state.qp)
        if obs_size < max_obs_size:
          zero_padding_obs = jnp.zeros((max_obs_size - obs_size,))
          obs = jnp.concatenate([state.obs, zero_padding_obs], axis=-1)
          src_mask = jnp.zeros((max_num_limb, max_num_limb)).at[jnp.index_exp[:num_limb, :num_limb]].set(jnp.ones((num_limb, num_limb)))
        else:
          obs = state.obs
          src_mask = None
        act = jit_inference_fn(params, obs, src_mask, key_sample)[1:1+action_size]
        state = jit_step_fn(state, act)
        rs.append(state.reward)
      avg_eval_len = len(rs)
      avg_eval_reward = jnp.sum(jnp.array(rs))
      print(f'{env_name} episode {i} len: {avg_eval_len}, reward: {avg_eval_reward}')

      html_path = os.path.join(output_dir, f'trajectory_{env_name}_{i}.html')
      html.save_html(html_path, env.sys, qps)

      _, key_init = jax.random.split(key_init, 2)


if __name__ == '__main__':
  app.run(main)
