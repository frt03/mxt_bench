import copy
import functools
import os
import pprint
import jax

from absl import app
from absl import flags
from brax.io import html
from brax.io import model
from brax.experimental.braxlines import experiments
from brax.experimental.braxlines.common import logger_utils
from datetime import datetime
import matplotlib.pyplot as plt

from algo import ppo_mlp
from procedural_envs import composer
from procedural_envs.misc.observers import GraphObserver, EXTRA_ROOT_NODE_DICT
from procedural_envs.tasks.observation_config import obs_config_dict

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'ant_reach_4', 'Name of environment to train.')
flags.DEFINE_string('obs_config', 'amorpheus', 'Name of observation config to train.')
flags.DEFINE_integer('total_env_steps', 100000000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('eval_frequency', 20, 'How many times to run an eval.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_envs', 2048, 'Number of envs to run in parallel.')
flags.DEFINE_integer('action_repeat', 1, 'Action repeat.')
flags.DEFINE_integer('batch_size', 1024, 'Batch size.')
flags.DEFINE_float('reward_scaling', 1.0, 'Reward scale.')
flags.DEFINE_integer('episode_length', 1000, 'Episode length.')
flags.DEFINE_float('entropy_cost', 1e-2, 'Entropy cost.')
flags.DEFINE_integer('unroll_length', 5, 'Unroll length.')
flags.DEFINE_float('discounting', 0.97, 'Discounting.')
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
flags.DEFINE_integer('num_minibatches', 32, 'Number')
flags.DEFINE_integer('num_update_epochs', 4,
                     'Number of times to reuse each transition for gradient '
                     'computation.')
flags.DEFINE_string('logdir', '', 'Logdir.')
flags.DEFINE_bool('normalize_observations', True,
                  'Whether to apply observation normalization.')
flags.DEFINE_integer('max_devices_per_host', None,
                     'Maximum number of devices to use per host. If None, '
                     'defaults to use as much as it can.')
flags.DEFINE_integer('num_save_html', 3, 'Number of Videos.')


def main(unused_argv):
  # save dir
  output_dir = os.path.join(
    FLAGS.logdir,
    f'ao_ppo_mlp_single_pro_{FLAGS.env}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
  print(f'Saving outputs to {output_dir}')
  os.makedirs(output_dir, exist_ok=True)

  environment_params = {
      'env_name': FLAGS.env,
      'obs_config': FLAGS.obs_config,
  }
  obs_config = obs_config_dict[FLAGS.obs_config]

  if ('handsup2' in FLAGS.env) and ('ant' in FLAGS.env):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['ant_handsup2']
  elif ('handsup2' in FLAGS.env) and ('centipede' in FLAGS.env):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['centipede_handsup2']
  elif ('handsup' in FLAGS.env) and ('unimal' in FLAGS.env):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['unimal_handsup']
  elif 'handsup' in FLAGS.env:
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['handsup']

  env_config = copy.deepcopy(environment_params)
  observer = GraphObserver(name=FLAGS.obs_config, **obs_config)
  # create env
  env_fn = composer.create_fn(env_name=FLAGS.env, observer=observer, observer2=observer)
  env = env_fn()
  print(f'action_size: {env.action_size}')
  print(f'observation_size: {env.observation_size}')
  print(f'observation_size2: {env.observation_size2}')
  print(f'num_node in observations: {env.num_node}')
  env_config['action_size'] = env.action_size
  env_config['observation_size'] = env.observation_size
  env_config['observation_size2'] = env.observation_size2
  # logging
  logger_utils.save_config(
      f'{output_dir}/obs_config.txt', env_config, verbose=True)

  train_job_params = {
    'action_repeat': FLAGS.action_repeat,
    'batch_size': FLAGS.batch_size,
    'checkpoint_logdir': output_dir,
    'discounting': FLAGS.discounting,
    'entropy_cost': FLAGS.entropy_cost,
    'episode_length': FLAGS.episode_length,
    'learning_rate': FLAGS.learning_rate,
    'log_frequency': FLAGS.eval_frequency,
    'local_state_size': env_config['observation_size'] // env.num_node,
    'normalize_observations': FLAGS.normalize_observations,
    'num_envs': FLAGS.num_envs,
    'num_minibatches': FLAGS.num_minibatches,
    'num_timesteps': FLAGS.total_env_steps,
    'num_update_epochs': FLAGS.num_update_epochs,
    'max_devices_per_host': FLAGS.max_devices_per_host,
    'reward_scaling': FLAGS.reward_scaling,
    'seed': FLAGS.seed,
    'unroll_length': FLAGS.unroll_length,
    'goal_env': env.metadata.goal_based_task}

  config = copy.deepcopy(train_job_params)
  config['env'] = FLAGS.env
  pprint.pprint(config)

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

  inference_fn, params, _ = ppo_mlp.train(
      environment_fn=env_fn,
      progress_fn=progress,
      **train_job_params)

  # Save to flax serialized checkpoint.
  filename = f'ao_ppo_mlp_single_pro_{FLAGS.env}_final.pkl'
  path = os.path.join(output_dir, filename)
  model.save_params(path, params)

  # output an episode trajectory
  _, _, _, key_init = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 4)
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
      act = jit_inference_fn(params, state.obs, key_sample)
      state = jit_step_fn(state, act)
      rs.append(state.reward)
    avg_eval_len = len(rs)
    avg_eval_reward = jax.numpy.sum(jax.numpy.array(rs))
    print(f'{FLAGS.env} episode {i} len: {avg_eval_len}, reward {avg_eval_reward}')

    html_path = os.path.join(output_dir, f'trajectory_{i}.html')
    html.save_html(html_path, env.sys, qps)

    _, key_init = jax.random.split(key_init, 2)


if __name__ == '__main__':
  app.run(main)
