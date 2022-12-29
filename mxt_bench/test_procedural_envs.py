import copy
import os
import jax

from absl import app
from absl import flags
from brax.io import html
from brax.experimental.braxlines.common import logger_utils
from datetime import datetime

from procedural_envs import composer
from procedural_envs.misc.observers import GraphObserver
from procedural_envs.tasks.observation_config import obs_config_dict

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'ant_reach_4', 'Name of environment to train.')
flags.DEFINE_string('obs_config', 'amorpheus', 'Name of observation config to train.')
flags.DEFINE_string('obs_config_test', 'mtg_v2_base_m', 'Name of observation config to test.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('logdir', '', 'Logdir.')
flags.DEFINE_integer('num_save_html', 5, 'Number of Videos.')


def main(args):
  env_name = FLAGS.env_name
  obs_config = obs_config_dict[FLAGS.obs_config]
  obs_config_test = obs_config_dict[FLAGS.obs_config_test]

  environment_params = {
      'env_name': FLAGS.env_name,
      'obs_config': FLAGS.obs_config,
      'obs_config_test': FLAGS.obs_config_test,
  }

  # save dir
  output_dir = os.path.join(
      FLAGS.logdir,
      f'test_procedural_envs_{FLAGS.env_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
  print(f'Saving outputs to {output_dir}')
  os.makedirs(output_dir, exist_ok=True)

  env_config = copy.deepcopy(environment_params)

  observer = GraphObserver(name=FLAGS.obs_config, **obs_config)
  observer_test = GraphObserver(name=FLAGS.obs_config_test, **obs_config_test)

  # create env
  env_fn = composer.create_fn(env_name=env_name, observer=observer, observer2=observer_test)

  env = env_fn()
  print(f'action_size: {env.action_size}')
  print(f'observation_size: {env.observation_size}')
  print(f'observation_size2: {env.observation_size2}')
  env_config['action_size'] = env.action_size
  env_config['observation_size'] = env.observation_size
  env_config['observation_size2'] = env.observation_size2
  # logging
  logger_utils.save_config(
      f'{output_dir}/obs_config.txt', env_config, verbose=True)

  jit_step_fn = jax.jit(env.step)
  jit_reset_fn = jax.jit(env.reset)

  _, _, _, key_init = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 4)

  for i in range(FLAGS.num_save_html):

    key = jax.random.PRNGKey(FLAGS.seed + 666)

    qps = []
    state = jit_reset_fn(key_init)

    while not state.done:
      key, key_sample = jax.random.split(key)
      qps.append(state.qp)
      act = jax.random.normal(key=key_sample, shape=(env.action_size,))
      state = jit_step_fn(state, act)

    html_path = os.path.join(output_dir, f'trajectory_{i}.html')
    html.save_html(html_path, env.sys, qps)

    _, key_init = jax.random.split(key_init, 2)


if __name__ == '__main__':
  app.run(main)
