import functools
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from absl import logging
from brax import envs
from brax.io import model
from brax.training import normalization
from brax.training import pmap
from brax.training.types import Params
from brax.training.types import PRNGKey

import flax
import jax
import jax.numpy as jnp
import numpy as onp
import optax

Metrics = Mapping[str, jnp.ndarray]


@flax.struct.dataclass
class Transition:
  """Contains data for one environment step."""
  o_tm1: jnp.ndarray
  a_tm1: jnp.ndarray
  r_t: jnp.ndarray
  o_t: jnp.ndarray
  d_t: jnp.ndarray  # discount (1-done)
  truncation_t: jnp.ndarray
  limb_mask: jnp.ndarray
  src_mask: jnp.ndarray


# The rewarder allows to change the reward of before the learner trains.
RewarderState = Any
RewarderInit = Callable[[int, PRNGKey], RewarderState]
ComputeReward = Callable[[RewarderState, Transition, PRNGKey],
                         Tuple[RewarderState, jnp.ndarray, Metrics]]
Rewarder = Tuple[RewarderInit, ComputeReward]


@flax.struct.dataclass
class ReplayBuffer:
  """Contains data related to a replay buffer."""
  data: jnp.ndarray
  src_mask: jnp.ndarray


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  policy_params: Params
  key: PRNGKey
  actor_steps: jnp.ndarray
  normalizer_params: Params
  # The is passed to the rewarder to update the reward.
  rewarder_state: Any


def train(
    environment_fns: List[Callable[..., envs.Env]],
    num_timesteps,
    episode_length: int,
    replay_buffer_data: jnp.ndarray,
    src_mask_data: jnp.ndarray,
    max_obs_size: int,
    max_num_limb: int,
    envs_list: Any,
    architecture_fn: Callable,
    local_state_size: int = 19,
    gradient_clipping: float = 0.1,
    action_repeat: int = 1,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    seed: int = 0,
    batch_size: int = 256,
    log_frequency: int = 10000,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = None,
    grad_updates_per_step: float = 1,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    # The rewarder is an init function and a compute_reward function.
    # It is used to change the reward before the learner trains on it.
    make_rewarder: Optional[Callable[[], Rewarder]] = None,
    checkpoint_logdir: Optional[str] = None):
  # jax.config.update('jax_log_compiles', True)

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)

  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d', jax.device_count(), process_count,
      process_id, local_device_count, local_devices_to_use)

  num_updates = int(grad_updates_per_step)
  batch_size_per_device = batch_size // local_devices_to_use

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  key_models, key_rewarder = jax.random.split(global_key, 2)
  local_key, key_env, key_eval = jax.random.split(local_key, 3)

  policy_model, _ = architecture_fn(
    obs_size=local_state_size, action_size=1, max_num_limb=max_num_limb)

  policy_optimizer = optax.chain(
    optax.clip(gradient_clipping),
    optax.adam(learning_rate=learning_rate),
  )
  key_policy, key_q = jax.random.split(key_models)
  policy_params = policy_model.init({'params': key_policy, 'dropout': key_q})
  policy_optimizer_state = policy_optimizer.init(policy_params)

  # count the number of parameters
  param_count = sum(x.size for x in jax.tree_leaves(policy_params))
  logging.info(f'num_policy_param: {param_count}')

  policy_optimizer_state, policy_params = pmap.bcast_local_devices(
      (policy_optimizer_state, policy_params), local_devices_to_use)

  normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
      normalization.create_observation_normalizer(
          max_obs_size,
          normalize_observations,
          pmap_to_devices=local_devices_to_use))

  if make_rewarder is not None:
    init, compute_reward = make_rewarder()
    rewarder_state = init(max_obs_size, key_rewarder)
    rewarder_state = pmap.bcast_local_devices(rewarder_state,
                                              local_devices_to_use)
  else:
    rewarder_state = None
    compute_reward = None

  key_debug = jax.random.PRNGKey(seed + 666)

  # define eval
  def do_one_step_eval(carry, unused_target_t, env_carry):
    state, policy_params, normalizer_params, key = carry
    _obs_size, _num_limb, _action_size, _eval_step_fn = env_carry
    key, key_sample = jax.random.split(key)
    pad_obs = jnp.concatenate(
        [state.obs, jnp.zeros((num_eval_envs, max_obs_size - _obs_size))],
        axis=-1)
    obs = obs_normalizer_apply_fn(normalizer_params, pad_obs)
    obs = obs.reshape(num_eval_envs, -1, local_state_size)
    mask = jnp.zeros(
      (max_num_limb, max_num_limb)
      ).at[jnp.index_exp[:_num_limb, :_num_limb]].set(jnp.ones((_num_limb, _num_limb)))
    actions, _ = policy_model.apply(policy_params, obs, mask, rngs={'dropout': key_sample})
    actions = jnp.tanh(actions)[:, 1:1+_action_size]
    nstate = _eval_step_fn(state, actions)
    return (nstate, policy_params, normalizer_params, key), ()

  def run_eval(state, key, policy_params,
               normalizer_params, do_one_step_eval_wrapper) -> Tuple[envs.State, PRNGKey]:
    policy_params, normalizer_params = jax.tree_map(
        lambda x: x[0], (policy_params, normalizer_params))
    (state, _, _, key), _ = jax.lax.scan(
        do_one_step_eval_wrapper, (state, policy_params, normalizer_params, key), (),
        length=episode_length // action_repeat)
    return state, key

  def actor_loss(policy_params: Params,
                 transitions: Transition, key: PRNGKey) -> jnp.ndarray:
    _o_tm1 = transitions.o_tm1.reshape(batch_size_per_device, -1, local_state_size)
    _a_tm1 = transitions.a_tm1.reshape(batch_size_per_device, -1, 1)
    mask = transitions.src_mask.reshape(batch_size_per_device, -1, max_num_limb, max_num_limb)
    action, _ = policy_model.apply(policy_params, _o_tm1, mask, rngs={'dropout': key})
    action = jnp.tanh(action).reshape(batch_size_per_device, -1, 1)
    limb_mask = transitions.limb_mask.reshape(batch_size_per_device, -1, 1)
    actor_loss = 0.5 * jnp.mean(jnp.square(action - _a_tm1) * limb_mask)
    return jnp.mean(actor_loss)

  actor_grad = jax.jit(jax.value_and_grad(actor_loss))

  @jax.jit
  def update_step(
      state: TrainingState,
      target_t: Tuple[jnp.ndarray, jnp.ndarray],
  ) -> Tuple[TrainingState, bool, Dict[str, jnp.ndarray]]:
    transitions, src_mask = target_t
    normalized_transitions = Transition(
        o_tm1=obs_normalizer_apply_fn(state.normalizer_params,
                                      transitions[:, :max_obs_size]),
        o_t=obs_normalizer_apply_fn(state.normalizer_params,
                                    transitions[:, max_obs_size:max_obs_size*2]),
        a_tm1=transitions[:, max_obs_size*2:max_obs_size*2+max_num_limb],
        limb_mask=transitions[:, max_obs_size*2+max_num_limb:max_obs_size*2+max_num_limb*2],
        r_t=transitions[:, max_obs_size*2+max_num_limb*2],
        d_t=transitions[:, max_obs_size*2+max_num_limb*2 + 1],
        truncation_t=transitions[:, max_obs_size*2+max_num_limb*2],
        src_mask=src_mask,
      )

    (key, _, _, key_actor, key_rewarder) = jax.random.split(state.key, 5)

    if compute_reward is not None:
      new_rewarder_state, rewards, rewarder_metrics = compute_reward(
          state.rewarder_state, normalized_transitions, key_rewarder)
      # Assertion prevents building errors.
      assert hasattr(normalized_transitions, 'replace')
      normalized_transitions = normalized_transitions.replace(r_t=rewards)
    else:
      new_rewarder_state = state.rewarder_state
      rewarder_metrics = {}

    actor_loss, actor_grads = actor_grad(state.policy_params,
                                         normalized_transitions, key_actor)
    actor_grads = jax.lax.pmean(actor_grads, axis_name='i')
    policy_params_update, policy_optimizer_state = policy_optimizer.update(
        actor_grads, state.policy_optimizer_state)
    policy_params = optax.apply_updates(state.policy_params, policy_params_update)

    metrics = {'actor_loss': actor_loss, **rewarder_metrics}

    new_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        key=key,
        actor_steps=state.actor_steps + 1,
        normalizer_params=state.normalizer_params,
        rewarder_state=new_rewarder_state)
    return new_state, metrics

  def sample_data(training_state, replay_buffer):
    key1, key2 = jax.random.split(training_state.key)
    idx = jax.random.randint(
        key2, (int(batch_size_per_device*num_updates),),
        minval=0,
        maxval=replay_buffer.data.shape[0])
    transitions = jnp.take(replay_buffer.data, idx, axis=0, mode='clip')
    transitions = jnp.reshape(transitions,
                              [num_updates, -1] + list(transitions.shape[1:]))
    src_mask = jnp.take(replay_buffer.src_mask, idx, axis=0, mode='clip')
    src_mask = jnp.reshape(src_mask,
                           [num_updates, -1] + list(src_mask.shape[1:]))
    # update normalization function
    normalizer_params = obs_normalizer_update_fn(
        training_state.normalizer_params, transitions[:, :max_obs_size])
    training_state = training_state.replace(
        key=key1, normalizer_params=normalizer_params)
    return training_state, (transitions, src_mask)

  def run_one_epoch(carry, unused_t):
    training_state, replay_buffer = carry

    training_state, (transitions, src_mask) = sample_data(training_state, replay_buffer)
    training_state, metrics = jax.lax.scan(
        update_step, training_state, (transitions, src_mask), length=num_updates)
    return (training_state, replay_buffer), metrics

  def run_training(training_state, replay_buffer):
    synchro = pmap.is_synchronized(
        training_state.replace(key=jax.random.PRNGKey(0)), axis_name='i')
    (training_state, replay_buffer), metrics = jax.lax.scan(
        run_one_epoch, (training_state, replay_buffer), (),
        length=log_frequency)
    metrics = jax.tree_map(jnp.mean, metrics)
    return training_state, replay_buffer, metrics, synchro

  run_training = jax.pmap(run_training, axis_name='i')

  training_state = TrainingState(
      policy_optimizer_state=policy_optimizer_state,
      policy_params=policy_params,
      key=jnp.stack(jax.random.split(local_key, local_devices_to_use)),
      actor_steps=jnp.zeros((local_devices_to_use,)),
      normalizer_params=normalizer_params,
      rewarder_state=rewarder_state)

  training_walltime = 0
  eval_walltime = 0
  sps = 0
  eval_sps = 0
  current_step = 0
  training_metrics = {}
  metrics = {}

  while True:
    logging.info('step %s', current_step)
    t = time.time()

    if process_id == 0:
      eval_logs = {}
      for env_name, environment_fn in zip(envs_list, environment_fns):
        core_eval_env = environment_fn(
          action_repeat=action_repeat,
          batch_size=num_eval_envs,
          episode_length=episode_length,
          eval_metrics=True)
        eval_step_fn = jax.jit(core_eval_env.step)
        eval_first_state = jax.jit(core_eval_env.reset)(key_eval)
        _, obs_size = eval_first_state.obs.shape
        num_limb = core_eval_env.num_node
        action_size = core_eval_env.action_size

        eval_wrapper = functools.partial(
            do_one_step_eval,
            env_carry=(obs_size, num_limb, action_size, eval_step_fn))
        eval_fn = jax.jit(functools.partial(run_eval, do_one_step_eval_wrapper=eval_wrapper))

        eval_state, key_debug = eval_fn(eval_first_state, key_debug,
                                        training_state.policy_params,
                                        training_state.normalizer_params)
        eval_metrics = eval_state.info['eval_metrics']
        eval_metrics.completed_episodes.block_until_ready()
        eval_walltime += time.time() - t
        eval_sps = (
            episode_length * eval_first_state.reward.shape[0] /
            (time.time() - t))
        avg_episode_length = (
            eval_metrics.completed_episodes_steps / eval_metrics.completed_episodes)
        success_rate = eval_metrics.success_episodes / eval_metrics.completed_episodes
        avg_final_distance = eval_metrics.final_distance / eval_metrics.completed_episodes
        for name, value in eval_metrics.completed_episodes_metrics.items():
          metrics[f'eval/{env_name}/episode_{name}'] = value / eval_metrics.completed_episodes
          metrics[f'eval/{env_name}/completed_episodes'] = eval_metrics.completed_episodes
        metrics[f'speed/{env_name}/eval_walltime'] = eval_walltime
        metrics[f'speed/{env_name}/eval_sps'] = eval_sps
        metrics[f'eval/{env_name}/avg_episode_length'] = avg_episode_length
        metrics[f'eval/{env_name}/avg_final_distance'] = avg_final_distance
        metrics[f'eval/{env_name}/success_rate'] = success_rate
        episode_reward = metrics[f'eval/{env_name}/episode_reward']
        logging.info(f'{env_name}/episode_reward: {episode_reward}')
        logging.info(f'{env_name}/avg_final_distance: {avg_final_distance}')
        logging.info(f'{env_name}/avg_episode_length: {avg_episode_length}')
        logging.info(f'{env_name}/success_rate: {success_rate}')
        t = time.time()
      for name, value in training_metrics.items():
        metrics[f'training/{name}'] = onp.mean(value)
      metrics['speed/sps'] = sps
      metrics['speed/training_walltime'] = training_walltime
      metrics['training/actor_grad_updates'] = training_state.actor_steps[0]

      logging.info(metrics)
      if progress_fn:
        progress_fn(current_step, metrics, None)

      if checkpoint_logdir:
        # Save current policy.
        normalizer_params = jax.tree_map(lambda x: x[0],
                                         training_state.normalizer_params)
        policy_params = jax.tree_map(lambda x: x[0],
                                     training_state.policy_params)
        params = normalizer_params, policy_params
        filename = f'bc_transformer_{current_step}.pkl'
        path = os.path.join(checkpoint_logdir, filename)
        model.save_params(path, params)

    if current_step >= num_timesteps:
      break

    # Create an initialize the replay buffer.
    if current_step == 0:
      t = time.time()

      replay_buffer = ReplayBuffer(
          data=replay_buffer_data.reshape(
              replay_buffer_data.shape[0],
              replay_buffer_data.shape[1]*replay_buffer_data.shape[2],
              replay_buffer_data.shape[3]),
          src_mask=src_mask_data.reshape(
              src_mask_data.shape[0],
              src_mask_data.shape[1]*src_mask_data.shape[2],
              src_mask_data.shape[3],
              src_mask_data.shape[4]))

      training_walltime += time.time() - t

    t = time.time()
    # optimization
    training_state, replay_buffer, training_metrics, synchro = run_training(
        training_state, replay_buffer)
    assert synchro[0], (current_step, training_state)
    jax.tree_map(lambda x: x.block_until_ready(), training_metrics)
    sps = (num_updates / (time.time() - t))
    training_walltime += time.time() - t
    current_step += num_updates * log_frequency

  normalizer_params = jax.tree_map(lambda x: x[0],
                                   training_state.normalizer_params)
  policy_params = jax.tree_map(lambda x: x[0], training_state.policy_params)

  inference = make_inference_fn(
    observation_size=max_obs_size,
    local_state_size=local_state_size,
    max_num_limb=max_num_limb,
    normalize_observations=normalize_observations,
    architecture_fn=architecture_fn)
  params = normalizer_params, policy_params

  pmap.synchronize_hosts()
  return (inference, params, metrics)


def make_inference_fn(observation_size,
                      local_state_size,
                      max_num_limb,
                      normalize_observations,
                      architecture_fn):
  """Creates params and inference function for the TD3-based agent."""
  _, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
      observation_size, normalize_observations)
  policy_model, _ = architecture_fn(
    obs_size=local_state_size, action_size=1, max_num_limb=max_num_limb)
  num_limb = int(observation_size / local_state_size)
  assert num_limb == max_num_limb

  def inference_fn(params, obs, mask, key):
    normalizer_params, policy_params = params
    obs = obs_normalizer_apply_fn(normalizer_params, obs)
    obs = obs.reshape(1, num_limb, local_state_size)
    action, _ = policy_model.apply(policy_params, obs, mask, rngs={'dropout': key})
    return jnp.tanh(action).ravel()

  return inference_fn
