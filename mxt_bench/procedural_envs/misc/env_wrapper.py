import flax
import jax

from brax.envs import State
from brax.envs import env as brax_env
from brax import jumpy as jp

from typing import Dict


@flax.struct.dataclass
class CustomState(State):
  """Environment state for training and inference."""
  obs2: jp.ndarray = None


@flax.struct.dataclass
class GoalEvalMetrics:
  current_episode_metrics: Dict[str, jp.ndarray]
  completed_episodes_metrics: Dict[str, jp.ndarray]
  completed_episodes: jp.ndarray
  completed_episodes_steps: jp.ndarray
  success_episodes: jp.ndarray  # added
  final_distance: jp.ndarray  # added


class GoalEvalWrapper(brax_env.Wrapper):
  """Brax env with goal-based eval metrics."""

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    reset_state = self.env.reset(rng)
    reset_state.metrics['reward'] = reset_state.reward
    eval_metrics = GoalEvalMetrics(
        current_episode_metrics=jax.tree_map(jp.zeros_like,
                                             reset_state.metrics),
        completed_episodes_metrics=jax.tree_map(
            lambda x: jp.zeros_like(jp.sum(x)), reset_state.metrics),
        completed_episodes=jp.zeros(()),
        completed_episodes_steps=jp.zeros(()),
        success_episodes=jp.zeros(()),
        final_distance=jp.zeros(()))
    reset_state.info['eval_metrics'] = eval_metrics
    return reset_state

  def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
    state_metrics = state.info['eval_metrics']
    if not isinstance(state_metrics, GoalEvalMetrics):
      raise ValueError(
          f'Incorrect type for state_metrics: {type(state_metrics)}')
    del state.info['eval_metrics']
    nstate = self.env.step(state, action)
    nstate.metrics['reward'] = nstate.reward
    # steps stores the highest step reached when done = True, and then
    # the next steps becomes action_repeat
    completed_episodes_steps = state_metrics.completed_episodes_steps + jp.sum(
        nstate.info['steps'] * nstate.done)
    current_episode_metrics = jax.tree_multimap(
        lambda a, b: a + b, state_metrics.current_episode_metrics,
        nstate.metrics)
    completed_episodes = state_metrics.completed_episodes + jp.sum(nstate.done)
    # additional metrics
    success_episodes = state_metrics.success_episodes + jp.sum(nstate.done * (1.0 - nstate.info['truncation']))
    final_distance = state_metrics.final_distance + jp.sum(nstate.done * nstate.metrics['distance'])

    completed_episodes_metrics = jax.tree_multimap(
        lambda a, b: a + jp.sum(b * nstate.done),
        state_metrics.completed_episodes_metrics, current_episode_metrics)
    current_episode_metrics = jax.tree_multimap(
        lambda a, b: a * (1 - nstate.done) + b * nstate.done,
        current_episode_metrics, nstate.metrics)

    eval_metrics = GoalEvalMetrics(
        current_episode_metrics=current_episode_metrics,
        completed_episodes_metrics=completed_episodes_metrics,
        completed_episodes=completed_episodes,
        completed_episodes_steps=completed_episodes_steps,
        success_episodes=success_episodes,
        final_distance=final_distance,)
    nstate.info['eval_metrics'] = eval_metrics
    return nstate
