"""Composer utility functions."""
import copy
from typing import Dict, Any

BASIC_DESC_KEYS = ('global_options', 'components', 'edges')


def merge_desc(desc1: Dict[str, Any], desc2: Dict[str, Any], override=False):
  """Merge desc2 to desc1.

  This will recursively traverse dictionaries to merge desc2 into desc1.
  If override, non-dictionary value will be overwritten.
  If not, tuples and lists are extended by defaults, while other types will
    error.

  Args:
    desc1: the desc that will be updated
    desc2: the update info to desc1
    override: a bool, override values or extend values
  """
  assert isinstance(desc1, dict), desc1
  assert isinstance(desc2, dict), desc2
  for k, v in desc2.items():
    if k not in desc1:
      desc1[k] = v
    else:
      v1 = desc1[k]
      assert type(v1) is type(v), f'type mismatch {k}: {v1} {v}'
      if isinstance(v, dict):
        merge_desc(desc1[k], v)
      elif override:
        desc1[k] = v
      elif isinstance(v, (tuple, list)):
        desc1[k] += v
      else:
        raise NotImplementedError(f'invalid merge {k}: {v1} {v}')


def edit_desc(env_desc: Dict[str, Any], desc_edits: Dict[str, Any]):
  """Edit desc dictionary."""
  env_desc = copy.deepcopy(env_desc)
  # add basic option types
  for key in BASIC_DESC_KEYS:
    env_desc[key] = env_desc.get(key, {})
  for key_str, value in desc_edits.items():
    # `key_str` is in a form '{key1}.{key2}.{key3}'
    #   for indexing env_desc[key1][key2][key3]
    keys = key_str.split('.')
    d = env_desc
    for _, key in enumerate(keys[:-1]):
      assert key in d, f'{key} not in {list(d.keys())}'
      d = d[key]
    d[keys[-1]] = value
  return env_desc
