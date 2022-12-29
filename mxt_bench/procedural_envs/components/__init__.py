import copy
import importlib
from typing import Any, Dict

DEFAULT_REGISTER_COMPONENTS = (
    'ant',
    'broken_ant',
    'centipede',
    'worm',
    'claw',
    'ground',
    'ball',
    'box',)

COMPONENT_MAPPING = {}


def list_components():
  return sorted(COMPONENT_MAPPING)


def exists(component: str):
  return component in COMPONENT_MAPPING


def register_component(component: str,
                       component_specs: Any = None,
                       load_path: str = None,
                       override: bool = False):
  """Register component library."""
  global COMPONENT_MAPPING
  if not override and component in COMPONENT_MAPPING:
    return COMPONENT_MAPPING[component]
  if component_specs is None:
    if not load_path:
      if '.' not in component:
        load_path = f'procedural_envs.components.{component}'
      else:
        load_path = component
    component_lib = importlib.import_module(load_path)
  COMPONENT_MAPPING[component] = component_lib
  return component_lib


def register_default_components():
  """Register all default components."""
  for component in DEFAULT_REGISTER_COMPONENTS:
    register_component(component)


def load_component(component: str,
                   component_specs: Dict[str, Any] = None,
                   component_params: Dict[str, Any] = None,
                   **override_specs) -> Dict[str, Any]:
  """Load component config information."""
  if isinstance(component, str):
    # if string, load a library under composer/components
    component_lib = register_component(
        component=component, component_specs=component_specs)
    specs_fn = component_lib.get_specs
  else:
    # else, assume it's a custom get_specs()
    specs_fn = component
  default_specs = specs_fn(**(component_params or {}))
  default_specs = copy.deepcopy(default_specs)
  default_specs.update(override_specs)
  return default_specs
