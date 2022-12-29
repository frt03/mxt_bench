"""component_lib: Box."""
from typing import Tuple

DEFAULT_OBSERVERS = ()

def generate_box_config(halfsize, frozen, mass=1., name="Box"):
  """Generate info for box"""
  base_config = f"""
    bodies {{
      name: "{name}"
      colliders {{
        box {{
          halfsize {{
            x: {halfsize[0]}
            y: {halfsize[1]}
            z: {halfsize[2]}
          }}
        }}
      }}
      inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
      mass: {mass}
    }}
    """
  collides = tuple() if frozen else (name,)
  return base_config, collides


def get_specs(halfsize: Tuple[float, float, float] = (1., 1., 0.75),
              frozen: bool = False,
              mass: float = 1.,
              name: str = "Box"):
  message_str, collides = generate_box_config(halfsize, frozen, mass, name)
  return dict(
      message_str=message_str,
      frozen=frozen,
      collides=collides,
      root=name,
      term_fn=None)
