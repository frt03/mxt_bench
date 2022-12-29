"""component_lib: Ball."""

DEFAULT_OBSERVERS = ()

def generate_ball_config(radius, frozen, mass=1., name="Ball"):
  """Generate info for ball"""
  base_config = f"""
    bodies {{
      name: "{name}"
      colliders {{
        sphere {{
          radius: {radius}
        }}
      }}
      inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
      mass: {mass}
    }}
    """
  collides = tuple() if frozen else (name,)
  return base_config, collides


def get_specs(
    radius: float = 1.0,
    frozen: bool = False,
    mass: float = 1.,
    name: str = "Ball"):
  message_str, collides = generate_ball_config(radius, frozen, mass, name)
  return dict(
      message_str=message_str,
      frozen=frozen,
      collides=collides,
      observers=(),
      root=name,
      term_fn=None)
