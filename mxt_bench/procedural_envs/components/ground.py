"""component_lib: Ground."""

SYSTEM_CONFIG = """
bodies {
  name: "Ground"
  colliders {
    plane {}
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
  frozen { all: true }
}
"""


def get_specs():
  return dict(
      message_str=SYSTEM_CONFIG,
      collides=('Ground',),
      root='Ground',
      term_fn=None,
      observers=())
