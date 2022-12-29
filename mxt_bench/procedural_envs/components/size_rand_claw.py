"""component_lib: Claw."""
import numpy as np
import itertools

DEFAULT_LEG_LENGTH = 0.4
DEFAULT_RADIUS = 0.08

def generate_claw_config_with_n_legs(n: int, size_scales: list):
  def template_leg(theta, ind, size_scale):
    radius = DEFAULT_RADIUS * size_scale
    leg_length = DEFAULT_LEG_LENGTH * size_scale

    tmp = f"""
      bodies {{
        name: "Aux 1_{str(ind)}"
        colliders {{
          rotation {{ x: 90 y: -90 }}
          capsule {{
            radius: {radius}
            length: {leg_length}
          }}
        }}
        inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
        mass: 1
      }}
      bodies {{
        name: "Aux 2_{str(ind)}"
        colliders {{
          rotation {{ x: 90 y: -90 }}
          capsule {{
            radius: {radius}
            length: {leg_length}
          }}
        }}
        inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
        mass: 1
      }}
      bodies {{
      name: "$ Body 4_{str(ind)}"
      colliders {{
        rotation {{ x: 90 y: -90 }}
        capsule {{
          radius: {radius}
          length: {leg_length}
          end: -1
        }}
      }}
      inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
      mass: 1
    }}
    joints {{
      name: "torso_Aux 1_{str(ind)}"
      parent_offset {{ x: {((leg_length/2.)+.08)*np.cos(theta)} y: {((leg_length/2.)+.08)*np.sin(theta)} }}
      child_offset {{ }}
      parent: "torso"
      child: "Aux 1_{str(ind)}"
      stiffness: 5000.0
      angular_damping: 35
      angle_limit {{ min: -30.0 max: 30.0 }}
      angle_limit {{ min: -30.0 max: 30.0 }}
      rotation {{ y: -90 }}
      reference_rotation {{ z: {theta*180/np.pi} }}
    }}
    joints {{
      name: "Aux 1_Aux 2_{str(ind)}"
      parent_offset {{ x: {leg_length/2. - radius}  }}
      child_offset {{ x:{-leg_length/2. + radius}  }}
      parent: "Aux 1_{str(ind)}"
      child: "Aux 2_{str(ind)}"
      stiffness: 5000.0
      angular_damping: 35
      rotation: {{ z: 90 }}
      angle_limit {{
        min: 10.0
        max: 80.0
      }}
    }}
    joints {{
      name: "Aux 2_$ Body 4_{str(ind)}"
      parent_offset {{ x: {leg_length/2. - radius}  }}
      child_offset {{ x:{-leg_length/2. + radius}  }}
      parent: "Aux 2_{str(ind)}"
      child: "$ Body 4_{str(ind)}"
      stiffness: 5000.0
      angular_damping: 35
      rotation: {{ z: 90 }}
      angle_limit {{
        min: 30.0
        max: 60.0
      }}
    }}
    actuators {{
      name: "torso_Aux 1_{str(ind)}"
      joint: "torso_Aux 1_{str(ind)}"
      strength: 500.0
      torque {{}}
    }}
    actuators {{
      name: "Aux 1_Aux 2_{str(ind)}"
      joint: "Aux 1_Aux 2_{str(ind)}"
      strength: 500.0
      torque {{}}
    }}
    actuators {{
      name: "Aux 2_$ Body 4_{str(ind)}"
      joint: "Aux 2_$ Body 4_{str(ind)}"
      strength: 500.0
      torque {{}}
    }}
    """
    collides = (f"Aux 1_{str(ind)}", f"Aux 2_{str(ind)}", f"$ Body 4_{str(ind)}")
    return tmp, collides

  base_config = f"""
    bodies {{
      name: "torso"
      colliders {{
        capsule {{
          radius: 0.25
          length: 0.5
          end: 1
        }}
      }}
      inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
      mass: 10
    }}
  """
  collides = ('torso',)
  size_iterator = itertools.cycle(size_scales)
  for i in range(n):
    size_scale = next(size_iterator)
    config_i, collides_i = template_leg((1. * i / n) * 2 * np.pi, i, size_scale)
    base_config += config_i
    collides += collides_i
  return base_config, collides


def get_specs(num_legs: int = 4, size_scales: list = []):
  assert len(size_scales) > 0
  message_str, collides = generate_claw_config_with_n_legs(num_legs, size_scales)
  return dict(
      message_str=message_str,
      collides=collides,
      root='torso',
      term_fn=None)
