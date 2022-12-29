"""component_lib: BrokenCentipede. Broken legs are pre-defined."""
import numpy as np


def generate_centipede_config_with_n_torso(n: int, broken_ids: tuple):
  assert n >= 2
  assert broken_ids[0] in (4, 5)
  """Generate info for n-torso centipede."""

  def template_torso(theta, ind):
    tmp = f"""
      bodies {{
        name: "torso_{str(ind)}"
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
      bodies {{
        name: "Aux 1_{str(ind)}"
        colliders {{
          rotation {{ x: 90 y: -90 }}
          capsule {{
            radius: 0.08
            length: 0.4428427219390869
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
            radius: 0.08
            length: 0.7256854176521301
            end: -1
          }}
        }}
        inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
        mass: 1
      }}
      joints {{
        name: "torso_{str(ind)}_Aux 1_{str(ind)}"
        parent_offset {{ x: {((0.4428427219390869/2.)+.08)*np.cos(theta)} y: {((0.4428427219390869/2.)+.08)*np.sin(theta)} }}
        child_offset {{ }}
        parent: "torso_{str(ind)}"
        child: "Aux 1_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -30.0 max: 30.0 }}
        rotation {{ y: -90 }}
        reference_rotation {{ z: {theta*180/np.pi} }}
      }}
      joints {{
        name: "Aux 1_{str(ind)}_$ Body 4_{str(ind)}"
        parent_offset {{ x: 0.14142136096954344  }}
        child_offset {{ x:-0.28284270882606505  }}
        parent: "Aux 1_{str(ind)}"
        child: "$ Body 4_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        rotation: {{ z: 90 }}
        angle_limit {{
          min: 30.0
          max: 70.0
        }}
      }}
      actuators {{
        name: "torso_{str(ind)}_Aux 1_{str(ind)}"
        joint: "torso_{str(ind)}_Aux 1_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      actuators {{
        name: "Aux 1_{str(ind)}_$ Body 4_{str(ind)}"
        joint: "Aux 1_{str(ind)}_$ Body 4_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      bodies {{
        name: "Aux 2_{str(ind)}"
        colliders {{
          rotation {{ x: 90 y: -90 }}
          capsule {{
            radius: 0.08
            length: 0.4428427219390869
          }}
        }}
        inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
        mass: 1
      }}
      bodies {{
        name: "$ Body 5_{str(ind)}"
        colliders {{
          rotation {{ x: 90 y: -90 }}
          capsule {{
            radius: 0.08
            length: 0.7256854176521301
            end: -1
          }}
        }}
        inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
        mass: 1
      }}
      joints {{
        name: "torso_{str(ind)}_Aux 2_{str(ind)}"
        parent_offset {{ x: {((0.4428427219390869/2.)+.08)*np.cos(-theta)} y: {((0.4428427219390869/2.)+.08)*np.sin(-theta)} }}
        child_offset {{ }}
        parent: "torso_{str(ind)}"
        child: "Aux 2_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -30.0 max: 30.0 }}
        rotation {{ y: -90 }}
        reference_rotation {{ z: {-theta*180/np.pi} }}
      }}
      joints {{
        name: "Aux 2_{str(ind)}_$ Body 5_{str(ind)}"
        parent_offset {{ x: 0.14142136096954344  }}
        child_offset {{ x:-0.28284270882606505  }}
        parent: "Aux 2_{str(ind)}"
        child: "$ Body 5_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        rotation: {{ z: 90 }}
        angle_limit {{
          min: 30.0
          max: 70.0
        }}
      }}
      actuators {{
        name: "torso_{str(ind)}_Aux 2_{str(ind)}"
        joint: "torso_{str(ind)}_Aux 2_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      actuators {{
        name: "Aux 2_{str(ind)}_$ Body 5_{str(ind)}"
        joint: "Aux 2_{str(ind)}_$ Body 5_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      """
    collides = (
      f'torso_{str(ind)}',
      f'Aux 1_{str(ind)}',
      f'$ Body 4_{str(ind)}',
      f'Aux 2_{str(ind)}',
      f'$ Body 5_{str(ind)}')

    return tmp, collides

  def broken_leg_4(theta, ind):
    tmp = f"""
      bodies {{
        name: "torso_{str(ind)}"
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
      bodies {{
        name: "Aux 1_{str(ind)}"
        colliders {{
          rotation {{ x: 90 y: -90 }}
          capsule {{
            radius: 0.08
            length: 0.4428427219390869
          }}
        }}
        inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
        mass: 1
      }}
      joints {{
        name: "torso_{str(ind)}_Aux 1_{str(ind)}"
        parent_offset {{ x: {((0.4428427219390869/2.)+.08)*np.cos(theta)} y: {((0.4428427219390869/2.)+.08)*np.sin(theta)} }}
        child_offset {{ }}
        parent: "torso_{str(ind)}"
        child: "Aux 1_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -30.0 max: 30.0 }}
        rotation {{ y: -90 }}
        reference_rotation {{ z: {theta*180/np.pi} }}
      }}
      actuators {{
        name: "torso_{str(ind)}_Aux 1_{str(ind)}"
        joint: "torso_{str(ind)}_Aux 1_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      bodies {{
        name: "Aux 2_{str(ind)}"
        colliders {{
          rotation {{ x: 90 y: -90 }}
          capsule {{
            radius: 0.08
            length: 0.4428427219390869
          }}
        }}
        inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
        mass: 1
      }}
      bodies {{
        name: "$ Body 5_{str(ind)}"
        colliders {{
          rotation {{ x: 90 y: -90 }}
          capsule {{
            radius: 0.08
            length: 0.7256854176521301
            end: -1
          }}
        }}
        inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
        mass: 1
      }}
      joints {{
        name: "torso_{str(ind)}_Aux 2_{str(ind)}"
        parent_offset {{ x: {((0.4428427219390869/2.)+.08)*np.cos(-theta)} y: {((0.4428427219390869/2.)+.08)*np.sin(-theta)} }}
        child_offset {{ }}
        parent: "torso_{str(ind)}"
        child: "Aux 2_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -30.0 max: 30.0 }}
        rotation {{ y: -90 }}
        reference_rotation {{ z: {-theta*180/np.pi} }}
      }}
      joints {{
        name: "Aux 2_{str(ind)}_$ Body 5_{str(ind)}"
        parent_offset {{ x: 0.14142136096954344  }}
        child_offset {{ x:-0.28284270882606505  }}
        parent: "Aux 2_{str(ind)}"
        child: "$ Body 5_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        rotation: {{ z: 90 }}
        angle_limit {{
          min: 30.0
          max: 70.0
        }}
      }}
      actuators {{
        name: "torso_{str(ind)}_Aux 2_{str(ind)}"
        joint: "torso_{str(ind)}_Aux 2_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      actuators {{
        name: "Aux 2_{str(ind)}_$ Body 5_{str(ind)}"
        joint: "Aux 2_{str(ind)}_$ Body 5_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      """
    collides = (
      f'torso_{str(ind)}',
      f'Aux 1_{str(ind)}',
      f'Aux 2_{str(ind)}',
      f'$ Body 5_{str(ind)}')

    return tmp, collides

  def broken_leg_5(theta, ind):
    tmp = f"""
      bodies {{
        name: "torso_{str(ind)}"
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
      bodies {{
        name: "Aux 1_{str(ind)}"
        colliders {{
          rotation {{ x: 90 y: -90 }}
          capsule {{
            radius: 0.08
            length: 0.4428427219390869
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
            radius: 0.08
            length: 0.7256854176521301
            end: -1
          }}
        }}
        inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
        mass: 1
      }}
      joints {{
        name: "torso_{str(ind)}_Aux 1_{str(ind)}"
        parent_offset {{ x: {((0.4428427219390869/2.)+.08)*np.cos(theta)} y: {((0.4428427219390869/2.)+.08)*np.sin(theta)} }}
        child_offset {{ }}
        parent: "torso_{str(ind)}"
        child: "Aux 1_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -30.0 max: 30.0 }}
        rotation {{ y: -90 }}
        reference_rotation {{ z: {theta*180/np.pi} }}
      }}
      joints {{
        name: "Aux 1_{str(ind)}_$ Body 4_{str(ind)}"
        parent_offset {{ x: 0.14142136096954344  }}
        child_offset {{ x:-0.28284270882606505  }}
        parent: "Aux 1_{str(ind)}"
        child: "$ Body 4_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        rotation: {{ z: 90 }}
        angle_limit {{
          min: 30.0
          max: 70.0
        }}
      }}
      actuators {{
        name: "torso_{str(ind)}_Aux 1_{str(ind)}"
        joint: "torso_{str(ind)}_Aux 1_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      actuators {{
        name: "Aux 1_{str(ind)}_$ Body 4_{str(ind)}"
        joint: "Aux 1_{str(ind)}_$ Body 4_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      bodies {{
        name: "Aux 2_{str(ind)}"
        colliders {{
          rotation {{ x: 90 y: -90 }}
          capsule {{
            radius: 0.08
            length: 0.4428427219390869
          }}
        }}
        inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
        mass: 1
      }}
      joints {{
        name: "torso_{str(ind)}_Aux 2_{str(ind)}"
        parent_offset {{ x: {((0.4428427219390869/2.)+.08)*np.cos(-theta)} y: {((0.4428427219390869/2.)+.08)*np.sin(-theta)} }}
        child_offset {{ }}
        parent: "torso_{str(ind)}"
        child: "Aux 2_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -30.0 max: 30.0 }}
        rotation {{ y: -90 }}
        reference_rotation {{ z: {-theta*180/np.pi} }}
      }}
      actuators {{
        name: "torso_{str(ind)}_Aux 2_{str(ind)}"
        joint: "torso_{str(ind)}_Aux 2_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      """
    collides = (
      f'torso_{str(ind)}',
      f'Aux 1_{str(ind)}',
      f'$ Body 4_{str(ind)}',
      f'Aux 2_{str(ind)}')

    return tmp, collides

  def broken_limb_leg_4(theta, ind):
    tmp = f"""
      bodies {{
        name: "torso_{str(ind)}"
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
      bodies {{
        name: "Aux 2_{str(ind)}"
        colliders {{
          rotation {{ x: 90 y: -90 }}
          capsule {{
            radius: 0.08
            length: 0.4428427219390869
          }}
        }}
        inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
        mass: 1
      }}
      bodies {{
        name: "$ Body 5_{str(ind)}"
        colliders {{
          rotation {{ x: 90 y: -90 }}
          capsule {{
            radius: 0.08
            length: 0.7256854176521301
            end: -1
          }}
        }}
        inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
        mass: 1
      }}
      joints {{
        name: "torso_{str(ind)}_Aux 2_{str(ind)}"
        parent_offset {{ x: {((0.4428427219390869/2.)+.08)*np.cos(-theta)} y: {((0.4428427219390869/2.)+.08)*np.sin(-theta)} }}
        child_offset {{ }}
        parent: "torso_{str(ind)}"
        child: "Aux 2_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -30.0 max: 30.0 }}
        rotation {{ y: -90 }}
        reference_rotation {{ z: {-theta*180/np.pi} }}
      }}
      joints {{
        name: "Aux 2_{str(ind)}_$ Body 5_{str(ind)}"
        parent_offset {{ x: 0.14142136096954344  }}
        child_offset {{ x:-0.28284270882606505  }}
        parent: "Aux 2_{str(ind)}"
        child: "$ Body 5_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        rotation: {{ z: 90 }}
        angle_limit {{
          min: 30.0
          max: 70.0
        }}
      }}
      actuators {{
        name: "torso_{str(ind)}_Aux 2_{str(ind)}"
        joint: "torso_{str(ind)}_Aux 2_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      actuators {{
        name: "Aux 2_{str(ind)}_$ Body 5_{str(ind)}"
        joint: "Aux 2_{str(ind)}_$ Body 5_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      """
    collides = (
      f'torso_{str(ind)}',
      f'Aux 2_{str(ind)}',
      f'$ Body 5_{str(ind)}')

    return tmp, collides

  def broken_limb_leg_5(theta, ind):
    tmp = f"""
      bodies {{
        name: "torso_{str(ind)}"
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
      bodies {{
        name: "Aux 1_{str(ind)}"
        colliders {{
          rotation {{ x: 90 y: -90 }}
          capsule {{
            radius: 0.08
            length: 0.4428427219390869
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
            radius: 0.08
            length: 0.7256854176521301
            end: -1
          }}
        }}
        inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
        mass: 1
      }}
      joints {{
        name: "torso_{str(ind)}_Aux 1_{str(ind)}"
        parent_offset {{ x: {((0.4428427219390869/2.)+.08)*np.cos(theta)} y: {((0.4428427219390869/2.)+.08)*np.sin(theta)} }}
        child_offset {{ }}
        parent: "torso_{str(ind)}"
        child: "Aux 1_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -30.0 max: 30.0 }}
        rotation {{ y: -90 }}
        reference_rotation {{ z: {theta*180/np.pi} }}
      }}
      joints {{
        name: "Aux 1_{str(ind)}_$ Body 4_{str(ind)}"
        parent_offset {{ x: 0.14142136096954344  }}
        child_offset {{ x:-0.28284270882606505  }}
        parent: "Aux 1_{str(ind)}"
        child: "$ Body 4_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        rotation: {{ z: 90 }}
        angle_limit {{
          min: 30.0
          max: 70.0
        }}
      }}
      actuators {{
        name: "torso_{str(ind)}_Aux 1_{str(ind)}"
        joint: "torso_{str(ind)}_Aux 1_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      actuators {{
        name: "Aux 1_{str(ind)}_$ Body 4_{str(ind)}"
        joint: "Aux 1_{str(ind)}_$ Body 4_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      """
    collides = (
      f'torso_{str(ind)}',
      f'Aux 1_{str(ind)}',
      f'$ Body 4_{str(ind)}')

    return tmp, collides

  def template_joint(ind):
    tmp = f"""
      joints {{
        name: "torso_{str(ind)}_torso_{str(ind+1)}"
        parent_offset {{ x: 0.25 z: 0.0 }}
        child_offset {{ x: -0.25 z: -0.0 }}
        parent: "torso_{str(ind)}"
        child: "torso_{str(ind+1)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -60.0 max: 60.0 }}
        rotation {{ y: -90 z: 0.0 }}
        reference_rotation {{ y: 0.0 }}
      }}
      actuators {{
        name: "torso_{str(ind)}_torso_{str(ind+1)}"
        joint: "torso_{str(ind)}_torso_{str(ind+1)}"
        strength: 300.0
        torque {{}}
      }}
      """
    collides = tuple()
    return tmp, collides

  base_config = f""""""
  collides = tuple()
  for i in range(n):
    theta = np.pi / 2
    if broken_ids[1] != i:
      config_i, collides_i = template_torso(theta, i)
    else:
      if broken_ids[0] == 4:
        if len(broken_ids) == 3:
          config_i, collides_i = broken_limb_leg_4(theta, i)
        else:
          config_i, collides_i = broken_leg_4(theta, i)
      elif broken_ids[0] == 5:
        if len(broken_ids) == 3:
          config_i, collides_i = broken_limb_leg_5(theta, i)
        else:
          config_i, collides_i = broken_leg_5(theta, i)
    base_config += config_i
    collides += collides_i
    if i < n - 1:
      joint_i, _ = template_joint(i)
      base_config += joint_i
  return base_config, collides


def get_specs(n: int = 3, broken_ids: tuple = (4, 0)):
  message_str, collides = generate_centipede_config_with_n_torso(n, broken_ids)
  return dict(
      message_str=message_str,
      collides=collides,
      root='torso_0',
      term_fn=None)
