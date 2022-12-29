import jax.numpy as jnp


AMORPHEUS = dict()
AMORPHEUS_GOAL = dict()

# registar torso
root_name = ['$ Torso', 'torso', 'torso_0', 'palm']
for name in root_name:
  AMORPHEUS[name] = jnp.array((1., 0., 0., 0.))
  AMORPHEUS_GOAL[name] = jnp.array((1., 0., 0., 0., 0.))

# registar thigh
thigh_name = ['bthigh', 'fthigh', 'right_thigh', 'left_thigh', 'thigh', 'thigh_left']
for i in range(10):
  thigh_name.append(f'Aux 1_{i}')
  thigh_name.append(f'Aux 2_{i}')
  thigh_name.append(f'thumb_proximal_{i}')
for i in range(1, 11):
  thigh_name.append(f'torso_{i}')
for i in range(0, 22):
  thigh_name.append(f'd1_limb_{i}')
for name in thigh_name:
  AMORPHEUS[name] = jnp.array((0., 1., 0., 0.))
  AMORPHEUS_GOAL[name] = jnp.array((0., 1., 0., 0., 0.))

# registar leg
leg_name = ['bshin', 'fshin', 'leg', 'lower_leg', 'right_shin', 'left_shin', 'leg_left']
for i in range(10):
  leg_name.append(f'$ Body 4_{i}')
  leg_name.append(f'$ Body 5_{i}')
  leg_name.append(f'thumb_middle_{i}')
for i in range(0, 22):
  leg_name.append(f'd2_limb_{i}')
for name in leg_name:
  AMORPHEUS[name] = jnp.array((0., 0., 1., 0.))
  AMORPHEUS_GOAL[name] = jnp.array((0., 0., 1., 0., 0.))

# registar foot
foot_name = ['bfoot', 'ffoot', 'foot', 'foot_left']
for i in range(10):
  foot_name.append(f'thumb_distal_{i}')
for i in range(0, 22):
  foot_name.append(f'd3_limb_{i}')
for name in foot_name:
  AMORPHEUS[name] = jnp.array((0., 0., 0., 1.))
  AMORPHEUS_GOAL[name] = jnp.array((0., 0., 0., 1., 0.))

# registar goal
goal_name = ['Ball', 'Target', 'Box', 'Z_Target']
for i in range(1, 3):
  goal_name.append(f'Ball_{i}')
  goal_name.append(f'Z_Target_{i}')
for name in goal_name:
  AMORPHEUS[name] = jnp.array((0., 0., 0., 0.))
  AMORPHEUS_GOAL[name] = jnp.array((0., 0., 0., 0., 1.))

# registar others
others_name = ['lwaist', 'pelvis', 'right_upper_arm', 'right_lower_arm', 'left_upper_arm', 'left_lower_arm']
for i in range(0, 22):
  others_name.append(f'd4_limb_{i}')
  others_name.append(f'limb_end_{i}')
for name in others_name:
  AMORPHEUS[name] = jnp.array((0., 0., 0., 0.))
  AMORPHEUS_GOAL[name] = jnp.array((0., 0., 0., 0., 0.))
