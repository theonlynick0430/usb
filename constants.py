# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785


CLEARANCE_HEIGHT = 0.25
PICK_HEIGHT = 0.15
INSERT_HEIGHT = 0.15
WIGGLE_HEIGHT = 0.17
WIGGLE_EPS = 0.03


# IK parameters
SOLVER = "quadprog"
POS_THRESHOLD = 1e-3
ORI_THRESHOLD = 1e-3
MAX_ITERS = 10000