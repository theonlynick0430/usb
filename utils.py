import mujoco
import numpy as np


def domain_randomization(data):
    """
    Add random noise to the USB plug's position and orientation.
    """    
    data.qpos[-7:-5] += np.random.uniform(-0.02, 0.02, size=2)
    axis = np.array([0, 0, 1])
    angle = np.random.uniform(0, np.pi/4)
    mujoco.mju_axisAngle2Quat(data.qpos[-4:], axis, angle)