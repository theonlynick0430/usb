import mujoco
import numpy as np
import time


def domain_randomization(data):
    """
    Add random noise to the USB plug's position and orientation.
    """    
    data.qpos[-7:-5] += np.random.uniform(-0.02, 0.02, size=2)
    axis = np.array([0, 0, 1])
    angle = np.random.uniform(0, np.pi/4)
    mujoco.mju_axisAngle2Quat(data.qpos[-4:], axis, angle)


def execute_joint_traj(
    model, 
    data, 
    viewer, 
    trajectory, 
    dof_ids, 
    actuator_ids, 
    dt, 
    integration_dt=0.05, 
    max_angvel=0.785,
    threshold=1e-3,
):
    """
    Follows a joint trajectory smoothly using velocity integration (dq), simulating realistic motion.

    Args:
        model: MjModel
        data: MjData
        viewer: mujoco.viewer viewer instance
        trajectory: list of joint configurations
        dof_ids: indices of controlled joints
        actuator_ids: indices of actuators
        dt: simulation timestep
        integration_dt: time to integrate each velocity command
        max_angvel: max allowable joint velocity (rad/s)
        threshold: threshold for the error
    """
    q = data.qpos.copy()
    jnt_range = model.jnt_range[dof_ids].T

    for q_target in trajectory:
        dq = np.zeros(model.nv)
        dq[dof_ids] = q_target - q[dof_ids]
        while np.linalg.norm(dq) > threshold:
            step_start = time.time()

            # clip to velocity limits
            if max_angvel > 0:
                dq_abs_max = np.abs(dq[dof_ids]).max()
                if dq_abs_max > max_angvel:
                    dq[dof_ids] *= max_angvel / dq_abs_max

            # integrate joint velocities
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            # clip to joint limits
            np.clip(q[dof_ids], *jnt_range, out=q[dof_ids])

            # set control signal 
            data.ctrl[actuator_ids] = q[dof_ids]

            mujoco.mj_step(model, data)
            viewer.sync()

            dq[dof_ids] = q_target - q[dof_ids]

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)