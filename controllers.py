import mujoco
import numpy as np
import time
from abc import ABC, abstractmethod


class Controller(ABC):
    def __init__(self, model, data, viewer, dof_ids, actuator_ids, ee_body_id, dt, integration_dt):
        """
        Base controller class.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            viewer: MuJoCo viewer
            dof_ids: IDs of the dof to control
            actuator_ids: IDs of the actuators to control
            ee_body_id: ID of the end-effector body
            dt: Simulation timestep in seconds
            integration_dt: Integration timestep in seconds
        """
        self.model = model
        self.data = data
        self.viewer = viewer
        self.dof_ids = dof_ids
        self.actuator_ids = actuator_ids
        self.ee_body_id = ee_body_id
        self.dt = dt
        self.integration_dt = integration_dt

    @abstractmethod
    def step(self, delta_ee_pos, delta_ee_ori, **kwargs):
        """
        Compute control signal and step simulation.

        Args:
            delta_ee_pos: Delta end-effector position expressed in the world frame
            delta_ee_ori: Delta end-effector orientation (axis-angle) expressed in the world frame
            **kwargs: Additional keyword arguments specific to the controller implementation
        """
        raise NotImplementedError("step implemented in subclass")
    
    def compute_error(self, target_ee_pos, target_ee_ori):
        """
        Compute the error between the current end-effector pose and the target pose.

        Args:
            target_ee_pos: Target end-effector position expressed in the world frame
            target_ee_ori: Target end-effector orientation (quat) expressed in the world frame

        Returns:
            error_pos: Position error expressed in the world frame
            error_ori: Orientation error (axis-angle) expressed in the world frame
        """
        error = np.zeros(6)
        error_pos = error[:3]
        error_ori = error[3:]   
        ee_quat = np.zeros(4)
        ee_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)
        # pos error
        error_pos[:] = target_ee_pos - self.data.body(self.ee_body_id).xpos
        # ori error
        mujoco.mju_mat2Quat(ee_quat, self.data.body(self.ee_body_id).xmat)
        mujoco.mju_negQuat(ee_quat_conj, ee_quat)
        mujoco.mju_mulQuat(error_quat, target_ee_ori, ee_quat_conj)
        mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
        return error_pos, error_ori
    
    def linear_action(self, target_ee_pos, target_ee_ori, threshold=0.01, max_steps=100, **kwargs):
        """
        Closed loop action to move the end-effector linearly to the target pose.

        Args:
            target_ee_pos: Target end-effector position expressed in the world frame
            target_ee_ori: Target end-effector orientation (quat) expressed in the world frame
            threshold: Threshold for the error
            **kwargs: Additional keyword arguments specific to the controller implementation
        """
        error_pos, error_ori = self.compute_error(target_ee_pos, target_ee_ori)
        
        error_norm = np.linalg.norm(np.concatenate([error_pos, error_ori]))
        steps = 0
        while error_norm > threshold and steps < max_steps:
            step_start = time.time()

            self.step(error_pos, error_ori, **kwargs)

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

            error_pos, error_ori = self.compute_error(target_ee_pos, target_ee_ori)
            error_norm = np.linalg.norm(np.concatenate([error_pos, error_ori]))
            steps += 1
            
            time_until_next_step = self.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        return steps < max_steps
    
class DiffIKController(Controller):
    def __init__(self, model, data, viewer, dof_ids, actuator_ids, ee_body_id, dt, integration_dt=1.0, damping=1e-4, max_angvel=0.0):
        """
        Differential inverse kinematics controller.

        Shout out Kevin Zakka. I largely adapted this code from his controller repo: 
        https://github.com/kevinzakka/mjctrl/blob/main/diffik.py
        I believe OSC controller is usually preferred (faster). However, I am most 
        familiar with this controller and can explain its logic so I chose it 
        for this challenge. 
        
        Args:
            damping: Damping term for the pseudoinverse. This is used to prevent joint velocities from
                becoming too large when the Jacobian is close to singular.
            max_angvel: Maximum allowable joint velocity in rad/s. Set to 0 to disable.
        """
        super().__init__(model, data, viewer, dof_ids, actuator_ids, ee_body_id, dt, integration_dt)
        self.damping = damping
        self.max_angvel = max_angvel
    
    def step(self, delta_ee_pos, delta_ee_ori, **kwargs):
        # note: compute jacobian wrt to all dof in env but control 
        # for non robot dof will always be zero 
        jac = np.zeros((6, self.model.nv))
        diag = self.damping * np.eye(6)
        
        # Get the Jacobian with respect to the end-effector body
        mujoco.mj_jacBody(self.model, self.data, jac[:3], jac[3:], self.ee_body_id)
        
        # solve J @ dq = error
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, np.concatenate([delta_ee_pos, delta_ee_ori]))
        
        # clip to velocity limits
        if self.max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self.max_angvel:
                dq *= self.max_angvel / dq_abs_max
        
        # integrate joint velocities
        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)

        # clip to joint limits
        jnt_range = self.model.jnt_range[self.dof_ids].T
        np.clip(q[self.dof_ids], *jnt_range, out=q[self.dof_ids])

        # set control signal
        self.data.ctrl[self.actuator_ids] = q[self.dof_ids]


def panda_gripper_action(model, data, viewer, actuator_id, dt, open=True):
    """
    Open or close the panda gripper.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        viewer: MuJoCo viewer
        actuator_id: ID of the actuator to control
        dt: Simulation timestep in seconds
        open: True to open the gripper, False to close it
    """
    if open:
        data.ctrl[actuator_id] = 255.0
    else:
        data.ctrl[actuator_id] = 0.0
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
    
