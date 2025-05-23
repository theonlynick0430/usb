import mujoco
import numpy as np

def diffik(target_pos, target_quat, model, data, ee_body_id, integration_dt=1.0, damping=1e-4, max_angvel=0.0):
    """
    Compute control signal using differential inverse kinematics.
    
    Shout out Kevin Zakka. I largely adapted this code from his controller repo: 
    https://github.com/kevinzakka/mjctrl/blob/main/diffik.py
    I believe OSC controller is usually preferred (faster). However, I am most 
    familiar with this controller and can explain its logic so I chose it 
    for this challenge. 
    
    Args:
        target_pos: Target end-effector position
        target_quat: Target end-effector orientation
        model: MuJoCo model
        data: MuJoCo data
        ee_body_id: ID of the end-effector body
        integration_dt: Integration timestep in seconds
        damping: Damping term for the pseudoinverse. This is used to prevent joint velocities from
            becoming too large when the Jacobian is close to singular.
        max_angvel: Maximum allowable joint velocity in rad/s. Set to 0 to disable.
        
    Returns:
        control_signal: Joint positions to achieve the target pose
    """
    # note: compute jacobian wrt to all dof in env but control 
    # for non robot dof will always be zero 
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    
    # pos error
    error_pos[:] = target_pos - data.body(ee_body_id).xpos
    
    # ori error
    mujoco.mju_mat2Quat(site_quat, data.body(ee_body_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    # calculate difference quat
    mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
    mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
    
    # Get the Jacobian with respect to the end-effector body
    mujoco.mj_jacBody(model, data, jac[:3], jac[3:], ee_body_id)
    
    # solve J @ dq = error
    dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
    
    # clip to velocity limits
    if max_angvel > 0:
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > max_angvel:
            dq *= max_angvel / dq_abs_max
    
    # integrate joint velocities
    q = data.qpos.copy()
    mujoco.mj_integratePos(model, q, dq, integration_dt)
    
    # clip to joint limits
    np.clip(q, *model.jnt_range.T, out=q)
    
    return q
