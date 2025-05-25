import mujoco
import mujoco.viewer
from controllers import DiffIKController, panda_gripper_action
from rrt_connect import RRTConnect
from constants import *
from utils import domain_randomization, execute_joint_traj
import mink
import time
import math
import numpy as np 
from copy import deepcopy
import hydra
from omegaconf import DictConfig, OmegaConf


def converge_ik(configuration, tasks, dt, solver, limits, pos_threshold, ori_threshold, max_iters):
    """Runs up to 'max_iters' of IK steps. Returns True if position and orientation are below thresholds."""
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3, limits=limits)
        configuration.integrate_inplace(vel, dt)

        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

        if pos_achieved and ori_achieved:
            return True
    return False

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):    
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    tag = ""
    if config.scene != "plain":
        tag = f"{config.scene}_"
    model = mujoco.MjModel.from_xml_path(f"franka_emika_panda/usb_{tag}scene.xml")
    data = mujoco.MjData(model)

    # Enable gravity compensation
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    ee_body_id = model.body("hand").id
    usb_plug_body_id = model.body("usb_plug").id
    usb_port_body_id = model.body("usb_port").id

    # note: exclude gripper from control
    dof_ids = np.arange(7)
    actuator_ids = np.arange(7)
    gripper_actuator_id = 7

    # Initial joint configuration saved as a keyframe in the XML file.
    key_id = model.key("home").id

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset the simulation to the initial keyframe.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Toggle site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        domain_randomization(data)

        mujoco.mj_step(model, data)
        viewer.sync()

        controller = DiffIKController(model, data, viewer, dof_ids, actuator_ids, ee_body_id, dt, integration_dt, damping, max_angvel)

        # GOAL: plan between two key configurations
        # 1) pre-grasp pose
        pre_grasp_pos = data.body(usb_plug_body_id).xpos.copy() + np.array([0, 0, CLEARANCE_HEIGHT])
        ee_initial_quat = np.array([0, 1, 0, 0])
        usb_plug_body_quat = np.zeros(4)
        mujoco.mju_mat2Quat(usb_plug_body_quat, data.body(usb_plug_body_id).xmat)
        pre_grasp_quat = np.zeros(4)
        mujoco.mju_mulQuat(pre_grasp_quat, usb_plug_body_quat, ee_initial_quat)
        # 2) pre-insert pose
        pre_insert_pos = data.body(usb_port_body_id).xpos.copy() + np.array([0, 0, CLEARANCE_HEIGHT])
        usb_port_body_quat = np.zeros(4)
        mujoco.mju_mat2Quat(usb_port_body_quat, data.body(usb_port_body_id).xmat)
        pre_insert_quat = np.zeros(4)
        mujoco.mju_mulQuat(pre_insert_quat, usb_port_body_quat, ee_initial_quat)

        # Generate collision-free joint configurations at pre-grasp and pre-insert poses
        configuration = mink.Configuration(model)

        ik_task = mink.FrameTask(
            frame_name="hand",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        posture_task = mink.PostureTask(model=model, cost=1e-2)
        tasks = [ik_task, posture_task]

        limits = [
            mink.ConfigurationLimit(model=model),
        ]

        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)
        
        # 1) pre-grasp configuration
        print("Finding pre-grasp configuration...")
        ik_task.set_target(mink.SE3(np.concatenate([pre_grasp_quat, pre_grasp_pos])))
        if converge_ik(configuration, tasks, 0.005, SOLVER, limits, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS):
            pre_grasp_config = configuration.q[:7].copy()
            print("Found pre-grasp configuration")
        else:
            print("Failed to find pre-grasp configuration")
            return
        # 2) pre-insert configuration
        print("Finding pre-insert configuration...")
        ik_task.set_target(mink.SE3(np.concatenate([pre_insert_quat, pre_insert_pos])))
        if converge_ik(configuration, tasks, 0.005, SOLVER, limits, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS):
            pre_insert_config = configuration.q[:7].copy()
            print("Found pre-insert configuration")
        else:
            print("Failed to find pre-insert configuration")
            return
        
        # Plan paths between key configurations
        planner = RRTConnect(model, deepcopy(data), dof_ids, step_size=config.step_size, max_iters=config.max_iters)
        # 1) plan from init to pre-grasp
        plan1 = planner.plan(data.qpos[:7], pre_grasp_config)
        print(f"Found path with {len(plan1)} configurations")
        planner.reset()
        # 2) plan from pre-grasp to pre-insert
        plan2 = planner.plan(pre_grasp_config, pre_insert_config)
        print(f"Found path with {len(plan2)} configurations")

        # Execute paths + scripted policy
        # execute plan from init to pre-grasp
        execute_joint_traj(model, data, viewer, plan1, dof_ids, actuator_ids, dt)
        # scripted grasp
        controller.linear_action(pre_grasp_pos + np.array([0, 0, PICK_HEIGHT-CLEARANCE_HEIGHT]), pre_grasp_quat, max_steps=1000)
        panda_gripper_action(model, data, viewer, gripper_actuator_id, dt, open=False)
        controller.linear_action(pre_grasp_pos, pre_grasp_quat, max_steps=1000)
        # execute plan from pre-grasp to pre-insert
        execute_joint_traj(model, data, viewer, plan2, dof_ids, actuator_ids, dt)
        # pause 
        for _ in range(10):
            mujoco.mj_step(model, data)
            viewer.sync()
        # scripted insert
        controller.linear_action(pre_insert_pos + np.array([0, 0, INSERT_HEIGHT-CLEARANCE_HEIGHT]), pre_insert_quat, max_steps=1000)
        panda_gripper_action(model, data, viewer, gripper_actuator_id, dt, open=True)
        controller.linear_action(pre_insert_pos + np.array([0, WIGGLE_EPS, WIGGLE_HEIGHT-CLEARANCE_HEIGHT]), pre_insert_quat, max_steps=750)
        controller.linear_action(pre_insert_pos + np.array([0, -WIGGLE_EPS, WIGGLE_HEIGHT-CLEARANCE_HEIGHT]), pre_insert_quat, max_steps=750)
        controller.linear_action(pre_insert_pos + np.array([0, 0, WIGGLE_HEIGHT-CLEARANCE_HEIGHT]), pre_insert_quat, max_steps=750)
        controller.linear_action(pre_insert_pos, pre_insert_quat, max_steps=1000)

        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()