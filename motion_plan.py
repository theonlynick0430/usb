import mujoco
import mujoco.viewer
from controllers import DiffIKController, panda_gripper_action
import mink
import time
import numpy as np 


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


# IK parameters
SOLVER = "quadprog"
POS_THRESHOLD = 1e-3
ORI_THRESHOLD = 1e-3
MAX_ITERS = 10000


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

def domain_randomization(data):
    """
    Add random noise to the USB plug's position and orientation.
    """    
    data.qpos[-7:-5] += np.random.uniform(-0.02, 0.02, size=2)
    axis = np.array([0, 0, 1])
    angle = np.random.uniform(0, np.pi/4)
    mujoco.mju_axisAngle2Quat(data.qpos[-4:], axis, angle)

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("franka_emika_panda/usb_scene.xml")
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

        # Generate collision-free joint configurations at pre-grasp and pre-insert using mink
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

        while viewer.is_running():
            step_start = time.time()

            # Set robot controls to pre-grasp configuration
            data.ctrl[:7] = pre_insert_config

            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()