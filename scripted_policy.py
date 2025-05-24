import mujoco
import mujoco.viewer
from controllers import DiffIKController, panda_gripper_action
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

        # Simplified state machine for usb plug insertion

        # 1) Pick up the usb plug
        # move above the usb plug
        usb_plug_pos = data.body(usb_plug_body_id).xpos.copy()
        ee_initial_quat = np.array([0, 1, 0, 0])
        controller.linear_action(usb_plug_pos + np.array([0, 0, CLEARANCE_HEIGHT]), ee_initial_quat, max_steps=2000)
        # rotate to usb plug orientation
        usb_plug_body_quat = np.zeros(4)
        mujoco.mju_mat2Quat(usb_plug_body_quat, data.body(usb_plug_body_id).xmat)
        aligned_ee_quat = np.zeros(4)
        mujoco.mju_mulQuat(aligned_ee_quat, usb_plug_body_quat, ee_initial_quat)
        controller.linear_action(usb_plug_pos + np.array([0, 0, CLEARANCE_HEIGHT]), aligned_ee_quat, max_steps=1000)
        # open gripper
        panda_gripper_action(model, data, viewer, gripper_actuator_id, dt, open=True)
        # move down
        controller.linear_action(usb_plug_pos + np.array([0, 0, PICK_HEIGHT]), aligned_ee_quat, max_steps=1000)
        # close gripper
        panda_gripper_action(model, data, viewer, gripper_actuator_id, dt, open=False)
        # move up 
        controller.linear_action(usb_plug_pos + np.array([0, 0, CLEARANCE_HEIGHT]), aligned_ee_quat, max_steps=1000)

        # 2) Insert the usb plug into the usb port
        # move above the usb port
        usb_port_pos = data.body(usb_port_body_id).xpos.copy()
        controller.linear_action(usb_port_pos + np.array([0, 0, CLEARANCE_HEIGHT]), aligned_ee_quat, max_steps=2000)
        # rotate to usb port orientation
        controller.linear_action(usb_port_pos + np.array([0, 0, CLEARANCE_HEIGHT]), ee_initial_quat, max_steps=1500)
        # move down
        controller.linear_action(usb_port_pos + np.array([0, 0, INSERT_HEIGHT]), ee_initial_quat, max_steps=1000)
        # open gripper
        panda_gripper_action(model, data, viewer, gripper_actuator_id, dt, open=True)
        # move up
        controller.linear_action(usb_port_pos + np.array([0, 0, CLEARANCE_HEIGHT]), ee_initial_quat, max_steps=1000)

        while viewer.is_running():
            step_start = time.time()

            # Step the simulation
            mujoco.mj_step(model, data)
            
            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()