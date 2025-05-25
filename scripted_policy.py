import mujoco
import mujoco.viewer
from controllers import DiffIKController, panda_gripper_action
from utils import domain_randomization
from constants import *
import time
import numpy as np 


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
        # pause 
        for _ in range(10):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.1)
        # move down
        controller.linear_action(usb_port_pos + np.array([0, 0, INSERT_HEIGHT]), ee_initial_quat, max_steps=1000)
        # open gripper
        panda_gripper_action(model, data, viewer, gripper_actuator_id, dt, open=True)
        # move up
        controller.linear_action(usb_port_pos + np.array([0, 0, CLEARANCE_HEIGHT]), ee_initial_quat, max_steps=1000)

        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            
            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()