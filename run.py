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


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("franka_emika_panda/usb_scene.xml")
    data = mujoco.MjData(model)

    # Enable gravity compensation
    model.body_gravcomp[:] = float(gravity_compensation)
    # disable for usb_plug
    usb_plug_body_id = model.body("usb_plug").id
    model.body_gravcomp[usb_plug_body_id] = 0.0
    model.opt.timestep = dt

    # End-effector body we wish to control, in this case the hand
    ee_body_id = model.body("hand").id

    # note: exclude gripper from control
    dof_ids = np.arange(7)
    actuator_ids = np.arange(7)

    # Initial joint configuration saved as a keyframe in the XML file.
    key_id = model.key("home").id

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset the simulation to the initial keyframe.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Toggle site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        mujoco.mj_step(model, data)
        viewer.sync()

        controller = DiffIKController(model, data, viewer, dof_ids, actuator_ids, ee_body_id, dt, integration_dt, damping, max_angvel)

        ee_body_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_body_quat, data.body(ee_body_id).xmat)
        usb_plug_body_id = model.body("usb_plug").id
        usb_plug_body_quat = np.zeros(4)
        mujoco.mju_mat2Quat(usb_plug_body_quat, data.body(usb_plug_body_id).xmat)
        target_ee_quat = np.zeros(4)
        mujoco.mju_mulQuat(target_ee_quat, ee_body_quat, usb_plug_body_quat)
        target_ee_pos = data.body(usb_plug_body_id).xpos.copy()
        target_ee_pos[2] += 0.25

        gripper_actuator_id = 7

        controller.linear_action(target_ee_pos, target_ee_quat, max_steps=2000)
        panda_gripper_action(model, data, viewer, gripper_actuator_id, dt, open=False)

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