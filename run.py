import mujoco
import mujoco.viewer
import time

def main():
    # Load the model from the USB scene
    model = mujoco.MjModel.from_xml_path("franka_emika_panda/usb_scene.xml")
    data = mujoco.MjData(model)

    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set the camera position for a good view
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20

        # Main simulation loop
        while viewer.is_running():
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Update the viewer
            viewer.sync()
            
            # Control the simulation speed
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
