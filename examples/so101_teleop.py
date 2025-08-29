"""
Keyboard Controls:
↑	- Move Forward (North)
↓	- Move Backward (South)
←	- Move Left (West)
→	- Move Right (East)
n	- Move Up
m	- Move Down
j	- Rotate Counterclockwise
k	- Rotate Clockwise
u	- Reset Scene
space	- Press to close gripper, release to open gripper
esc	- Quit
"""

import random
import threading

import genesis as gs
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R


class KeyboardDevice:
    def __init__(self):
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()
        self.listener.join()

    def on_press(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.add(key)

    def on_release(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.discard(key)

    def get_cmd(self):
        return self.pressed_keys


def build_scene():
    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="info", backend=gs.cpu)
    np.set_printoptions(precision=7, suppress=True)

    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=4,  # More substeps for better collision handling
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            gravity=(0, 0, -9.8),
            box_box_detection=True,
            constraint_timeconst=0.02,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.7),
            camera_lookat=(0.2, 0.0, 0.1),
            camera_fov=50,
            max_FPS=60,
        ),
        show_viewer=True,
        show_FPS=False,
    )

    ########################## entities ##########################
    entities = dict()
    entities["plane"] = scene.add_entity(
        gs.morphs.Plane(),
    )

    entities["robot"] = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.MJCF(
            file="genesis/assets/xml/so101_robot/so101_robot.xml",
            euler=(0, 0, 0),
            convexify=True,
            decompose_robot_error_threshold=0,
        ),
        # vis_mode="collision",
    )
    entities["cube"] = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.07),
            size=(0.04, 0.04, 0.04),
        ),
        surface=gs.surfaces.Default(color=(0.5, 1, 0.5)),
    )

    entities["target"] = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    ########################## build ##########################
    scene.build()

    return scene, entities


def run_sim(scene, entities, clients):
    robot = entities["robot"]
    target_entity = entities["target"]

    # SO101 has 6 DOFs: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    n_dofs = robot.n_dofs

    # Define specific joint indices for direct control
    shoulder_pan_dof = np.array([0])  # Base rotation (left/right)
    shoulder_lift_dof = np.array([1])  # Shoulder up/down
    elbow_flex_dof = np.array([2])  # Elbow bend
    wrist_flex_dof = np.array([3])  # Wrist bend
    wrist_roll_dof = np.array([4])  # Wrist rotation (j/k keys)
    gripper_dof = np.array([5])  # Gripper open/close (space)

    # Get current joint positions for direct control
    current_q = robot.get_qpos()

    def reset_scene():
        # Reset robot to initial position
        initial_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # All joints at zero
        robot.set_qpos(initial_q)

        # Reset cube to random position (like original keyboard_teleop)
        entities["cube"].set_pos((random.uniform(0.2, 0.4), random.uniform(-0.2, 0.2), 0.05))
        entities["cube"].set_quat(R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat(scalar_first=True))

    print("\nKeyboard Controls:")
    print("↑/↓\t- Shoulder up/down (2nd joint)")
    print("←/→\t- Base rotate left/right (bottom joint)")
    print("n/m\t- Elbow up/down (3rd joint)")
    print("o/p\t- Wrist flex up/down (3rd joint)")
    print("j/k\t- Wrist rotate counterclockwise/clockwise (4th joint)")
    print("u\t- Reset Scene")
    print("space\t- Press to close gripper, release to open gripper")
    print("esc\t- Quit")

    # reset scen before starting teleoperation
    reset_scene()

    # start teleoperation
    stop = False
    while not stop:
        pressed_keys = clients["keyboard"].pressed_keys.copy()

        # reset scene:
        reset_flag = False
        reset_flag |= keyboard.KeyCode.from_char("u") in pressed_keys
        if reset_flag:
            reset_scene()

        # stop teleoperation
        stop = keyboard.Key.esc in pressed_keys

        # Direct joint control
        current_q = robot.get_qpos()
        djoint = 0.05  # Joint movement step size (increased for faster movement)

        # Track gripper state
        is_close_gripper = False

        for key in pressed_keys:
            if key == keyboard.Key.up:
                # Move shoulder_lift down (2nd joint) - REVERSED
                current_q[1] -= djoint
            elif key == keyboard.Key.down:
                # Move shoulder_lift up (2nd joint) - REVERSED
                current_q[1] += djoint
            elif key == keyboard.Key.right:
                # Move shoulder_pan left (bottom joint) - REVERSED
                current_q[0] -= djoint
            elif key == keyboard.Key.left:
                # Move shoulder_pan right (bottom joint) - REVERSED
                current_q[0] += djoint
            elif key == keyboard.KeyCode.from_char("n"):
                # Move elbow down (3rd joint) - REVERSED
                current_q[2] -= djoint
            elif key == keyboard.KeyCode.from_char("m"):
                # Move elbow up (3rd joint) - REVERSED
                current_q[2] += djoint
            elif key == keyboard.KeyCode.from_char("j"):
                # Rotate wrist clockwise (4th joint) - REVERSED
                current_q[4] -= djoint
            elif key == keyboard.KeyCode.from_char("k"):
                # Rotate wrist counterclockwise (4th joint) - REVERSED
                current_q[4] += djoint
            elif key == keyboard.KeyCode.from_char("o"):
                # Wrist flex down (3rd joint) - REVERSED
                current_q[3] -= djoint
            elif key == keyboard.KeyCode.from_char("p"):
                # Wrist flex up (3rd joint) - REVERSED
                current_q[3] += djoint
            elif key == keyboard.Key.space:
                # Close gripper
                is_close_gripper = True

        # Apply joint positions for arm joints only (like Franka)
        motors_dof = np.arange(n_dofs - 1)  # All joints except gripper
        robot.control_dofs_position(current_q[:-1], motors_dof)

        # Control gripper with force like Franka (increased force)
        if is_close_gripper:
            robot.control_dofs_force(np.array([-1.0]), gripper_dof)  # Much stronger closing force
        else:
            robot.control_dofs_force(np.array([1.0]), gripper_dof)  # Stronger opening force

        # Update target entity position to follow wrist joint (closest to gripper)
        wrist_link = robot.get_link("wrist")
        wrist_pos = wrist_link.get_pos()
        wrist_quat = wrist_link.get_quat()
        target_entity.set_qpos(np.concatenate([np.array(wrist_pos), np.array(wrist_quat)]))

        scene.step()


def main():
    clients = dict()
    clients["keyboard"] = KeyboardDevice()
    clients["keyboard"].start()

    scene, entities = build_scene()
    run_sim(scene, entities, clients)


if __name__ == "__main__":
    main()
