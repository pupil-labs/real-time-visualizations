# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
#     "pupil-labs-realtime-api",
#     "matplotlib",
# ]
# ///

import threading
from pathlib import Path
from queue import Empty, Queue

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pupil_labs.realtime_api.simple import Device, discover_one_device
from scipy.spatial.transform import Rotation as R

# --- Configuration Constants ---
FALLBACK_DEVICE_ADDRESS: str = "192.168.1.34"
FALLBACK_DEVICE_PORT: int = 8080
ANIMATION_FRAME_RATE: int = 30  # FPS
MODEL_FILE = Path(__file__).parent / "imu.obj"

# We will use a background thread to collect data
# via Neon's Real-time API. This helps to offload the data
# acquisition from the main thread, allowing for smoother
# visualization updates.

data_queue = Queue(maxsize=1)


def data_acquisition_loop():
    while True:
        # See our Python API Documentation for more info about these two functions:
        # https://pupil-labs.github.io/pl-realtime-api/dev/
        imu = device.receive_imu_datum()

        # Let's only pass data to the visualization when all relevant streams have
        # provided a datum. This makes the visualization logic simpler.
        if not imu:
            continue

        # Clear out the queue if it's already full.
        if data_queue.full():
            data_queue.get_nowait()

        data_queue.put_nowait(
            {
                "imu": imu,
            }
        )


# This is a helper function to load 3D models from OBJ files.
# It assumes the standard specification for OBJ files.
def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()
                face = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                faces.append(face)
    return np.array(vertices), faces


# Now, we make class to hold all the elements of the matplotlib figure.
# This makes it easier to organize the visualization logic later.
# The figure will have one section that simply displays the Neon module,
# rotated to match the current IMU orientation. It will update in real-time.
class Visualization:
    def __init__(self):
        self.original_vertices, self.faces = load_obj(MODEL_FILE)

        self.fig = plt.figure(figsize=(6, 6))
        self.fig.canvas.manager.set_window_title("Neon IMU Visualization")
        self.fig.patch.set_facecolor("black")

        self._setup_main_3d_axis()
        self._add_base_IMU_mesh()

    def _setup_main_3d_axis(self):
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Scale axes
        x, y, z = (
            self.original_vertices[:, 0],
            self.original_vertices[:, 1],
            self.original_vertices[:, 2],
        )
        self.ax.set_xlim(x.min(), x.max())
        self.ax.set_ylim(y.min(), y.max())
        self.ax.set_zlim(z.min(), z.max())

        # Makes sure that 3D plots have equal aspect ratio on all sides.
        # Adapted from:
        # https://github.com/matplotlib/matplotlib/issues/17172#issuecomment-830139107
        self.ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(self.ax, f"get_{a}lim")() for a in "xyz")]
        )

        self.ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
        self.ax.grid(False)
        self.ax.set_axis_off()

        self.ax.set_facecolor("black")

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

    def _add_base_IMU_mesh(self):
        mesh = [[self.original_vertices[idx] for idx in face] for face in self.faces]
        self.poly3d = Poly3DCollection(mesh, facecolors="gray", alpha=1.0, shade=True)
        self.ax.add_collection3d(self.poly3d)

    def update_IMU_plot(self, rotation_matrix):
        # Rotate vertices and update mesh
        rotated = self.original_vertices @ rotation_matrix.T
        new_mesh = [[rotated[idx] for idx in face] for face in self.faces]
        self.poly3d.set_verts(new_mesh)


if __name__ == "__main__":
    # First, establish a connection to Neon.
    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        device = Device(address=FALLBACK_DEVICE_ADDRESS, port=FALLBACK_DEVICE_PORT)
    if device is None:
        raise RuntimeError("No device found.")
    print(f"Connecting to device at {device.address}:{device.port}...")
    print("Connection successful.")

    # Start up the data acquisition thread.
    data_acquisition_thread = threading.Thread(
        target=data_acquisition_loop, daemon=True
    )
    data_acquisition_thread.start()

    viz = Visualization()

    # Now, we get to the main animation callback that will be repeatedly called by matplotlib's
    # animation routines for each frame. It essentially:
    #
    # - Gets the latest data in the data_queue, as provided by the data acquisition thread.
    # - Converts the data into a format that is more easily used for plotting.
    # - Updates the plot objects with the new data.
    # - Finally, the function returns, passing the plot objects to matplotlib's animation routine, so that it can draw
    #   the next frame to the plot.
    #
    # The frame_number argument is required by matplotlib, even if you do not use it in your function.
    def update(frame_number):
        try:
            data = data_queue.get_nowait()
            imu = data["imu"]
        except Empty:
            pass

        # If there is IMU data available, then update the model and
        # the 3D visualization.
        if imu is not None:
            quaternion = imu.quaternion
            rotation_matrix = R.from_quat(quaternion).as_matrix()

            viz.update_IMU_plot(rotation_matrix)

        # Finally, we return a list of plot objects, per
        # matplotlib's conventions.
        return [viz.poly3d]

    # Wrap the actual animation "loop" in a try-except-finally block.
    # This is done because if anything goes wrong or the
    # user quits the program, then we need to cleanly close our
    # connection to Neon and clean up any OpenCV windows.
    try:
        # Create a matplotlib animation function that will run the `update` function
        # every 33ms. This will update our animation at ~30 FPS. We pass `blit=False`,
        # because matplotlib's 3D functionality does not support blitting.
        interval_ms = 1000 / ANIMATION_FRAME_RATE
        ani = animation.FuncAnimation(viz.fig, update, interval=33, blit=False)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # No matter what happens, we need to cleanly close our connection to Neon.
        print("Cleaning up resources...")
        device.close()
        print("Done.")
