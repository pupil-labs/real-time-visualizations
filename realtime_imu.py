import threading
from queue import Queue

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pupil_labs.realtime_api.simple import Device
from scipy.spatial.transform import Rotation as R


def transform_imu_to_world(imu_coordinates, imu_quaternions):
    # This array contains a timeseries of transformation matrices,
    # as calculated from the IMU's timeseries of quaternions values.
    imu_to_world_matrices = R.from_quat(imu_quaternions).as_matrix()

    if np.ndim(imu_coordinates) == 1:
        return imu_to_world_matrices @ imu_coordinates
    else:
        return np.array(
            [
                imu_to_world @ imu_coord
                for imu_to_world, imu_coord in zip(
                    imu_to_world_matrices, imu_coordinates
                )
            ]
        )


def imu_heading_in_world(imu_quaternions):
    heading_neutral_in_imu_coords = np.array([0.0, 1.0, 0.0])
    return transform_imu_to_world(heading_neutral_in_imu_coords, imu_quaternions)


device = Device(address="192.168.1.35", port=8080)

imu_queue = Queue(maxsize=1)


def data_acquisition():
    while True:
        imu = device.receive_imu_datum()

        data = {
            "imu": imu,
        }

        if imu_queue.full():
            imu_queue.get_nowait()

        imu_queue.put_nowait(data)


imu_thread = threading.Thread(target=data_acquisition)
imu_thread.start()


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


original_vertices, faces = load_obj("imu.obj")

fig = plt.figure()
fig.canvas.manager.set_window_title("3D Eye State Visualization")
fig.patch.set_facecolor("black")

ax = fig.add_subplot(131, projection="3d")
ax.axis("square")
ax.set_aspect("equal", adjustable="box")
ax.set_box_aspect([1, 1, 1])

ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
ax.grid(False)
ax.set_axis_off()

ax.set_facecolor("black")

mesh = [[original_vertices[idx] for idx in face] for face in faces]
poly3d = Poly3DCollection(mesh, facecolors="gray", alpha=1.0, shade=True)
ax.add_collection3d(poly3d)

# Scale axes
x, y, z = original_vertices[:, 0], original_vertices[:, 1], original_vertices[:, 2]
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_zlim(min(z), max(z))

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax2 = fig.add_subplot(132)

ax2.axis("square")
ax2.set_aspect("equal", adjustable="box")
ax2.set_xlim(-1.29, 1.29)
ax2.set_ylim(-1.29, 1.29)

ax2.axis("off")
ax2.set_facecolor("black")
ax2.set_xticks([])
ax2.set_yticks([])

circle = plt.Circle((0, 0), 1.0, color="white", fill=False)
ax2.add_patch(circle)

ax2.text(0, 1.11, "N", ha="center", va="bottom", fontsize="xx-large", color="white")
ax2.text(0, -1.11, "S", ha="center", va="top", fontsize="xx-large", color="white")
ax2.text(1.11, 0, "E", ha="left", va="center", fontsize="xx-large", color="white")
ax2.text(-1.11, 0, "W", ha="right", va="center", fontsize="xx-large", color="white")

ax2.plot([0, 0], [1.0, 1.08], color="white")
ax2.plot([0, 0], [-1.0, -1.08], color="white")
ax2.plot([1.0, 1.08], [0, 0], color="white")
ax2.plot([-1.0, -1.08], [0, 0], color="white")

nsew_plot = ax2.quiver(
    0,
    0,
    0,
    0,
    color="b",
    scale=1,
    scale_units="xy",
    angles="xy",
    width=0.01,
)

ax3 = fig.add_subplot(133)

ax3.axis("square")
ax3.set_aspect("equal", adjustable="box")
ax3.set_xlim(-1.29, 1.29)
ax3.set_ylim(-1.29, 1.29)

ax3.axis("off")
ax3.set_facecolor("black")
ax3.set_xticks([])
ax3.set_yticks([])

circle = plt.Circle((0, 0), 1.0, color="white", fill=False)
ax3.add_patch(circle)

ax3.plot([0, 0], [1.0, 1.08], color="white")
ax3.plot([0, 0], [-1.0, -1.08], color="white")

ax3.text(0, 1.11, "Sky", ha="center", va="bottom", fontsize="xx-large", color="white")
ax3.text(0, -1.11, "Earth", ha="center", va="top", fontsize="xx-large", color="white")

ax3.hlines(0, -1.0, 1.0, color="white", linestyle="--")

earth_sky_plot = ax3.quiver(
    0,
    0,
    0,
    0,
    color="b",
    scale=1,
    scale_units="xy",
    angles="xy",
    width=0.01,
)


def update(frame):
    if not imu_queue.empty():
        data = imu_queue.get_nowait()
        quat = data["imu"].quaternion
        rotation_matrix = R.from_quat(quat).as_matrix()

        # Rotate vertices and update mesh
        rotated = original_vertices @ rotation_matrix.T
        new_mesh = [[rotated[idx] for idx in face] for face in faces]
        poly3d.set_verts(new_mesh)

        heading_in_world = imu_heading_in_world(quat)

        nsew_heading = np.array([heading_in_world[0], heading_in_world[1]])
        nsew_heading /= np.linalg.norm(nsew_heading)

        global nsew_plot
        nsew_plot.remove()
        nsew_plot = ax2.quiver(
            0,
            0,
            nsew_heading[0],
            nsew_heading[1],
            color="b",
            scale=1,
            scale_units="xy",
            angles="xy",
            width=0.01,
        )

        earth_sky_heading = np.array([heading_in_world[1], heading_in_world[2]])
        earth_sky_heading /= np.linalg.norm(earth_sky_heading)

        global earth_sky_plot
        earth_sky_plot.remove()
        earth_sky_plot = ax3.quiver(
            0,
            0,
            earth_sky_heading[0],
            earth_sky_heading[1],
            color="b",
            scale=1,
            scale_units="xy",
            angles="xy",
            width=0.01,
        )

        return (poly3d,)


ani = FuncAnimation(fig, update, interval=33, blit=False)
plt.tight_layout()
plt.show()
