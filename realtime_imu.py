import threading
from queue import Queue

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pupil_labs.realtime_api.simple import Device
from scipy.spatial.transform import Rotation as R

device = Device(address="192.168.1.57", port=8080)

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


def update(frame):
    if not imu_queue.empty():
        data = imu_queue.get_nowait()
        quat = data["imu"].quaternion
        rotation_matrix = R.from_quat(quat).as_matrix()

        # Rotate vertices and update mesh
        rotated = original_vertices @ rotation_matrix.T
        new_mesh = [[rotated[idx] for idx in face] for face in faces]
        poly3d.set_verts(new_mesh)

        return (poly3d,)


ani = FuncAnimation(fig, update, interval=33, blit=False)
plt.tight_layout()
plt.show()
