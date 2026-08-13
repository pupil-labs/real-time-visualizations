# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python",
#     "numpy",
#     "pupil-labs-realtime-api",
#     "pyqtgraph",
#     "PySide6",
#     "PyOpenGL",
#     "scipy",
# ]
# ///

import sys
import threading
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pupil_labs.realtime_api.simple import Device, discover_one_device
from pyqtgraph import functions as fn
from pyqtgraph.opengl import GLMeshItem, MeshData
from pyqtgraph.Qt import QtGui
from PySide6 import QtCore
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial.transform import Rotation as R

import colors
from threeD_eye_model import ThreeDEyeModel, apply_pose_to_texture


class CenteredArrowItem(pg.ArrowItem):
    def setStyle(self, **opts):
        # http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/graphicsItems/ArrowItem.html#ArrowItem.setStyle
        # self.opts.update(opts)

        arrowOpts = [
            "headLen",
            "tipAngle",
            "baseAngle",
            "tailLen",
            "tailWidth",
            "headWidth",
        ]
        allowedOpts = ["angle", "pen", "brush", "pxMode"] + arrowOpts
        needUpdate = False
        for k, v in opts.items():
            if k not in allowedOpts:
                raise KeyError('Invalid arrow style option "%s"' % k)
            if self.opts.get(k) != v:
                needUpdate = True
            self.opts[k] = v

        if not needUpdate:
            return

        opt = dict([(k, self.opts[k]) for k in arrowOpts if k in self.opts])
        tr = QtGui.QTransform()
        path = fn.makeArrowPath(**opt)
        tr.rotate(self.opts["angle"])
        p = -path.boundingRect().center() * 2
        tr.translate(p.x(), p.y())
        self.path = tr.map(path)
        self.setPath(self.path)

        self.setPen(fn.mkPen(self.opts["pen"]))
        self.setBrush(fn.mkBrush(self.opts["brush"]))

        if self.opts["pxMode"]:
            self.setFlags(
                self.flags() | self.GraphicsItemFlag.ItemIgnoresTransformations
            )
        else:
            self.setFlags(
                self.flags() & ~self.GraphicsItemFlag.ItemIgnoresTransformations
            )


# --- Configuration Constants ---
# FALLBACK_DEVICE_ADDRESS: str = "192.168.1.229"
FALLBACK_DEVICE_ADDRESS: str = "192.168.178.36"
FALLBACK_DEVICE_PORT: int = 8080
ANIMATION_FRAME_RATE: int = 30  # FPS
MODEL_FILE = Path(__file__).parent / "imu.obj"


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


def transform_scene_to_imu(
    coords_in_scene, translation_in_imu=np.array([0.0, -1.3, -6.62])
):
    imu_scene_rotation_diff = np.deg2rad(-90 - 12)
    scene_to_imu = np.array(
        [
            [1.0, 0.0, 0.0],
            [
                0.0,
                np.cos(imu_scene_rotation_diff),
                -np.sin(imu_scene_rotation_diff),
            ],
            [
                0.0,
                np.sin(imu_scene_rotation_diff),
                np.cos(imu_scene_rotation_diff),
            ],
        ]
    )

    coords_in_imu = scene_to_imu @ coords_in_scene.T

    coords_in_imu[0, :] += translation_in_imu[0]
    coords_in_imu[1, :] += translation_in_imu[1]
    coords_in_imu[2, :] += translation_in_imu[2]

    return coords_in_imu.T


def transform_scene_to_world(
    coords_in_scene, imu_quaternions, translation_in_imu=np.array([0.0, -1.3, -6.62])
):
    coords_in_imu = transform_scene_to_imu(coords_in_scene, translation_in_imu)
    return transform_imu_to_world(coords_in_imu, imu_quaternions)


def gaze_3d_to_world(cart_gazes_in_scene, imu_quaternions):
    return transform_scene_to_world(
        cart_gazes_in_scene, imu_quaternions, translation_in_imu=np.zeros(3)
    )


def cartesian_to_spherical_world(world_points_3d):
    """
    Convert points in 3D Cartesian world coordinates to spherical coordinates.

    For elevation:
      - Neutral orientation = 0 (i.e., parallel with horizon)
      - Upwards is positive
      - Downwards is negative

    For azimuth:
      - Neutral orientation = 0 (i.e., aligned with magnetic North)
      - Leftwards is positive
      - Rightwards is negative
    """

    x = world_points_3d[:, 0]
    y = world_points_3d[:, 1]
    z = world_points_3d[:, 2]

    radii = np.sqrt(x**2 + y**2 + z**2)

    elevation = -(np.arccos(z / radii) - np.pi / 2)
    azimuth = np.arctan2(y, x) - np.pi / 2

    # Keep all azimuth values in the range of [-180, 180] to remain
    # consistent with the yaw orientation values provided by the IMU.
    azimuth[azimuth < -np.pi] += 2 * np.pi
    azimuth[azimuth > np.pi] -= 2 * np.pi

    elevation = np.rad2deg(elevation)
    azimuth = np.rad2deg(azimuth)

    return elevation, azimuth


def load_obj(path):
    """Very basic OBJ loader (triangles only, no materials/normals/UVs)."""
    vertices = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):  # vertex line
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):  # face line
                parts = line.strip().split()
                # OBJ indices start at 1
                face = [int(p.split("/")[0]) - 1 for p in parts[1:4]]
                faces.append(face)
    return np.array(vertices, dtype=float), np.array(faces, dtype=int)


data_queue = Queue(maxsize=1)
scene_queue = Queue()
render_queue = Queue()


def data_acquisition_loop():
    while True:
        # See our Python API Documentation for more info about these two functions:
        # https://pupil-labs.github.io/pl-realtime-api/dev/
        imu = device.receive_imu_datum(timeout_seconds=0.012)
        gaze = device.receive_gaze_datum(timeout_seconds=0.012)
        scene_image = device.receive_scene_video_frame(timeout_seconds=0.033)
        eye_image = device.receive_eyes_video_frame(timeout_seconds=0.012)

        # Let's only pass data to the visualization when all relevant streams have
        # provided a datum. This makes the visualization logic simpler.
        if not all((scene_image, eye_image, gaze, imu)):
            continue

        # Clear out the queue if it's already full.
        if data_queue.full():
            data_queue.get()

        data_queue.put(
            {
                "eye": eye_image.bgr_pixels,
                "imu": imu,
                "gaze": gaze,
            }
        )

        scene_queue.put({"scene": scene_image.bgr_pixels, "gaze": gaze})


def video_writing_loop():
    while True:
        scene_data = scene_queue.get()
        render_data = render_queue.get()

        if scene_data is not None:
            if scene_data == "stop" or render_data == "stop":
                break
            else:
                scene_img = scene_data["scene"]
                gaze = scene_data["gaze"]
                render_img = render_data["img"]
        else:
            continue

        if not hasattr(video_writing_loop, "writer"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writing_loop.writer = cv2.VideoWriter(
                "rt_imu_scene.mp4", fourcc, 29, (1600, 1200)
            )

        if not hasattr(video_writing_loop, "render_writer"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writing_loop.render_writer = cv2.VideoWriter(
                "rt_imu_eye.mp4", fourcc, 29, (1200, 1200)
            )

        cv2.circle(
            scene_img,
            (int(gaze.x), int(gaze.y)),
            radius=80,
            color=(0, 0, 255),
            thickness=15,
        )

        video_writing_loop.writer.write(scene_img)

        video_writing_loop.render_writer.write(render_img)

    video_writing_loop.writer.release()
    video_writing_loop.render_writer.release()


# ---- Main PyQtGraph app ----
class Visualization(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Eye Model Visualization")
        self.setGeometry(100, 100, 600, 600)
        self.setStyleSheet("background-color: rgb(0, 0, 0);")

        central = QWidget()
        layout = QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.threeD_eye_view = gl.GLViewWidget()
        # self.threeD_eye_view.setBackgroundColor(colors.BACKGROUND)
        self.threeD_eye_view.setBackgroundColor(colors.BLACK)
        self.threeD_eye_view.setCameraPosition(
            distance=100,
            elevation=-5,  # angle above XY plane
            azimuth=10,  # rotation around Z axis
        )
        self.threeD_eye_view.pan(dx=0, dy=-12, dz=-17)
        # self.threeD_eye_view.setAntialiasing(aa=True)
        layout.addWidget(self.threeD_eye_view, stretch=1)

        hlayout = QHBoxLayout()
        layout.addLayout(hlayout, stretch=1)

        self.imu_view = gl.GLViewWidget()
        # self.imu_view.setBackgroundColor(colors.BACKGROUND)
        self.imu_view.setBackgroundColor(colors.BLACK)
        self.imu_view.setCameraPosition(distance=3)
        # self.imu_view.setAntialiasing(aa=True)
        hlayout.addWidget(self.imu_view, stretch=1)

        self.nsew_plot = pg.plot()

        self.nsew_plot.hideButtons()
        self.nsew_plot.setMenuEnabled(enableMenu=False)

        self.nsew_plot.getAxis("bottom").setStyle(showValues=False)
        self.nsew_plot.getAxis("top").setStyle(showValues=False)
        self.nsew_plot.getAxis("right").setStyle(showValues=False)
        self.nsew_plot.getAxis("left").setStyle(showValues=False)

        self.nsew_plot.getAxis("bottom").setTicks([])
        self.nsew_plot.getAxis("left").setTicks([])
        self.nsew_plot.getAxis("top").setTicks([])
        self.nsew_plot.getAxis("right").setTicks([])

        self.nsew_plot.showAxis("bottom", show=False)
        self.nsew_plot.showAxis("left", show=False)

        self.nsew_plot.setAspectLocked(lock=True, ratio=1.0)
        self.nsew_plot.setXRange(-1.2, 1.2)
        self.nsew_plot.setYRange(-1.2, 1.2)
        self.nsew_plot.setTitle("Overhead")

        self.nsew_plot.setAntialiasing(True)

        hlayout.addWidget(self.nsew_plot, stretch=1)

        text_N = pg.TextItem(text="N", color="w", anchor=(0.0, 0.0))
        text_N.setPos(-0.1, 1.3)
        self.nsew_plot.addItem(text_N)

        text_E = pg.TextItem(text="E", color="w", anchor=(0.0, 0.0))
        text_E.setPos(1.0, 0.15)
        self.nsew_plot.addItem(text_E)

        text_S = pg.TextItem(text="S", color="w", anchor=(0.0, 0.0))
        text_S.setPos(-0.1, -1.0)
        self.nsew_plot.addItem(text_S)

        text_W = pg.TextItem(text="W", color="w", anchor=(0.0, 0.0))
        text_W.setPos(-1.3, 0.15)
        self.nsew_plot.addItem(text_W)

        self.imu_nsew_arrow = CenteredArrowItem(
            angle=0,
            tailLen=50,
            headLen=15,
            tipAngle=25,
            baseAngle=0,
            pen=pg.mkPen(color=colors.BLUE_255, width=3),
            brush=pg.mkBrush(color=colors.BLUE_255),
        )
        self.imu_nsew_arrow.setPos(0, 0)
        self.nsew_plot.addItem(self.imu_nsew_arrow)

        self.gaze_nsew_arrow = CenteredArrowItem(
            angle=0,
            tailLen=50,
            headLen=15,
            tipAngle=25,
            baseAngle=0,
            pen=pg.mkPen(color=colors.RED_255, width=3),
            brush=pg.mkBrush(color=colors.RED_255),
        )
        self.gaze_nsew_arrow.setPos(0, 0)
        self.nsew_plot.addItem(self.gaze_nsew_arrow)

        # Center and radius in pixels
        radius = 1
        circle_item = QGraphicsEllipseItem(-radius, -radius, 2 * radius, 2 * radius)
        circle_item.setPen(pg.mkPen("w", width=2))  # white border
        circle_item.setBrush(QtGui.QBrush(QtCore.Qt.transparent))  # transparent inside

        self.nsew_plot.addItem(circle_item)
        circle_item.setPos(0, 0)  # move to center in data coordinates

        self.earth_sky_plot = pg.plot()
        hlayout.addWidget(self.earth_sky_plot, stretch=1)

        self.earth_sky_plot.hideButtons()
        self.earth_sky_plot.setMenuEnabled(enableMenu=False)

        self.earth_sky_plot.getAxis("bottom").setStyle(showValues=False)
        self.earth_sky_plot.getAxis("top").setStyle(showValues=False)
        self.earth_sky_plot.getAxis("right").setStyle(showValues=False)
        self.earth_sky_plot.getAxis("left").setStyle(showValues=False)

        self.earth_sky_plot.getAxis("bottom").setTicks([])
        self.earth_sky_plot.getAxis("left").setTicks([])
        self.earth_sky_plot.getAxis("top").setTicks([])
        self.earth_sky_plot.getAxis("right").setTicks([])

        self.earth_sky_plot.showAxis("bottom", show=False)
        self.earth_sky_plot.showAxis("left", show=False)

        self.earth_sky_plot.setAspectLocked(lock=True, ratio=1.0)
        self.earth_sky_plot.setXRange(-1.2, 1.2)
        self.earth_sky_plot.setYRange(-1.2, 1.2)
        self.earth_sky_plot.setTitle("Side-profile")

        self.earth_sky_plot.setAntialiasing(True)

        text_Sky = pg.TextItem(text="Sky", color="w", anchor=(0.0, 0.0))
        text_Sky.setPos(-0.2, 1.3)
        self.earth_sky_plot.addItem(text_Sky)

        text_Earth = pg.TextItem(text="Earth", color="w", anchor=(0.0, 0.0))
        text_Earth.setPos(-0.2, -1.0)
        self.earth_sky_plot.addItem(text_Earth)

        self.imu_earth_sky_arrow = CenteredArrowItem(
            angle=0,
            tailLen=50,
            headLen=15,
            tipAngle=25,
            baseAngle=0,
            pen=pg.mkPen(color=colors.BLUE_255, width=3),
            brush=pg.mkBrush(color=colors.BLUE_255),
        )
        self.imu_earth_sky_arrow.setPos(0, 0)
        self.earth_sky_plot.addItem(self.imu_earth_sky_arrow)

        self.gaze_earth_sky_arrow = CenteredArrowItem(
            angle=0,
            tailLen=50,
            headLen=15,
            tipAngle=25,
            baseAngle=0,
            pen=pg.mkPen(color=colors.RED_255, width=3),
            brush=pg.mkBrush(color=colors.RED_255),
        )
        self.gaze_earth_sky_arrow.setPos(0, 0)
        self.earth_sky_plot.addItem(self.gaze_earth_sky_arrow)

        # Create a dotted line to represent the horizon
        self.earth_sky_plot.plot(
            [-1, 1], [0, 0], pen=pg.mkPen(color="w", width=2, style=QtCore.Qt.DotLine)
        )

        circle_item2 = QGraphicsEllipseItem(-radius, -radius, 2 * radius, 2 * radius)
        circle_item2.setPen(pg.mkPen("w", width=2))  # white border
        circle_item2.setBrush(QtGui.QBrush(QtCore.Qt.transparent))  # transparent inside

        self.earth_sky_plot.addItem(circle_item2)
        circle_item2.setPos(0, 0)  # move to center in data coordinates

        self._setup_3d_scene()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(16)

    def _setup_3d_scene(self):
        self.eye_left = ThreeDEyeModel(np.array([-30, 0, 0]), color=colors.RED)
        self.eye_left.add_to_view(self.threeD_eye_view)

        self.eye_right = ThreeDEyeModel(np.array([30, 0, 0]), color=colors.GREEN)
        self.eye_right.add_to_view(self.threeD_eye_view)

        size = 64
        dummy_texture_data = np.empty((size, size, 4), dtype=np.ubyte)
        texture_scale = 0.12

        self.eye_texture_left = gl.GLImageItem(dummy_texture_data, smooth=True)
        apply_pose_to_texture(self.eye_texture_left, 0, size, texture_scale)
        self.threeD_eye_view.addItem(self.eye_texture_left)

        self.eye_texture_right = gl.GLImageItem(dummy_texture_data, smooth=True)
        apply_pose_to_texture(self.eye_texture_right, 1, size, texture_scale)
        self.threeD_eye_view.addItem(self.eye_texture_right)

        # Load your OBJ file
        vertices, faces = load_obj("imu.obj")  # 👈 replace with your .obj path

        # Create mesh data and mesh item
        meshdata = MeshData(vertexes=vertices, faces=faces)
        self.mesh_item = GLMeshItem(
            meshdata=meshdata,
            smooth=True,  # enable Gouraud shading
            shader="shaded",  # simple default shader
            drawFaces=True,
            drawEdges=False,
            edgeColor=(1, 1, 1, 1),
        )

        # Optional: scale or rotate
        self.mesh_item.scale(1, 1, 1)
        self.mesh_item.translate(0, 0, 0)

        # Add to scene
        self.imu_view.addItem(self.mesh_item)

    def update_plot(self):
        # Grab frame
        pixmap = self.grab()
        qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)

        # Convert to numpy array
        w_img = qimg.width()
        h_img = qimg.height()
        ptr = qimg.bits()
        ptr = ptr.tobytes()
        arr = np.frombuffer(ptr, np.uint8).reshape((h_img, w_img, 4))

        # Drop alpha and convert to BGR for OpenCV
        frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        render_queue.put({"img": frame})

        imu = None
        gaze = None
        eye_img = None
        try:
            data = data_queue.get()
            eye_img = data["eye"]
            imu = data["imu"]
            gaze = data["gaze"]
        except Empty:
            pass

        if eye_img is not None:
            eye_image_left = eye_img[:, 192:, :]
            eye_image_right = eye_img[:, :192, :]

            eye_image_left_resized = cv2.resize(
                eye_image_left, (64, 64), interpolation=cv2.INTER_AREA
            )
            eye_image_left_fin = np.flipud(
                np.fliplr(np.rot90(eye_image_left_resized)))
            eye_image_left_fin = pg.functions.makeARGB(eye_image_left_fin, useRGBA=True)
            # Reduce alpha channel (make more transparent)
            eye_image_left_fin[0][..., 3] = (
                eye_image_left_fin[0][..., 3] * 0.5
            ).astype(np.uint8)

            eye_image_right_resized = cv2.resize(
                eye_image_right, (64, 64), interpolation=cv2.INTER_AREA
            )
            eye_image_right_fin = np.flipud(
                np.fliplr(np.rot90(eye_image_right_resized)))
            eye_image_right_fin = pg.functions.makeARGB(
                eye_image_right_fin, useRGBA=True
            )
            # Reduce alpha channel (make more transparent)
            eye_image_right_fin[0][..., 3] = (
                eye_image_right_fin[0][..., 3] * 0.5
            ).astype(np.uint8)

            self.eye_texture_left.setData(eye_image_left_fin[0])
            self.eye_texture_right.setData(eye_image_right_fin[0])

        # If there is gaze data available, then update the
        # ThreeDEyeModels and the optical axis plots.
        if gaze is not None:
            eye_center_left = np.array(
                [
                    gaze.eyeball_center_left_x,
                    gaze.eyeball_center_left_y,
                    gaze.eyeball_center_left_z,
                ]
            )
            optical_axis_vector_left = np.array(
                [
                    gaze.optical_axis_left_x,
                    gaze.optical_axis_left_y,
                    gaze.optical_axis_left_z,
                ]
            )
            pupil_diameter_left = gaze.pupil_diameter_left
            self.eye_left.update(
                eye_center_left,
                optical_axis_vector_left,
                pupil_diameter_left,
                gaze.eyelid_angle_top_left,
                gaze.eyelid_angle_bottom_left,
            )

            eye_center_right = np.array(
                [
                    gaze.eyeball_center_right_x,
                    gaze.eyeball_center_right_y,
                    gaze.eyeball_center_right_z,
                ]
            )
            optical_axis_vector_right = np.array(
                [
                    gaze.optical_axis_right_x,
                    gaze.optical_axis_right_y,
                    gaze.optical_axis_right_z,
                ]
            )
            pupil_diameter_right = gaze.pupil_diameter_right
            self.eye_right.update(
                eye_center_right,
                optical_axis_vector_right,
                pupil_diameter_right,
                gaze.eyelid_angle_top_right,
                gaze.eyelid_angle_bottom_right,
            )

        if imu is not None and gaze is not None:
            quaternion = imu.quaternion
            rotvec = R.from_quat(quaternion).as_rotvec()

            # Get axis–angle
            angle_rad = np.linalg.norm(rotvec)
            axis = rotvec / angle_rad if angle_rad > 1e-8 else [1, 0, 0]

            self.mesh_item.resetTransform()
            self.mesh_item.scale(1.0, 1.0, 1.0)
            self.mesh_item.rotate(np.degrees(angle_rad), *axis)

            euler = R.from_quat(quaternion).as_euler("xyz", degrees=True)

            self.imu_nsew_arrow.setStyle(angle=(-euler[2] + 90) % 360)
            self.imu_earth_sky_arrow.setStyle(angle=euler[0] % 360)

            xy = np.array([[gaze.x, gaze.y]], dtype=np.float32).reshape(-1, 1, 2)
            xy_undist = cv2.undistortPoints(xy, K, D)
            xy_undist = xy_undist.reshape(-1, 2)

            xy_homogeneous = cv2.convertPointsToHomogeneous(xy_undist).flatten()
            cartesian_gaze_in_scene = np.array(
                [xy_homogeneous[0], xy_homogeneous[1], xy_homogeneous[2]]
            )

            cartesian_gaze_in_world = gaze_3d_to_world(
                cartesian_gaze_in_scene.reshape(1, 3), [quaternion]
            )
            euler_gaze = cartesian_to_spherical_world(cartesian_gaze_in_world)

            self.gaze_nsew_arrow.setStyle(angle=(-euler_gaze[1][0] + 90) % 360)
            self.gaze_earth_sky_arrow.setStyle(angle=euler_gaze[0][0] % 360)


if __name__ == "__main__":
    # First, establish a connection to Neon.
    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        device = Device(address=FALLBACK_DEVICE_ADDRESS, port=FALLBACK_DEVICE_PORT)
    if device is None:
        raise RuntimeError("No device found.")
    print(f"Connecting to device at {device.address}:{device.port}...")
    print("Connection successful.")

    calibration = device.get_calibration()
    K = calibration.scene_camera_matrix
    D = calibration.scene_distortion_coefficients

    # Start up the data acquisition thread.
    data_acquisition_thread = threading.Thread(
        target=data_acquisition_loop, daemon=True
    )
    data_acquisition_thread.start()

    # Start up the video writing thread.
    video_writing_thread = threading.Thread(target=video_writing_loop, daemon=True)
    video_writing_thread.start()

    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)

    try:
        viz = Visualization()
        viz.show()
        app.exec()

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # No matter what happens, we need to cleanly close our connection to Neon.
        print("Cleaning up resources...")
        scene_queue.put("stop")
        render_queue.put("stop")
        device.close()
        print("Done.")
        sys.exit()
