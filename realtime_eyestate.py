# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python",
#     "numpy",
#     "pupil-labs-realtime-api",
#     "pyqtgraph",
#     "PySide6",
#     "PyOpenGL",
#     "pglive",
# ]
# ///

import sys
import threading
from queue import Empty, Full, Queue

import cv2
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pglive.sources.data_connector import DataConnector
from pglive.sources.live_axis_range import LiveAxisRange
from pglive.sources.live_plot import LiveLinePlot
from pglive.sources.live_plot_widget import LivePlotWidget
from queue import Queue, Full, Empty
from pupil_labs.realtime_api.simple import Device, discover_one_device
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QImage
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

import colors
from threeD_eye_model import (
    ThreeDEyeModel,
    apply_pose_to_texture,
    configure_eye_camera_view,
    make_eye_texture_pair,
)


# --- Configuration Constants ---
FALLBACK_DEVICE_ADDRESS: str = "10..178.36"
FALLBACK_DEVICE_PORT: int = 8080
ANIMATION_FRAME_RATE: int = 30  # FPS

# We will use a background thread to collect data
# via Neon's Real-time API. This helps to offload the data
# acquisition from the main thread, allowing for smoother
# visualization updates.

FRAME_QUEUE_MAXSIZE = 2
ENABLE_VIDEO_RECORDING = True

data_queue = Queue(maxsize=1)
scene_queue = Queue(maxsize=FRAME_QUEUE_MAXSIZE)
render_queue = Queue(maxsize=FRAME_QUEUE_MAXSIZE)


def put_latest(queue, item):
    try:
        queue.put_nowait(item)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        queue.put_nowait(item)


def data_acquisition_loop():
    while True:
        # See our Python API Documentation for more info about these two functions:
        # https://pupil-labs.github.io/pl-realtime-api/dev/
        scene_image = device.receive_scene_video_frame(timeout_seconds=0.033)
        eye_image = device.receive_eyes_video_frame(timeout_seconds=0.012)
        gaze = device.receive_gaze_datum(timeout_seconds=0.012)

        # Let's only pass data to the visualization when all relevant streams have
        # provided a datum. This makes the visualization logic simpler.
        if not all((scene_image, eye_image, gaze)):
            continue

        put_latest(
            data_queue,
            {
                "eye": eye_image.bgr_pixels,
                "gaze": gaze,
            },
        )

        if ENABLE_VIDEO_RECORDING:
            put_latest(scene_queue, {"scene": scene_image.bgr_pixels, "gaze": gaze})


def video_writing_loop():
    try:
        while True:
            scene_data = scene_queue.get()
            if scene_data == "stop":
                break

            render_data = render_queue.get()
            if render_data == "stop":
                break

            if not scene_data or not render_data:
                continue

            scene_img = scene_data["scene"]
            gaze = scene_data["gaze"]
            render_img = render_data["img"]

            if not hasattr(video_writing_loop, "writer"):
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writing_loop.writer = cv2.VideoWriter(
                    "rt_eyestate_scene.mp4", fourcc, 29, (1600, 1200)
                )

            if not hasattr(video_writing_loop, "render_writer"):
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writing_loop.render_writer = cv2.VideoWriter(
                    "rt_eyestate_eye.mp4", fourcc, 29, (1200, 1200)
                )

            cv2.circle(
                scene_img,
                (int(gaze.x), int(gaze.y)),
                radius=35,
                color=(0, 0, 255),
                thickness=8,
            )

            video_writing_loop.writer.write(scene_img)

            video_writing_loop.render_writer.write(render_img)
    finally:
        if hasattr(video_writing_loop, "writer") and video_writing_loop.writer is not None:
            video_writing_loop.writer.release()
        if hasattr(video_writing_loop, "render_writer") and video_writing_loop.render_writer is not None:
            video_writing_loop.render_writer.release()


def cartesian_to_spherical(vec):
    x = vec[0]
    y = vec[1]
    z = vec[2]

    radii = np.sqrt(x**2 + y**2 + z**2)

    elevation = -(np.arccos(z / radii) - np.pi / 2)
    azimuth = np.arctan2(y, x) - np.pi / 2

    return azimuth, elevation


def closest_points_between_rays(p1, d1, p2, d2, eps=1e-8):
    """
    Given two rays (p1 + s * d1) and (p2 + t * d2) with d1,d2 as direction vectors
    (need not be normalized), compute the closest points on each infinite line,
    their midpoint, the separation distance, and the vergence angle (deg)
    based on the midpoint.

    Args:
        p1, p2: array-like, shape (3,) - origins of the two rays (eye positions)
        d1, d2: array-like, shape (3,) - direction vectors of the optical axes
        eps: float - small threshold for handling near-parallel lines

    Returns:
        dict with keys:
          'p_closest_1' : closest point on line1 (3,)
          'p_closest_2' : closest point on line2 (3,)
          'midpoint'    : (p_closest_1 + p_closest_2) / 2
          'separation'  : ||p_closest_1 - p_closest_2|| (distance between the two closest points)
          'vergence_deg' : vergence angle in degrees (angle between gaze vectors to midpoint)
          's' : parameter along line1 (p1 + s*d1)
          't' : parameter along line2 (p2 + t*d2)
          'parallel' : True if lines are (nearly) parallel
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)

    # normalize directions to avoid scale issues
    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    if n1 == 0 or n2 == 0:
        raise ValueError("Direction vectors must be non-zero")
    u = d1 / n1
    v = d2 / n2
    w0 = p1 - p2

    a = np.dot(u, u)  # =1
    b = np.dot(u, v)
    c = np.dot(v, v)  # =1
    e = np.dot(u, w0)
    f = np.dot(v, w0)

    denom = a * c - b * b

    result = {}
    if abs(denom) < eps:
        # Lines are nearly parallel. Choose s by projecting w0 onto u,
        # and set t so the point on line2 is the projection of that point onto line2.
        result["parallel"] = True
        s = e / a
        pt1 = p1 + s * u
        # project vector (pt1 - p2) onto v to find t
        t = np.dot(pt1 - p2, v) / 1.0
        pt2 = p2 + t * v
    else:
        result["parallel"] = False
        s = (b * f - c * e) / denom
        t = (a * f - b * e) / denom
        pt1 = p1 + s * u
        pt2 = p2 + t * v

    midpoint = 0.5 * (pt1 + pt2)
    sep = np.linalg.norm(pt1 - pt2)

    # compute vergence: vectors from each eye to midpoint
    v1 = midpoint - p1
    v2 = midpoint - p2
    nv1 = np.linalg.norm(v1)
    nv2 = np.linalg.norm(v2)
    if nv1 == 0 or nv2 == 0:
        vergence_deg = 0.0
    else:
        cosang = np.clip(np.dot(v1, v2) / (nv1 * nv2), -1.0, 1.0)
        vergence_deg = np.degrees(np.arccos(cosang))

    result.update(
        {
            "p_closest_1": pt1,
            "p_closest_2": pt2,
            "midpoint": midpoint,
            "separation": sep,
            "vergence_deg": vergence_deg,
            "s": s,
            "t": t,
        }
    )
    return result


class Visualization(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Eye State Visualization")
        self.setGeometry(100, 100, 600, 600)
        self.setStyleSheet("background-color: rgb(0, 0, 0);")

        # --- Central widget ---
        central = QWidget()
        layout = QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # --- 3D OpenGL view (left) ---
        self.view = configure_eye_camera_view()
        layout.addWidget(self.view, stretch=2)

        my_font = QFont("Arial", 12)

        # Bottom plot
        self.plot_widget = LivePlotWidget(
            title="Vergence Angle [deg]",
            y_range_controller=LiveAxisRange(fixed_range=[0, 40]),
        )

        self.plot_widget.hideButtons()
        self.plot_widget.setMenuEnabled(enableMenu=False)

        self.plot_widget.getAxis("bottom").setStyle(showValues=False)
        self.plot_widget.getAxis("top").setStyle(showValues=False)
        self.plot_widget.getAxis("right").setStyle(showValues=False)
        self.plot_widget.getAxis("left").setStyle(showValues=False)

        self.plot_widget.getAxis("bottom").setTicks([])
        self.plot_widget.getAxis("left").setTicks([])

        self.plot_widget.getAxis("left").label.setFont(my_font)

        self.plot_widget.setAntialiasing(True)

        self.plot_curve = LiveLinePlot(pen=pg.mkPen(color=colors.PURPLE_255, width=3))

        self.plot_widget.addItem(self.plot_curve)

        self.data_connector = DataConnector(
            self.plot_curve, max_points=300, update_rate=200
        )
        layout.addWidget(self.plot_widget, stretch=1)

        # Timer for updating plot
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(16)

        self._setup_3d_scene()

    def _setup_3d_scene(self):
        self.eye_left = ThreeDEyeModel(np.array([-30, 0, 0]), color=colors.RED)
        self.eye_left.add_to_view(self.view)

        self.eye_right = ThreeDEyeModel(np.array([30, 0, 0]), color=colors.GREEN)
        self.eye_right.add_to_view(self.view)

        size = 84
        dummy_texture_data = np.empty((size, size, 4), dtype=np.ubyte)

        checkerboard = np.indices((size, size)).sum(axis=0) % 2
        dummy_texture_data[checkerboard == 0] = (0, 0, 0, 255)
        dummy_texture_data[checkerboard == 1] = (255, 255, 255, 255)

        texture_scale = 0.16

        self.eye_texture_left = gl.GLImageItem(dummy_texture_data, smooth=True)
        apply_pose_to_texture(self.eye_texture_left, 0, size, texture_scale)
        self.view.addItem(self.eye_texture_left)

        self.eye_texture_right = gl.GLImageItem(dummy_texture_data, smooth=True)
        apply_pose_to_texture(self.eye_texture_right, 1, size, texture_scale)
        self.view.addItem(self.eye_texture_right)

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
        if ENABLE_VIDEO_RECORDING:
            put_latest(render_queue, {"img": frame})

        eye_img = None
        gaze = None
        try:
            data = data_queue.get()
            eye_img = data["eye"]
            gaze = data["gaze"]
        except Empty:
            pass

        if eye_img is not None:
            eye_texture_left, eye_texture_right = make_eye_texture_pair(
                eye_img,
                size=84,
                alpha_scale=0.75,
            )
            self.eye_texture_left.setData(eye_texture_left)
            self.eye_texture_right.setData(eye_texture_right)

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

            p1 = eye_center_left
            p2 = eye_center_right
            d1 = optical_axis_vector_left
            d2 = optical_axis_vector_right
            result = closest_points_between_rays(p1, d1, p2, d2, eps=1e-8)

            self.data_connector.cb_append_data_point(result["vergence_deg"])


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

    # Start up the video writing thread only when recording is enabled.
    video_writing_thread = None
    if ENABLE_VIDEO_RECORDING:
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
        # No matter what happens, we need to cleanly close our connection to Neon
        # and close all OpenCV windows.
        print("Cleaning up resources...")
        if ENABLE_VIDEO_RECORDING and video_writing_thread is not None:
            put_latest(scene_queue, "stop")
            put_latest(render_queue, "stop")
            video_writing_thread.join(timeout=5.0)
        device.close()
        cv2.destroyAllWindows()
        print("Done.")
        sys.exit()
