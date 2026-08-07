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
from queue import Empty, Queue

import cv2
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pglive.sources.data_connector import DataConnector
from pglive.sources.live_axis_range import LiveAxisRange
from pglive.sources.live_plot import LiveLinePlot
from pglive.sources.live_plot_widget import LivePlotWidget
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
from threeD_eye_model import ThreeDEyeModel, apply_pose_to_texture


# --- Configuration Constants ---
FALLBACK_DEVICE_ADDRESS: str = "192.168.178.36"
FALLBACK_DEVICE_PORT: int = 8080
ANIMATION_FRAME_RATE: int = 30  # FPS

# We will use a background thread to collect data
# via Neon's Real-time API. This helps to offload the data
# acquisition from the main thread, allowing for smoother
# visualization updates.

data_queue = Queue(maxsize=1)
scene_queue = Queue()
render_queue = Queue()


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
            # if not all((eye_image, gaze)):
            continue

        # Clear out the queue if it's already full.
        if data_queue.full():
            data_queue.get()

        data_queue.put(
            {
                "eye": eye_image.bgr_pixels,
                "gaze": gaze,
                # "scene": scene_image.bgr_pixels,
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
                "rt_pupil_scene.mp4", fourcc, 29, (1600, 1200)
            )

        if not hasattr(video_writing_loop, "render_writer"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writing_loop.render_writer = cv2.VideoWriter(
                "rt_pupil_eye.mp4", fourcc, 29, (1200, 1200)
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


# Now, we make class to hold all the elements of the matplotlib figure.
# This makes it easier to organize the visualization logic later.
# The figure will have three sections:
# - A space at the top to display the current eye image.
# - A space in the middle to display the animated 3D eye model.
# - A space at the bottom for plots that will show the optical axis vectors, for quick inspection
#   of things, like vergence angle.


class Visualization(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Eye Model Visualization")
        self.setGeometry(100, 100, 600, 600)
        self.setStyleSheet("background-color: rgb(0, 0, 0);")

        # --- Central widget ---
        central = QWidget()
        layout = QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # --- 3D OpenGL view (left) ---
        self.view = gl.GLViewWidget()
        # self.view.setBackgroundColor(colors.BACKGROUND)
        self.view.setBackgroundColor(colors.BLACK)
        self.view.setCameraPosition(
            distance=100,
            elevation=-5,  # angle above XY plane
            azimuth=10,  # rotation around Z axis
        )
        self.view.pan(dx=0, dy=-12, dz=-17)
        # self.view.setAntialiasing(aa=True)
        layout.addWidget(self.view, stretch=2)

        my_font = QFont("Arial", 12)

        # Bottom plot
        self.plot_widget = LivePlotWidget(
            title="Pupil Diameter [mm]",
            y_range_controller=LiveAxisRange(fixed_range=[0, 12]),
            # labels={"left": ("Degrees")},
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
        # self.plot_widget.titleLabel.item.setFont(my_font)

        self.plot_widget.setAntialiasing(True)

        self.plot_curve_left = LiveLinePlot(pen=pg.mkPen(color=colors.RED_255, width=3))
        self.plot_curve_right = LiveLinePlot(
            pen=pg.mkPen(color=colors.GREEN_255, width=3)
        )

        self.plot_widget.addItem(self.plot_curve_left)
        self.plot_widget.addItem(self.plot_curve_right)

        self.data_connector_left = DataConnector(
            self.plot_curve_left, max_points=300, update_rate=200
        )
        self.data_connector_right = DataConnector(
            self.plot_curve_right, max_points=300, update_rate=200
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

        size = 64
        dummy_texture_data = np.empty((size, size, 4), dtype=np.ubyte)

        checkerboard = np.indices((size, size)).sum(axis=0) % 2
        dummy_texture_data[checkerboard == 0] = (0, 0, 0, 255)
        dummy_texture_data[checkerboard == 1] = (255, 255, 255, 255)

        texture_scale = 0.25

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
        render_queue.put({"img": frame})

        # self.view.setBackgroundColor(colors.BACKGROUND)

        eye_img = None
        gaze = None
        # scene_img = None
        try:
            data = data_queue.get()
            eye_img = data["eye"]
            gaze = data["gaze"]
            # scene_img = data["scene"]
        except Empty:
            pass

        # If there is a scene image and gaze data available, then draw a circle
        # at the gaze point and display the resulting image via OpenCV's imshow function.
        # if scene_img is not None and gaze is not None:
        # if gaze is not None:
        # key = cv2.waitKey(1) & 0xFF
        # if key == 27:
        # cv2.destroyAllWindows()

        # cv2.circle(
        #     scene_img,
        #     (int(gaze.x), int(gaze.y)),
        #     radius=80,
        #     color=(0, 0, 255),
        #     thickness=15,
        # )

        # cv2.imshow("Scene Camera + Gaze Overlay - Press ESC to quit", scene_img)

        if eye_img is not None:
            eye_image_left = eye_img[:, 192:, :]
            eye_image_right = eye_img[:, :192, :]

            eye_image_left_resized = cv2.resize(
                eye_image_left, (64, 64), interpolation=cv2.INTER_AREA
            )
            eye_image_left_fin = np.flipud(
                np.flipud(np.fliplr(np.rot90(eye_image_left_resized)))
            )
            eye_image_left_fin = pg.functions.makeARGB(eye_image_left_fin, useRGBA=True)
            # Reduce alpha channel (make more transparent)
            eye_image_left_fin[0][..., 3] = (
                eye_image_left_fin[0][..., 3] * 0.5
            ).astype(np.uint8)

            eye_image_right_resized = cv2.resize(
                eye_image_right, (64, 64), interpolation=cv2.INTER_AREA
            )
            eye_image_right_fin = np.flipud(
                np.flipud(np.fliplr(np.rot90(eye_image_right_resized)))
            )
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

            self.data_connector_left.cb_append_data_point(pupil_diameter_left)
            self.data_connector_right.cb_append_data_point(pupil_diameter_right)


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
        # No matter what happens, we need to cleanly close our connection to Neon
        # and close all OpenCV windows.
        print("Cleaning up resources...")
        scene_queue.put("stop")
        render_queue.put("stop")
        device.close()
        cv2.destroyAllWindows()
        print("Done.")
        sys.exit()
