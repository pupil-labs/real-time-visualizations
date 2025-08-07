import threading
import time
from queue import Queue

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pupil_labs.eye_state_filtering import BinocularEyeStateFilter, cart2sph, sph2cart
from pupil_labs.realtime_api.simple import Device
from pupil_labs.realtime_api.streaming.eye_events import (
    BlinkEventData,
    FixationEventData,
    FixationOnsetEventData,
)
from scipy.spatial.transform import Rotation as R

device = Device(address="192.168.1.57", port=8080)
# device = Device(address="172.20.10.2", port=8080)


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


eye_state_filter = BinocularEyeStateFilter()


def extract_eye_state_from_realtime_api(sample):
    # left
    E0 = [
        sample.eyeball_center_left_x,
        sample.eyeball_center_left_y,
        sample.eyeball_center_left_z,
    ]
    O0 = np.asarray(
        [
            sample.optical_axis_left_x,
            sample.optical_axis_left_y,
            sample.optical_axis_left_z,
        ]
    )
    phi0, theta0 = cart2sph(O0)[0]
    P_left = sample.pupil_diameter_left

    EL0 = [
        sample.eyelid_angle_bottom_left,
        sample.eyelid_angle_top_left,
        sample.eyelid_aperture_left,
    ]

    # right
    E1 = [
        sample.eyeball_center_right_x,
        sample.eyeball_center_right_y,
        sample.eyeball_center_right_z,
    ]
    O1 = np.asarray(
        [
            sample.optical_axis_right_x,
            sample.optical_axis_right_y,
            sample.optical_axis_right_z,
        ]
    )
    phi1, theta1 = cart2sph(O1)[0]
    P_right = sample.pupil_diameter_right

    EL1 = [
        sample.eyelid_angle_bottom_right,
        sample.eyelid_angle_top_right,
        sample.eyelid_aperture_right,
    ]

    return np.asarray(
        [
            *E0,
            phi0,
            theta0,
            P_left,
            *EL0,
            *E1,
            phi1,
            theta1,
            P_right,
            *EL1,
        ]
    )


class ThreeDEye:
    reference_vec = np.array([0, 1, 0])  # Forward direction

    def __init__(self, scale=12):
        self.scale = scale

        self.arrow = None

        # dummy default values for initializing & testing
        self.eye_center = np.array([0.0, 0.0, 0.0])
        self.optical_axis = np.array([0.0, 30.0, 0.0])
        self.pupil_diameter = 4.0
        self.eyelid_top_angle = 0.2
        self.eyelid_bottom_angle = -0.2

        x0, y0, z0 = self._generate_eye_sphere()

        # the generated sphere opening is by default oriented downwards, along the y-axis.
        # so, we rotate it to have the front of the eye facing forward along the z-axis.
        self.rot_x = R.from_euler("x", 90, degrees=True)
        x0_r, y0_r, z0_r = self._correct_initial_eye_rotation(x0, y0, z0, self.rot_x)

        self._base_eye_sphere = (x0_r, y0_r, z0_r)

        X, Y, Z = self._generate_pupil_disk()
        self._base_pupil_disk = (X, Y, Z)

        x, y, z = self._generate_eye_lid()
        self._base_bottom_eye_lid = (x, y, z)

        x, y, z = self._generate_eye_lid()
        self._base_top_eye_lid = (x, y, z)

        self._update_eye()

    def update_eye(
        self,
        eye_center,
        optical_axis,
        pupil_diameter,
        eyelid_top_angle,
        eyelid_bottom_angle,
    ):
        self.eye_center = eye_center

        # match with matplotlib conventions
        self.eye_center[1] *= -1
        self.eye_center[2] *= -1

        # match with matplotlib conventions
        self.optical_axis = np.array(
            [optical_axis[0], optical_axis[2], optical_axis[1]]
        )

        self.pupil_diameter = pupil_diameter

        self.eyelid_top_angle = eyelid_top_angle
        self.eyelid_bottom_angle = eyelid_bottom_angle

        self._update_eye()

    def _update_eye(self):
        self._compute_eye_rotation()
        self._rotate_eye_to_optic_axis()
        self._transfrom_pupil_disk()
        self.transformed_top_eye_lid = self._rotate_eye_lid(
            self._base_top_eye_lid, self.eyelid_top_angle
        )
        self.transformed_bottom_eye_lid = self._rotate_eye_lid(
            self._base_bottom_eye_lid, self.eyelid_bottom_angle
        )

    def _generate_eye_sphere(self, res=20):
        u = np.linspace(0, 2 * np.pi, res)
        v = np.linspace(0, np.pi - np.pi / 8, res)
        x = self.scale * np.outer(np.cos(u), np.sin(v))
        y = self.scale * np.outer(np.sin(u), np.sin(v))
        z = self.scale * np.outer(np.ones_like(u), np.cos(v))
        return x, y, z

    def _correct_initial_eye_rotation(self, x, y, z, rot):
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        rotated = rot.apply(pts)
        x_r, y_r, z_r = rotated[:, 0], rotated[:, 1], rotated[:, 2]
        return x_r.reshape(x.shape), y_r.reshape(y.shape), z_r.reshape(z.shape)

    def _compute_eye_rotation(self):
        axis_x, axis_y, axis_z = self.optical_axis

        axis_r = np.linalg.norm(self.optical_axis)
        azimuth = np.arctan2(axis_y, axis_x)  # rotation about z axis
        elevation = np.arcsin(axis_z / axis_r)  # rotation about y axis

        azimuth -= np.pi / 2  # 0 azimuth is forward for Neon

        self.optic_rot = R.from_euler("zx", [azimuth, elevation], degrees=False)

    def _rotate_eye_to_optic_axis(self):
        (x0_r, y0_r, z0_r) = self._base_eye_sphere

        x_rot, y_rot, z_rot = self._correct_initial_eye_rotation(
            x0_r, y0_r, z0_r, self.optic_rot
        )
        x_rot += self.eye_center[0]
        y_rot += self.eye_center[1]
        z_rot += self.eye_center[2]
        self.transformed_eye_sphere = (x_rot, y_rot, z_rot)

    def _generate_pupil_disk(self, resolution=15):
        r = np.linspace(0, 1.0, resolution)
        theta = np.linspace(0, 2 * np.pi, resolution)
        R, T = np.meshgrid(r, theta)

        X = R * np.cos(T)
        Y = np.zeros_like(X)  # flat disk in XZ plane (y = 0)
        Z = R * np.sin(T)

        return X, Y, Z

    def _transfrom_pupil_disk(self):
        (X, Y, Z) = self._base_pupil_disk

        # first, scale the pupil disk by pupil diameter
        X_s = X * (self.pupil_diameter / 2)
        Y_s = Y * (self.pupil_diameter / 2)
        Z_s = Z * (self.pupil_diameter / 2)

        X_t, Y_t, Z_t = self._rotate_and_shift_pupil_disk(
            X_s,
            Y_s,
            Z_s,
            rot_matrix=(self.optic_rot).as_matrix(),
            center=self.eye_center,
        )
        self.transformed_pupil_disk = (X_t, Y_t, Z_t)

    def _rotate_and_shift_pupil_disk(
        self, X, Y, Z, rot_matrix=None, center=np.array([0, 0, 0])
    ):
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        # the pupil sits on the surface of the eye sphere
        pts += [0.0, self.scale + 2.0, 0.0]

        # rotate according to the optic axis rotation
        pts = pts @ rot_matrix.T

        # shift it to where the eye is centered
        pts += center

        X_t = pts[:, 0].reshape(X.shape)
        Y_t = pts[:, 1].reshape(Y.shape)
        Z_t = pts[:, 2].reshape(Z.shape)

        return X_t, Y_t, Z_t

    def _generate_eye_lid(self):
        # Tube along circular arc
        theta = np.linspace(0, np.pi, 25)  # curve angle
        R = self.scale + 1.5  # major radius (curve path)
        r = 0.45  # tube radius (thickness)
        n_circle = 8  # resolution of circular cross-section

        # Cross-section circle angles
        phi = np.linspace(0, 2 * np.pi, n_circle)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Allocate arrays
        X = np.zeros((n_circle, len(theta)))
        Y = np.zeros_like(X)
        Z = np.zeros_like(X)

        # Build tube by sweeping the cross-section
        for i, t in enumerate(theta):
            # Center point on the arc
            cx = R * np.cos(t)
            cy = R * np.sin(t)
            cz = 0

            # Tangent vector (derivative of arc)
            tx = -R * np.sin(t)
            ty = R * np.cos(t)
            tz = 0
            tangent = np.array([tx, ty, tz])
            tangent /= np.linalg.norm(tangent)

            # Normal vector (pointing in Z)
            normal = np.array([0, 0, 1])

            # Binormal = tangent × normal
            binormal = np.cross(tangent, normal)

            # Build circular cross-section at this position
            for j in range(n_circle):
                X[j, i] = cx + r * (cos_phi[j] * normal[0] + sin_phi[j] * binormal[0])
                Y[j, i] = cy + r * (cos_phi[j] * normal[1] + sin_phi[j] * binormal[1])
                Z[j, i] = cz + r * (cos_phi[j] * normal[2] + sin_phi[j] * binormal[2])

        return X, Y, Z

    def _rotate_eye_lid(self, eye_lid_pts, eye_lid_angle):
        (x, y, z) = eye_lid_pts

        rot = R.from_euler("x", -eye_lid_angle).as_matrix()
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        rotated = pts @ rot.T

        X_rot = rotated[:, 0].reshape(x.shape)
        Y_rot = rotated[:, 1].reshape(y.shape)
        Z_rot = rotated[:, 2].reshape(z.shape)

        Y_rot += 9.0

        X_rot += self.eye_center[0]
        Y_rot += self.eye_center[1]
        Z_rot += self.eye_center[2]

        return (X_rot, Y_rot, Z_rot)

    def plot(self, ax, optic_axis_color):
        self._plot_eye_and_pupil(ax)
        self._plot_optic_axis(ax, optic_axis_color)
        self._plot_eye_lids(ax)

    def _plot_eye_and_pupil(self, ax):
        ax.plot_wireframe(
            self.transformed_eye_sphere[0],
            self.transformed_eye_sphere[1],
            self.transformed_eye_sphere[2],
            color="darkgray",
        )

        ax.plot_surface(
            self.transformed_pupil_disk[0],
            self.transformed_pupil_disk[1],
            self.transformed_pupil_disk[2],
            color="black",
            shade=False,
        )

    def _plot_optic_axis(self, ax, color):
        if self.arrow in ax.artists:
            self.arrow.remove()

        self.arrow = ax.quiver(
            self.eye_center[0],
            self.eye_center[1],
            self.eye_center[2],
            *self.optical_axis,
            color=color,
            length=30.0,
            normalize=True,
        )

    def _plot_eye_lids(self, ax):
        x, y, z = self.transformed_top_eye_lid
        ax.plot_surface(
            x,
            y,
            z,
            color="white",
            linewidth=2,
            rstride=1,
            cstride=1,
            shade=False,
        )

        x, y, z = self.transformed_bottom_eye_lid
        ax.plot_surface(
            x,
            y,
            z,
            color="white",
            linewidth=2,
            rstride=1,
            cstride=1,
            shade=False,
        )


# Setup plot
fig = plt.figure(figsize=(6, 6))
fig.canvas.manager.set_window_title("3D Eye State Visualization")
fig.patch.set_facecolor("black")

eyestate_ax = fig.add_subplot(111, projection="3d")
eyestate_ax.set_box_aspect([1, 1, 1])
eyestate_ax.set_xlim([-40, 40])
eyestate_ax.set_ylim([-40, 60])
eyestate_ax.set_zlim([50, 10])
eyestate_ax.set_xlabel("X")
eyestate_ax.set_ylabel("Z")
eyestate_ax.set_zlabel("Y")

eyestate_ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
eyestate_ax.grid(False)
eyestate_ax.set_axis_off()

eyestate_ax.set_facecolor("black")

eyestate_ax.view_init(elev=12, azim=90)

inset_ax_eye_image = inset_axes(
    eyestate_ax, width="70%", height="28%", loc="upper right"
)
inset_ax_eye_image.axis("off")
eye_image_display = inset_ax_eye_image.imshow(
    np.zeros((192, int((192 * 2) * 1.2), 3), dtype=np.uint8)
)  # placeholder


inset_ax_blinks = inset_axes(eyestate_ax, width="30%", height="28%", loc="upper left")
inset_ax_blinks.set_facecolor("black")
inset_ax_blinks.set_xlim(0, 100)
inset_ax_blinks.set_ylim(0, 1.3)
inset_ax_blinks.set(xticklabels=[], yticklabels=[])
inset_ax_blinks.grid(False)
inset_ax_blinks.text(
    0.05,
    0.95,
    "Blinks",
    color="white",
    fontsize=10,
    transform=inset_ax_blinks.transAxes,
    va="top",
)
inset_ax_blinks.tick_params(colors="black")
for spine in inset_ax_blinks.spines.values():
    spine.set_edgecolor("white")

(plot_blinks,) = inset_ax_blinks.plot([], [], linewidth=1.5)
plot_blinks.set_color((99 / 255, 108 / 255, 191 / 255))

inset_ax_pupil_diameter = inset_axes(
    eyestate_ax, width="46%", height="22%", loc="lower left"
)
inset_ax_pupil_diameter.set_facecolor("black")
inset_ax_pupil_diameter.set_xlim(0, 100)
inset_ax_pupil_diameter.set_ylim(0, 8.0)
inset_ax_pupil_diameter.set(xticklabels=[], yticklabels=[])
inset_ax_pupil_diameter.grid(False)
inset_ax_pupil_diameter.text(
    0.02,
    0.90,
    "Pupil Diameter [mm]",
    color="white",
    fontsize=10,
    transform=inset_ax_pupil_diameter.transAxes,
    va="top",
)
inset_ax_pupil_diameter.tick_params(colors="black")
for spine in inset_ax_pupil_diameter.spines.values():
    spine.set_edgecolor("white")

(plot_left_pupil,) = inset_ax_pupil_diameter.plot([], [], linewidth=1.5)
plot_left_pupil.set_color((101 / 255, 188 / 255, 118 / 255))

(plot_right_pupil,) = inset_ax_pupil_diameter.plot([], [], linewidth=1.5)
plot_right_pupil.set_color((229 / 255, 111 / 255, 114 / 255))

inset_ax_optaxes_xz = inset_axes(
    eyestate_ax, width="25%", height="22%", loc="lower right"
)
inset_ax_optaxes_xz.set_facecolor("black")
inset_ax_optaxes_xz.set_xlim(-55, 55)
inset_ax_optaxes_xz.set_ylim(-10, 180)
inset_ax_optaxes_xz.set(xticklabels=[], yticklabels=[])
inset_ax_optaxes_xz.grid(False)
inset_ax_optaxes_xz.text(
    0.05,
    0.15,
    "Optical Axes (X,Z)",
    color="white",
    fontsize=10,
    transform=inset_ax_optaxes_xz.transAxes,
    va="top",
)
inset_ax_optaxes_xz.tick_params(colors="black")
for spine in inset_ax_optaxes_xz.spines.values():
    spine.set_edgecolor("white")

plot_axes_left_xz = inset_ax_optaxes_xz.quiver(
    -2,
    -4,
    0,
    0,
    color=(101 / 255, 188 / 255, 118 / 255),
    scale=1,
    scale_units="xy",
    angles="xy",
    width=0.01,
)

plot_axes_right_xz = inset_ax_optaxes_xz.quiver(
    2,
    -4,
    0,
    0,
    color=(229 / 255, 111 / 255, 114 / 255),
    scale=1,
    scale_units="xy",
    angles="xy",
    width=0.01,
)

inset_ax_optaxes_zy = fig.add_axes([0.78 - 0.3, 0.1 / 2.75, 0.24, 0.2095])
# inset_ax_optaxes_zy = inset_axes(eyestate_ax, width="30%", height="22%", loc="lower right")
inset_ax_optaxes_zy.set_facecolor("black")
inset_ax_optaxes_zy.set_xlim(25, 100)
inset_ax_optaxes_zy.set_ylim(10, -35)
inset_ax_optaxes_zy.set(xticklabels=[], yticklabels=[])
inset_ax_optaxes_zy.grid(False)
inset_ax_optaxes_zy.text(
    0.05,
    0.15,
    "Optical Axes (Z,Y)",
    color="white",
    fontsize=10,
    transform=inset_ax_optaxes_zy.transAxes,
    va="top",
)
inset_ax_optaxes_zy.tick_params(colors="black")
for spine in inset_ax_optaxes_zy.spines.values():
    spine.set_edgecolor("white")

plot_axes_left_zy = inset_ax_optaxes_zy.quiver(
    -2,
    -4,
    0,
    0,
    color=(101 / 255, 188 / 255, 118 / 255),
    scale=1,
    scale_units="xy",
    angles="xy",
    width=0.01,
)

plot_axes_right_zy = inset_ax_optaxes_zy.quiver(
    2,
    -4,
    0,
    0,
    color=(229 / 255, 111 / 255, 114 / 255),
    scale=1,
    scale_units="xy",
    angles="xy",
    width=0.01,
)


data_queue = Queue(maxsize=1)
signal_data = []

filtering = True


def data_acquisition_loop():
    while True:
        scene_image = device.receive_scene_video_frame(timeout_seconds=0.01)
        eye_image = device.receive_eyes_video_frame(timeout_seconds=0.01)
        gaze = device.receive_gaze_datum(timeout_seconds=0.01)
        eye_event = device.receive_eye_events(timeout_seconds=0.01)
        imu = device.receive_imu_datum()

        if eye_image is not None and gaze is not None and scene_image is not None:
            scene_image = scene_image.bgr_pixels
            eye_image = eye_image.bgr_pixels

            eye_image = cv2.resize(
                eye_image, (int(eye_image.shape[1] * 1.2), eye_image.shape[0])
            )

            if filtering:
                eye_state = extract_eye_state_from_realtime_api(gaze)
                timestamp = gaze.timestamp_unix_seconds * 1e9
                eye_state_filtered = eye_state_filter.filter(eye_state, timestamp)
            else:
                eye_state_filtered = None

            if data_queue.full():
                data_queue.get_nowait()

            data_queue.put_nowait(
                {
                    "eye": eye_image,
                    "scene": scene_image,
                    "gaze": gaze,
                    "eye_event": eye_event,
                    "eye_state_filtered": eye_state_filtered,
                    "imu": imu,
                }
            )


data_acquisition_thread = threading.Thread(target=data_acquisition_loop, daemon=True)
data_acquisition_thread.start()

blink_data = []
blinked = False
blink_time = 0.0

fixating = False

left_pupil_data = []
right_pupil_data = []

eye_left = ThreeDEye()
eye_right = ThreeDEye()


def update(frame_number):
    global blink_data
    global blinked
    global blink_time

    global fixating

    global left_pupil_data
    global right_pupil_data

    global plot_axes_left_xz
    global plot_axes_right_xz
    global plot_axes_left_zy
    global plot_axes_right_zy

    global filtering

    frame = None
    gaze = None
    eye_event = None
    if not data_queue.empty():
        data = data_queue.get()
        frame = data["eye"]
        scene_frame = data["scene"]
        gaze = data["gaze"]
        eye_event = data["eye_event"]
        eye_state_filtered = data["eye_state_filtered"]
        quaternion = data["imu"].quaternion

    if frame is not None and scene_frame is not None:
        if isinstance(eye_event, FixationEventData) and eye_event.event_type == 1:
            fixating = False
        elif (
            isinstance(eye_event, FixationOnsetEventData) and eye_event.event_type == 3
        ):
            fixating = True

        if fixating:
            cv2.rectangle(
                scene_frame,
                (int(gaze.x) - 40, int(gaze.y) - 40),
                (int(gaze.x) + 40, int(gaze.y) + 40),
                (255, 0, 0),
                thickness=8,
            )
        else:
            cv2.circle(
                scene_frame,
                (int(gaze.x), int(gaze.y)),
                radius=80,
                color=(0, 0, 255),
                thickness=15,
            )

        heading_in_world = imu_heading_in_world(quaternion)

        earth_sky_heading = np.array([heading_in_world[1], heading_in_world[2]])
        earth_sky_heading /= np.linalg.norm(earth_sky_heading)

        # Draw a black rectangle in the bottom right corner of the scene_frame
        rect_height = 320
        rect_width = 320
        y1 = scene_frame.shape[0] - rect_height
        y2 = scene_frame.shape[0]
        x1 = scene_frame.shape[1] - rect_width * 2
        x2 = scene_frame.shape[1] - rect_width
        overlay = scene_frame.copy()
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),
            thickness=-1,
        )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, scene_frame, 1 - alpha, 0, scene_frame)

        # Draw a white ring of radius 150 in the black rectangle
        ring_center = (x1 + rect_width // 2, y1 + rect_height // 2)
        cv2.circle(
            scene_frame,
            ring_center,
            115,
            (255, 255, 255),
            thickness=6,
        )

        # Draw Sky, Earth at the edge of the circle
        directions = {
            "Sky": (0, -1),
            "Earth": (0, 1),
        }

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        for label, (dx, dy) in directions.items():
            text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            text_x = int(ring_center[0] + dx * 138 - text_size[0] // 2)
            text_y = int(ring_center[1] + dy * 138 + text_size[1] // 2)
            cv2.putText(
                scene_frame,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

        # Draw an arrow representing the earth_sky_heading direction on the scene_frame
        arrow_length = 108  # pixels
        center_x = int(x1 + rect_width // 2)
        center_y = int(y1 + rect_height // 2)
        end_x = int(center_x + earth_sky_heading[0] * arrow_length)
        end_y = int(center_y + -earth_sky_heading[1] * arrow_length)
        cv2.arrowedLine(
            scene_frame,
            (center_x, center_y),
            (end_x, end_y),
            color=(255, 0, 0),
            thickness=5,
            tipLength=0.23,
        )

        # Draw a black rectangle in the bottom right corner of the scene_frame
        rect_height = 320
        rect_width = 320
        y1 = scene_frame.shape[0] - rect_height
        y2 = scene_frame.shape[0]
        x1 = scene_frame.shape[1] - rect_width
        x2 = scene_frame.shape[1]
        overlay = scene_frame.copy()
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),
            thickness=-1,
        )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, scene_frame, 1 - alpha, 0, scene_frame)

        # Draw a white ring of radius 150 in the black rectangle
        ring_center = (x1 + rect_width // 2, y1 + rect_height // 2)
        cv2.circle(
            scene_frame,
            ring_center,
            115,
            (255, 255, 255),
            thickness=6,
        )

        # Draw N, S, E, W at the edge of the circle
        directions = {
            "N": (0, -1),
            "S": (0, 1),
            "E": (1, 0),
            "W": (-1, 0),
        }
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        for label, (dx, dy) in directions.items():
            text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            text_x = int(ring_center[0] + dx * 138 - text_size[0] // 2)
            text_y = int(ring_center[1] + dy * 138 + text_size[1] // 2)
            cv2.putText(
                scene_frame,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

        nsew_heading = np.array([heading_in_world[0], heading_in_world[1]])
        nsew_heading /= np.linalg.norm(nsew_heading)

        # Draw an arrow representing the nsew_heading direction on the scene_frame
        arrow_length = 108  # pixels
        center_x = int(x1 + rect_width // 2)
        center_y = int(y1 + rect_height // 2)
        end_x = int(center_x + nsew_heading[0] * arrow_length)
        end_y = int(center_y + -nsew_heading[1] * arrow_length)
        cv2.arrowedLine(
            scene_frame,
            (center_x, center_y),
            (end_x, end_y),
            color=(255, 0, 0),
            thickness=5,
            tipLength=0.23,
        )

        # Draw a black rectangle in the bottom left corner of the scene_frame
        rect_height = 110
        rect_width = 565
        margin = 0
        y1 = scene_frame.shape[0] - margin - rect_height
        y2 = scene_frame.shape[0] - margin
        x1 = margin
        x2 = margin + rect_width
        cv2.rectangle(
            scene_frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),
            thickness=-1,
        )

        if filtering:
            cv2.putText(
                scene_frame,
                "Filtering",
                (20, scene_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                scene_frame,
                "No Filtering",
                (20, scene_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Scene Camera - Press ESC to quit", scene_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return
        elif key == ord("f"):
            filtering = not filtering
            print(f"Filtering {'enabled' if filtering else 'disabled'}")

        eye_image_display.set_data(frame)

    if isinstance(eye_event, BlinkEventData):
        blink_time = time.time_ns()
        blinked = True

    if blinked:
        y = 0.8
        if time.time_ns() - blink_time > (0.1 * 1e9):
            blinked = False
    else:
        y = 0.0

    blink_data.append(y)
    if len(blink_data) > 100:
        blink_data = blink_data[-100:]

    x_vals = np.arange(len(blink_data))
    plot_blinks.set_data(x_vals, blink_data)

    pupil_diameter_left = 0.0
    pupil_diameter_right = 0.0
    if gaze is not None:
        if filtering and eye_state_filtered is not None:
            eye_center_left = np.array(
                [
                    eye_state_filtered[0],
                    eye_state_filtered[1],
                    eye_state_filtered[2],
                ]
            )
            opt_axis_filtered_left = sph2cart(
                np.array([eye_state_filtered[3], eye_state_filtered[4]])
            )
            optical_axis_left = np.array(
                [
                    opt_axis_filtered_left[0][0],
                    opt_axis_filtered_left[0][1],
                    opt_axis_filtered_left[0][2],
                ]
            )
            pupil_diameter_left = eye_state_filtered[5]
            eye_left.update_eye(
                eye_center_left,
                optical_axis_left,
                pupil_diameter_left,
                eye_state_filtered[7],
                eye_state_filtered[6],
            )
            eye_center_right = np.array(
                [
                    eye_state_filtered[9],
                    eye_state_filtered[10],
                    eye_state_filtered[11],
                ]
            )
            opt_axis_filtered_right = sph2cart(
                np.array([eye_state_filtered[12], eye_state_filtered[13]])
            )
            optical_axis_right = np.array(
                [
                    opt_axis_filtered_right[0][0],
                    opt_axis_filtered_right[0][1],
                    opt_axis_filtered_right[0][2],
                ]
            )
            pupil_diameter_right = eye_state_filtered[14]
            eye_right.update_eye(
                eye_center_right,
                optical_axis_right,
                pupil_diameter_right,
                eye_state_filtered[16],
                eye_state_filtered[15],
            )
        else:
            eye_center_left = np.array(
                [
                    gaze.eyeball_center_left_x,
                    gaze.eyeball_center_left_y,
                    gaze.eyeball_center_left_z,
                ]
            )
            optical_axis_left = np.array(
                [
                    gaze.optical_axis_left_x,
                    gaze.optical_axis_left_y,
                    gaze.optical_axis_left_z,
                ]
            )
            pupil_diameter_left = gaze.pupil_diameter_left
            eye_left.update_eye(
                eye_center_left,
                optical_axis_left,
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
            optical_axis_right = np.array(
                [
                    gaze.optical_axis_right_x,
                    gaze.optical_axis_right_y,
                    gaze.optical_axis_right_z,
                ]
            )
            pupil_diameter_right = gaze.pupil_diameter_right
            eye_right.update_eye(
                eye_center_right,
                optical_axis_right,
                pupil_diameter_right,
                gaze.eyelid_angle_top_right,
                gaze.eyelid_angle_bottom_right,
            )

        for artist in list(eyestate_ax.collections):
            artist.remove()

        eye_left.plot(eyestate_ax, (101 / 255, 188 / 255, 118 / 255))
        eye_right.plot(eyestate_ax, (229 / 255, 111 / 255, 114 / 255))

        left_pupil_data.append(pupil_diameter_left)
        if len(left_pupil_data) > 100:
            left_pupil_data = left_pupil_data[-100:]

        x_vals = np.arange(len(left_pupil_data))
        plot_left_pupil.set_data(x_vals, left_pupil_data)

        right_pupil_data.append(pupil_diameter_right)
        if len(right_pupil_data) > 100:
            right_pupil_data = right_pupil_data[-100:]

        x_vals = np.arange(len(right_pupil_data))
        plot_right_pupil.set_data(x_vals, right_pupil_data)

        oax_left_xz = np.array([optical_axis_left[0], optical_axis_left[2]])
        oax_left_xz /= np.linalg.norm(oax_left_xz)
        oax_left_xz *= 125
        plot_axes_left_xz.remove()
        plot_axes_left_xz = inset_ax_optaxes_xz.quiver(
            eye_center_left[0],
            eye_center_left[2],
            oax_left_xz[0],
            oax_left_xz[1],
            color=(101 / 255, 188 / 255, 118 / 255),
            scale=1,
            scale_units="xy",
            angles="xy",
            width=0.01,
        )

        oax_right_xz = np.array([optical_axis_right[0], optical_axis_right[2]])
        oax_right_xz /= np.linalg.norm(oax_right_xz)
        oax_right_xz *= 125
        plot_axes_right_xz.remove()
        plot_axes_right_xz = inset_ax_optaxes_xz.quiver(
            eye_center_right[0],
            eye_center_right[2],
            oax_right_xz[0],
            oax_right_xz[1],
            color=(229 / 255, 111 / 255, 114 / 255),
            scale=1,
            scale_units="xy",
            angles="xy",
            width=0.01,
        )

        oax_left_zy = np.array([optical_axis_left[2], optical_axis_left[1]])
        oax_left_zy /= np.linalg.norm(oax_left_zy)
        oax_left_zy *= 50
        plot_axes_left_zy.remove()
        plot_axes_left_zy = inset_ax_optaxes_zy.quiver(
            eye_center_left[2],
            eye_center_left[1],
            oax_left_zy[0],
            oax_left_zy[1],
            color=(101 / 255, 188 / 255, 118 / 255),
            scale=1,
            scale_units="xy",
            angles="xy",
            width=0.01,
        )

        oax_right_zy = np.array([optical_axis_right[2], optical_axis_right[1]])
        oax_right_zy /= np.linalg.norm(oax_right_zy)
        oax_right_zy *= 50
        plot_axes_right_zy.remove()
        plot_axes_right_zy = inset_ax_optaxes_zy.quiver(
            eye_center_right[2],
            eye_center_right[1],
            oax_right_zy[0],
            oax_right_zy[1],
            color=(229 / 255, 111 / 255, 114 / 255),
            scale=1,
            scale_units="xy",
            angles="xy",
            width=0.01,
        )

    return (
        eyestate_ax.collections
        + [eye_left.arrow]
        + [eye_right.arrow]
        + [plot_blinks]
        + [eye_image_display]
        + [plot_left_pupil]
        + [plot_right_pupil]
        + [plot_axes_left_xz]
        + [plot_axes_right_xz]
        + [plot_axes_left_zy]
        + [plot_axes_right_zy]
    )


ani = animation.FuncAnimation(fig, update, interval=33, blit=False)
plt.tight_layout()
plt.show()
