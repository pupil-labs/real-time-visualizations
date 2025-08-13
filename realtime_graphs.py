import threading
import time
from queue import Queue

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from pupil_labs.realtime_api.simple import Device
from pupil_labs.realtime_api.streaming.eye_events import (
    BlinkEventData,
)

device = Device(address="192.168.1.57", port=8080)
# device = Device(address="172.20.10.2", port=8080)


# Setup plot
fig, (ax_eye_image, ax_blinks, ax_pupil_diameter) = plt.subplots(
    3, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1, 1, 1]}
)
fig.canvas.manager.set_window_title("3D Eye State Visualization")
fig.patch.set_facecolor("black")

# Eye image subplot
ax_eye_image.axis("off")
eye_image_display = ax_eye_image.imshow(
    np.zeros((192, int(192 * 2), 3), dtype=np.uint8)
)  # placeholder

# Blinks subplot
ax_blinks.set_facecolor("black")
ax_blinks.set_xlim(0, 200)
ax_blinks.set_ylim(0, 1.3)
ax_blinks.set_xticklabels([])
ax_blinks.set_yticklabels([])
ax_blinks.grid(False)
ax_blinks.text(
    0.05,
    0.95,
    "Blinks",
    color="white",
    fontsize=10,
    transform=ax_blinks.transAxes,
    va="top",
)
ax_blinks.tick_params(colors="black")
for spine in ax_blinks.spines.values():
    spine.set_edgecolor("white")

(plot_blinks,) = ax_blinks.plot([], [], linewidth=1.5)
plot_blinks.set_color((99 / 255, 108 / 255, 191 / 255))

# Pupil diameter subplot
ax_pupil_diameter.set_facecolor("black")
ax_pupil_diameter.set_xlim(0, 200)
ax_pupil_diameter.set_ylim(0, 8.0)
ax_pupil_diameter.set_xticklabels([])
ax_pupil_diameter.set_yticklabels([])
ax_pupil_diameter.grid(False)
ax_pupil_diameter.text(
    0.02,
    0.90,
    "Pupil Diameter [mm]",
    color="white",
    fontsize=10,
    transform=ax_pupil_diameter.transAxes,
    va="top",
)
ax_pupil_diameter.tick_params(colors="black")
for spine in ax_pupil_diameter.spines.values():
    spine.set_edgecolor("white")

(plot_left_pupil,) = ax_pupil_diameter.plot([], [], linewidth=1.5)
plot_left_pupil.set_color((101 / 255, 188 / 255, 118 / 255))

(plot_right_pupil,) = ax_pupil_diameter.plot([], [], linewidth=1.5)
plot_right_pupil.set_color((229 / 255, 111 / 255, 114 / 255))


data_queue = Queue(maxsize=1)


def data_acquisition_loop():
    while True:
        eye_image = device.receive_eyes_video_frame(timeout_seconds=0.01)
        gaze = device.receive_gaze_datum(timeout_seconds=0.01)
        eye_event = device.receive_eye_events(timeout_seconds=0.01)

        if gaze is not None and eye_image is not None:
            eye_image = eye_image.bgr_pixels

            eye_image = cv2.resize(
                eye_image, (int(eye_image.shape[1] * 1.2), eye_image.shape[0])
            )

            if data_queue.full():
                data_queue.get_nowait()

            data_queue.put_nowait(
                {
                    "eye": eye_image,
                    "gaze": gaze,
                    "eye_event": eye_event,
                }
            )


data_acquisition_thread = threading.Thread(target=data_acquisition_loop, daemon=True)
data_acquisition_thread.start()

blink_data = []
blinked = False
blink_time = 0.0

left_pupil_data = []
right_pupil_data = []


def update(frame_number):
    global blink_data
    global blinked
    global blink_time

    global left_pupil_data
    global right_pupil_data

    frame = None
    eye_event = None
    gaze = None
    if not data_queue.empty():
        data = data_queue.get()
        frame = data["eye"]
        gaze = data["gaze"]
        eye_event = data["eye_event"]

    if frame is not None:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return

        eye_image_display.set_data(frame)

    if isinstance(eye_event, BlinkEventData):
        blink_time = time.time_ns()
        blinked = True

    if blinked:
        y = 0.8
        if time.time_ns() - blink_time > (0.05 * 1e9):
            blinked = False
    else:
        y = 0.0

    blink_data.append(y)
    if len(blink_data) > 200:
        blink_data = blink_data[-200:]

    x_vals = np.arange(len(blink_data))
    plot_blinks.set_data(x_vals, blink_data)

    pupil_diameter_left = 0.0
    pupil_diameter_right = 0.0
    if gaze is not None:
        pupil_diameter_left = gaze.pupil_diameter_left
        pupil_diameter_right = gaze.pupil_diameter_right

        left_pupil_data.append(pupil_diameter_left)
        if len(left_pupil_data) > 200:
            left_pupil_data = left_pupil_data[-200:]

        x_vals = np.arange(len(left_pupil_data))
        plot_left_pupil.set_data(x_vals, left_pupil_data)

        right_pupil_data.append(pupil_diameter_right)
        if len(right_pupil_data) > 200:
            right_pupil_data = right_pupil_data[-200:]

        x_vals = np.arange(len(right_pupil_data))
        plot_right_pupil.set_data(x_vals, right_pupil_data)

    return [plot_blinks] + [eye_image_display] + [plot_left_pupil] + [plot_right_pupil]


ani = animation.FuncAnimation(fig, update, interval=33, blit=False)
plt.tight_layout()
plt.show()
