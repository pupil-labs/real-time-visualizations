# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python",
#     "numpy",
#     "pupil-labs-realtime-api",
#     "matplotlib",
# ]
# ///

import threading
import time
from collections import deque
from queue import Empty, Queue

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from pupil_labs.realtime_api.simple import Device, discover_one_device
from pupil_labs.realtime_api.streaming.eye_events import (
    BlinkEventData,
)

# --- Configuration Constants ---
FALLBACK_DEVICE_ADDRESS: str = "192.168.1.34"
FALLBACK_DEVICE_PORT: int = 8080
ANIMATION_FRAME_RATE: int = 30  # FPS
TIMESERIES_DATA_POINTS: int = 200  # Number of points to show in plots
BLINK_PULSE_DURATION_S: float = 0.05  # Seconds for which a blink pulse is plotted

# We will use a background thread to collect data
# via Neon's Real-time API. This helps to offload the data
# acquisition from the main thread, allowing for smoother
# visualization updates.

data_queue = Queue(maxsize=1)


def data_acquisition_loop():
    while True:
        # See our Python API Documentation for more info about these two functions:
        # https://pupil-labs.github.io/pl-realtime-api/dev/
        eye_image = device.receive_eyes_video_frame(timeout_seconds=0.01)
        gaze = device.receive_gaze_datum(timeout_seconds=0.01)
        eye_event = device.receive_eye_events(timeout_seconds=0.01)

        # Let's only pass data to the visualization when all relevant streams have
        # provided a datum. This makes the visualization logic simpler.
        if not all((eye_image, gaze)):
            continue

        # Clear out the queue if it's already full.
        if data_queue.full():
            data_queue.get_nowait()

        data_queue.put_nowait(
            {
                "eye": eye_image.bgr_pixels,
                "gaze": gaze,
                "eye_event": eye_event,
            }
        )


# Now, we make class to hold all the elements of the matplotlib figure.
# This makes it easier to organize the visualization logic later.
# The figure will have three sections:
# - A space at the top to display the current eye image.
# - A space in the middle to display the blink events.
# - A space at the bottom for the pupil diameter plots.


class Visualization:
    GREEN = (101 / 255, 188 / 255, 118 / 255)
    RED = (229 / 255, 111 / 255, 114 / 255)
    BLUE = (99 / 255, 108 / 255, 191 / 255)

    def __init__(self):
        self.blink_end_time_ns = 0
        # self.blinked = False
        # self.blink_data = 0.0
        # self.blink_data = []
        # self.left_pupil_data = []
        # self.right_pupil_data = []

        self.blink_data = (
            deque([0.0] * TIMESERIES_DATA_POINTS, maxlen=TIMESERIES_DATA_POINTS),
        )
        self.left_pupil_data = (
            deque([0.0] * TIMESERIES_DATA_POINTS, maxlen=TIMESERIES_DATA_POINTS),
        )
        self.right_pupil_data = (
            deque([0.0] * TIMESERIES_DATA_POINTS, maxlen=TIMESERIES_DATA_POINTS),
        )

        self.fig, (self.ax_eye_image, self.ax_blinks, self.ax_pupil_diameter) = (
            plt.subplots(3, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1, 1, 1]})
        )
        self.fig.canvas.manager.set_window_title("Neon Graphs Visualization")
        self.fig.patch.set_facecolor("black")

        self._setup_plot_axes()

    def _setup_plot_axes(self):
        # Eye image subplot
        self.ax_eye_image.axis("off")
        self.eye_image_display = self.ax_eye_image.imshow(
            np.zeros((192, 192 * 2, 3), dtype=np.uint8)
        )  # placeholder

        # Blinks subplot
        self.ax_blinks.set_facecolor("black")
        self.ax_blinks.set_xlim(0, TIMESERIES_DATA_POINTS)
        self.ax_blinks.set_ylim(0, 1.3)
        self.ax_blinks.set_xticklabels([])
        self.ax_blinks.set_yticklabels([])
        self.ax_blinks.grid(False)
        self.ax_blinks.text(
            0.05,
            0.95,
            "Blinks",
            color="white",
            fontsize=10,
            transform=self.ax_blinks.transAxes,
            va="top",
        )
        self.ax_blinks.tick_params(colors="black")
        for spine in self.ax_blinks.spines.values():
            spine.set_edgecolor("white")

        (self.blinks_plot,) = self.ax_blinks.plot([], [], linewidth=1.5)
        self.blinks_plot.set_color((99 / 255, 108 / 255, 191 / 255))

        # Pupil diameter subplot
        self.ax_pupil_diameter.set_facecolor("black")
        self.ax_pupil_diameter.set_xlim(0, 200)
        self.ax_pupil_diameter.set_ylim(0, 8.0)
        self.ax_pupil_diameter.set_xticklabels([])
        self.ax_pupil_diameter.set_yticklabels([])
        self.ax_pupil_diameter.grid(False)
        self.ax_pupil_diameter.text(
            0.02,
            0.90,
            "Pupil Diameter [mm]",
            color="white",
            fontsize=10,
            transform=self.ax_pupil_diameter.transAxes,
            va="top",
        )
        self.ax_pupil_diameter.tick_params(colors="black")
        for spine in self.ax_pupil_diameter.spines.values():
            spine.set_edgecolor("white")

        (self.left_pupil_plot,) = self.ax_pupil_diameter.plot([], [], linewidth=1.5)
        self.left_pupil_plot.set_color(self.GREEN)

        (self.right_pupil_plot,) = self.ax_pupil_diameter.plot([], [], linewidth=1.5)
        self.right_pupil_plot.set_color(self.RED)

    def update_blinks_plot(self, blink_value):
        self.blink_data.append(blink_value)
        # if len(self.blink_data) > 200:
        # self.blink_data = self.blink_data[-200:]

        x_vals = np.arange(len(self.blink_data))
        self.blinks_plot.set_data(x_vals, self.blink_data)

    def update_pupil_plots(self, pupil_diameter_left, pupil_diameter_right):
        self.left_pupil_data.append(pupil_diameter_left)
        # if len(self.left_pupil_data) > 200:
        # self.left_pupil_data = self.left_pupil_data[-200:]

        x_vals = np.arange(len(self.left_pupil_data))
        self.left_pupil_plot.set_data(x_vals, self.left_pupil_data)

        self.right_pupil_data.append(pupil_diameter_right)
        # if len(self.right_pupil_data) > 200:
        # self.right_pupil_data = self.right_pupil_data[-200:]

        x_vals = np.arange(len(self.right_pupil_data))
        self.right_pupil_plot.set_data(x_vals, self.right_pupil_data)


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
        # Without having a brief pause before updating images, the matplotlib animation stutters.
        # It is also required for properly updating OpenCV windows.
        # Here, we use OpenCV's waitKey function to introduce a small 1ms delay and
        # also check if the user pressed Esc, which will quit the visualization.
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return

        eye_img = None
        eye_event = None
        gaze = None
        try:
            data = data_queue.get_nowait()
            eye_img = data["eye"]
            gaze = data["gaze"]
            eye_event = data["eye_event"]
        except Empty:
            pass

        # If there is an eye image available, then draw it to the
        # associated plot.
        if eye_img is not None:
            viz.eye_image_display.set_data(eye_img)

        if isinstance(eye_event, BlinkEventData):
            # viz.blinked = True
            # viz.blink_time = time.time_ns()
            viz.blink_end_time_ns = time.time_ns() + (BLINK_PULSE_DURATION_S * 1e9)

        # if viz.blinked:
        #     val = 0.8
        #     if time.time_ns() - viz.blink_time > (0.05 * 1e9):
        #         viz.blinked = False
        # else:
        #     val = 0.0
        blink_value = 0.8 if time.time_ns() < viz.blink_end_time_ns else 0.0

        viz.update_blinks_plot(blink_value)

        # If there is gaze data available, then update the
        # pupil diameter plots.
        if gaze is not None:
            viz.update_pupil_diameter_plots(
                gaze.pupil_diameter_left, gaze.pupil_diameter_right
            )

        # Finally, we return a list of plot objects, per
        # matplotlib's conventions.
        return (
            [viz.blinks_plot]
            + [viz.eye_image_display]
            + [viz.left_pupil_plot]
            + [viz.right_pupil_plot]
        )

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
