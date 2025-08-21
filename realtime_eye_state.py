import threading
from queue import Queue

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pupil_labs.realtime_api.simple import Device

from threeD_eye_model import ThreeDEyeModel

# We will use a background thread to collect data
# via Neon's Real-time API. This helps to offload the data
# acquisition from the main thread, allowing for smoother
# visualization updates.

data_queue = Queue(maxsize=1)


def data_acquisition_loop():
    while True:
        # See our Python API Documentation for more info about these two functions:
        # https://pupil-labs.github.io/pl-realtime-api/dev/
        scene_image = device.receive_scene_video_frame(timeout_seconds=0.03)
        eye_image = device.receive_eyes_video_frame(timeout_seconds=0.01)
        gaze = device.receive_gaze_datum(timeout_seconds=0.01)

        # Let's only pass data to the visualization when all relevant streams have
        # provided a datum. This makes the visualization logic simpler.
        if eye_image is not None and gaze is not None and scene_image is not None:
            eye_image = eye_image.bgr_pixels
            scene_image = scene_image.bgr_pixels

            # Clear out the queue if it's already full.
            if data_queue.full():
                data_queue.get_nowait()

            data_queue.put_nowait(
                {
                    "eye": eye_image,
                    "gaze": gaze,
                    "scene": scene_image,
                }
            )


# Now, we make class to hold all the elements of the matplotlib figure.
# This makes it easier to organize the visualization logic later.
# The figure will have three sections:
# - A space at the top to display the current eye image.
# - A space in the middle to display the animated 3D eye model.
# - A space at the bottom for plots that will show the optical axis vectors, for quick inspection
#   of things, like vergence angle.


class ThreeDVisualization:
    GREEN = (101 / 255, 188 / 255, 118 / 255)
    RED = (229 / 255, 111 / 255, 114 / 255)

    def __init__(self):
        self.fig = plt.figure(figsize=(6, 6))

        self.fig.canvas.manager.set_window_title("3D Eye State Visualization")
        self.fig.patch.set_facecolor("black")

        # These are the main axes in the figure that hold the 3D plots of the eye model,
        # as well as the insets that we will be adding.
        self.eyestate_ax = self.fig.add_subplot(111, projection="3d")
        self.eyestate_ax.set_xlim([-40, 40])
        self.eyestate_ax.set_ylim([50, 38])
        self.eyestate_ax.set_zlim([-40, 60])
        # self.eyestate_ax.set_zlim([50, 38])

        # Makes sure that 3D plots have equal aspect ratio on all sides.
        # Adapted from:
        # https://github.com/matplotlib/matplotlib/issues/17172#issuecomment-830139107
        self.eyestate_ax.set_box_aspect(
            [
                ub - lb
                for lb, ub in (
                    getattr(self.eyestate_ax, f"get_{a}lim")() for a in "xyz"
                )
            ]
        )

        self.eyestate_ax.set_xlabel("X")
        self.eyestate_ax.set_ylabel("Z")
        self.eyestate_ax.set_zlabel("Y")
        self.eyestate_ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
        self.eyestate_ax.grid(False)
        self.eyestate_ax.set_axis_off()
        self.eyestate_ax.set_facecolor("black")
        self.eyestate_ax.view_init(elev=12, azim=90)

        # This inset will display the current eye image.
        inset_ax_eye_image = inset_axes(
            self.eyestate_ax, width="100%", height="28%", loc="upper left"
        )
        inset_ax_eye_image.axis("off")
        # We need to plot something into the insets,
        # in order to have a plot object that will be updated
        # in the animation loop
        self.eye_image_plot = inset_ax_eye_image.imshow(
            np.zeros((192, int(192 * 2), 3), dtype=np.uint8)
        )  # placeholder

        # This inset will display the optical axes of the left and right eye from an overhead view.
        # In other words, it will display the X and Z coordinates of the optical axes.
        self.inset_ax_optaxes_xz = inset_axes(
            self.eyestate_ax,
            width="25%",
            height="22%",
            loc="lower left",
            bbox_to_anchor=(0.17, 0, 1, 1),
            bbox_transform=self.eyestate_ax.transAxes,
        )
        self.inset_ax_optaxes_xz.set_facecolor("black")
        self.inset_ax_optaxes_xz.set_xlim(-55, 55)
        self.inset_ax_optaxes_xz.set_ylim(-70, 100)
        self.inset_ax_optaxes_xz.set(xticklabels=[], yticklabels=[])
        self.inset_ax_optaxes_xz.grid(False)
        self.inset_ax_optaxes_xz.text(
            0.05,
            0.15,
            "Optical Axes (X,Z)",
            color="white",
            fontsize=10,
            transform=self.inset_ax_optaxes_xz.transAxes,
            va="top",
        )
        self.inset_ax_optaxes_xz.tick_params(colors="black")
        for spine in self.inset_ax_optaxes_xz.spines.values():
            spine.set_edgecolor("white")

        self.optaxes_left_xz_plot = FancyArrowPatch(
            (2, -4),
            (2, -4),
            color=ThreeDVisualization.GREEN,
            arrowstyle="->",
            mutation_scale=15,
            lw=2,
        )
        self.inset_ax_optaxes_xz.add_patch(self.optaxes_left_xz_plot)
        self.optaxes_right_xz_plot = FancyArrowPatch(
            (2, -4),
            (2, -4),
            color=ThreeDVisualization.RED,
            arrowstyle="->",
            mutation_scale=15,
            lw=2,
        )
        self.inset_ax_optaxes_xz.add_patch(self.optaxes_right_xz_plot)

        # This inset will display the optical axes of the left and right eye from a side-profile view.
        # In other words, it will display the Z and Y coordinates of the optical axes.
        self.inset_ax_optaxes_zy = inset_axes(
            self.eyestate_ax,
            width="25%",
            height="22%",
            loc="lower right",
            bbox_to_anchor=(-0.15, 0, 1, 1),
            bbox_transform=self.eyestate_ax.transAxes,
        )
        self.inset_ax_optaxes_zy.set_facecolor("black")
        self.inset_ax_optaxes_zy.set_xlim(-60, 30)
        self.inset_ax_optaxes_zy.set_ylim(-10, 35)
        self.inset_ax_optaxes_zy.set(xticklabels=[], yticklabels=[])
        self.inset_ax_optaxes_zy.grid(False)
        self.inset_ax_optaxes_zy.text(
            0.05,
            0.15,
            "Optical Axes (Z,Y)",
            color="white",
            fontsize=10,
            transform=self.inset_ax_optaxes_zy.transAxes,
            va="top",
        )
        self.inset_ax_optaxes_zy.tick_params(colors="black")
        for spine in self.inset_ax_optaxes_zy.spines.values():
            spine.set_edgecolor("white")

        self.optaxes_left_zy_plot = FancyArrowPatch(
            (2, -4),
            (2, -4),
            color=ThreeDVisualization.GREEN,
            arrowstyle="->",
            mutation_scale=15,
            lw=2,
        )
        self.inset_ax_optaxes_zy.add_patch(self.optaxes_left_zy_plot)
        self.optaxes_right_zy_plot = FancyArrowPatch(
            (2, -4),
            (2, -4),
            color=ThreeDVisualization.RED,
            arrowstyle="->",
            mutation_scale=15,
            lw=2,
        )
        self.inset_ax_optaxes_zy.add_patch(self.optaxes_right_zy_plot)


if __name__ == "__main__":
    # First, establish a connection to Neon.
    device = Device(address="192.168.1.34", port=8080)
    # device = discover_one_device()

    # Start up the data acquisition threads.
    data_acquisition_thread = threading.Thread(
        target=data_acquisition_loop, daemon=True
    )
    data_acquisition_thread.start()

    # Create a separate ThreeDEyeModel instance for each eye.
    eye_left = ThreeDEyeModel()
    eye_right = ThreeDEyeModel()

    threeD_viz = ThreeDVisualization()

    # Now, we get to the main animation callback that will be repeatedly called by matplotlib's
    # animation routines for each frame. It essentially:
    #
    # - Gets the latest data in the data_queue, as provided by the data acquisition thread.
    # - Converts the eyestate data into a format that is more easily used for plotting.
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
        gaze = None
        scene_img = None
        if not data_queue.empty():
            data = data_queue.get_nowait()
            eye_img = data["eye"]
            gaze = data["gaze"]
            scene_img = data["scene"]

        # If there is a scene image and gaze data available, then draw a circle
        # at the gaze point and display the resulting image via OpenCV's imshow function.
        if scene_img is not None and gaze is not None:
            cv2.circle(
                scene_img,
                (int(gaze.x), int(gaze.y)),
                radius=80,
                color=(0, 0, 255),
                thickness=15,
            )

            cv2.imshow("Scene Camera + Gaze Overlay - Press ESC to quit", scene_img)

        # If there is an eye image available, then draw it to the
        # associated plot.
        if eye_img is not None:
            threeD_viz.eye_image_plot.set_data(eye_img)

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
            eye_left.update(
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
            eye_right.update(
                eye_center_right,
                optical_axis_vector_right,
                pupil_diameter_right,
                gaze.eyelid_angle_top_right,
                gaze.eyelid_angle_bottom_right,
            )

            # We use matplotlib's `plot_surface` to draw the eyeball,
            # and we need to clear them from the plot before drawing the next
            # frame. Comment these two lines to see the effect.
            for artist in list(threeD_viz.eyestate_ax.collections):
                artist.remove()

            # Plot the measured eyestate in 3D!
            eye_left.plot(threeD_viz.eyestate_ax, threeD_viz.GREEN)
            eye_right.plot(threeD_viz.eyestate_ax, threeD_viz.RED)

            # Plot the optical axes from an overhead view.
            optaxis_left_xz = np.array(
                [optical_axis_vector_left[0], optical_axis_vector_left[2]]
            )
            # Normalize and rescale the vector to make it easier to see in the
            # plot.
            optaxis_left_xz /= np.linalg.norm(optaxis_left_xz)
            optaxis_left_xz *= 125
            start = (eye_center_left[0], eye_center_left[2])
            end = (
                eye_center_left[0] + optaxis_left_xz[0],
                eye_center_left[2] + optaxis_left_xz[1],
            )
            threeD_viz.optaxes_left_xz_plot.set_positions(start, end)

            optaxis_right_xz = np.array(
                [optical_axis_vector_right[0], optical_axis_vector_right[2]]
            )
            optaxis_right_xz /= np.linalg.norm(optaxis_right_xz)
            optaxis_right_xz *= 125
            start = (eye_center_right[0], eye_center_right[2])
            end = (
                eye_center_right[0] + optaxis_right_xz[0],
                eye_center_right[2] + optaxis_right_xz[1],
            )
            threeD_viz.optaxes_right_xz_plot.set_positions(start, end)

            # We do similar for the side-profile view of the optical axes.
            optaxis_left_zy = np.array(
                [optical_axis_vector_left[2], optical_axis_vector_left[1]]
            )
            optaxis_left_zy /= np.linalg.norm(optaxis_left_zy)
            optaxis_left_zy *= 50
            start = (eye_center_left[2], eye_center_left[1])
            end = (
                eye_center_left[2] + optaxis_left_zy[0],
                eye_center_left[1] + optaxis_left_zy[1],
            )
            threeD_viz.optaxes_left_zy_plot.set_positions(start, end)

            optaxis_right_zy = np.array(
                [optical_axis_vector_right[2], optical_axis_vector_right[1]]
            )
            optaxis_right_zy /= np.linalg.norm(optaxis_right_zy)
            optaxis_right_zy *= 50
            start = (eye_center_right[2], eye_center_right[1])
            end = (
                eye_center_right[2] + optaxis_right_zy[0],
                eye_center_right[1] + optaxis_right_zy[1],
            )
            threeD_viz.optaxes_right_zy_plot.set_positions(start, end)

        # Finally, we return a list of plot objects, per
        # matplotlib's conventions.
        return (
            threeD_viz.eyestate_ax.collections
            + [eye_left.optical_axis.optical_axis_quiver_plot]
            + [eye_right.optical_axis.optical_axis_quiver_plot]
            + [threeD_viz.eye_image_plot]
            + [threeD_viz.optaxes_left_xz_plot]
            + [threeD_viz.optaxes_right_xz_plot]
            + [threeD_viz.optaxes_left_zy_plot]
            + [threeD_viz.optaxes_right_zy_plot]
        )

    # Wrap the actual animation "loop" in a try-except-finally block.
    # This is done because if anything goes wrong or the
    # user quits the program, then we need to cleanly close our
    # connection to Neon and clean up any OpenCV windows.
    try:
        # Create a matplotlib animation function that will run the `update` function
        # every 33ms. This will update our animation at ~30 FPS. We pass `blit=False`,
        # because matplotlib's 3D functionality does not support blitting.
        ani = animation.FuncAnimation(threeD_viz.fig, update, interval=33, blit=False)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # No matter what happens, we need to clean close our connection to Neon
        # and close all OpenCV windows.
        device.close()
        cv2.destroyAllWindows()
