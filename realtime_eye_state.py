import queue
import threading
from queue import Queue

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from pupil_labs.realtime_api.simple import Device

from threeD_eye_model import ThreeDEyeModel

# Wrap everything in a big try-except-finally block.
# This is done because if anything goes wrong or the
# user quits the program, then we need to cleanly close our
# connection to Neon and clean up any OpenCV windows.
try:
    # First, establish a connection to Neon.
    device = Device(address="192.168.1.34", port=8080)
    # device = discover_one_device()

    # Next, let's prepare a thread that will collect data
    # via Neon's Real-time API. This helps to offload the data
    # acquisition from the main thread, allowing for smoother
    # visualization updates.

    # A Queue is used to pass the data from the acquistion
    # thread to the visualization loop.
    data_queue = Queue(maxsize=2)

    # This function runs in a separate thread and continuously collects data
    # from the device.
    def data_acquisition_loop():
        while True:
            try:
                # See our Python API Documentation for more info about these two functions:
                # https://pupil-labs.github.io/pl-realtime-api/dev/
                eye_image = device.receive_eyes_video_frame(timeout_seconds=0.01)
                gaze = device.receive_gaze_datum(timeout_seconds=0.01)
                scene_image = device.receive_scene_video_frame()

                # Let's only pass data to the visualization when all relevant streams have
                # provided a datum. This makes the visualization logic simpler.
                if (
                    eye_image is not None
                    and gaze is not None
                    and scene_image is not None
                ):
                    eye_image = eye_image.bgr_pixels
                    scene_image = scene_image.bgr_pixels

                    # Clear out the queue if it's already full.
                    # while not data_queue.empty():
                    # data_queue.get()
                    # data_queue.get()

                    data_queue.put(
                        {"eye": eye_image, "gaze": gaze, "scene": scene_image}
                    )
                    data_queue.put(
                        {"eye": eye_image, "gaze": gaze, "scene": scene_image}
                    )
            except queue.Full:
                pass

    # Start the data acquisition thread. It will now run in the background.
    data_acquisition_thread = threading.Thread(
        target=data_acquisition_loop, daemon=True
    )
    data_acquisition_thread.start()

    # For interpreting data, it can also be helpful to see the scene camera feed, with a
    # gaze overlay in real-time. We will receive scene camera images in a separate thread,
    # to improve efficiency a bit.
    # Similar to before, we will send the scene camera images to the visualization routine
    # via a queue.
    scene_queue = Queue(maxsize=1)

    # This function runs in a separate thread and continuously collects scene camera images
    # from the device.
    def scene_acquisition_loop():
        while True:
            try:
                # See our Python API Documentation for more info about this function:
                # https://pupil-labs.github.io/pl-realtime-api/dev/
                scene_image = device.receive_scene_video_frame()

                # Let's only pass scene images to the visualization when an image is available.
                # This makes the visualization logic simpler.
                if scene_image is not None:
                    scene_image = scene_image.bgr_pixels

                    # Clear out the queue if it's already full.
                    if scene_queue.full():
                        scene_queue.get()

                    scene_queue.put(scene_image)
            except queue.Full:
                pass

    # Start the scene image acquisition thread. It will now run in the background.
    # scene_acquisition_thread = threading.Thread(
    # target=scene_acquisition_loop, daemon=True
    # )
    # scene_acquisition_thread.start()

    # We need a separate ThreeDEyeModel instance for each eye.
    eye_left = ThreeDEyeModel()
    eye_right = ThreeDEyeModel()

    # Now, we can prepare the matplotlib figure.
    # It will have three sections:
    # - A space at the top to display the current eye image.
    # - A space in the middle to display the animated 3D eye model.
    # - A space at the bottom for plots that will show the optical axis vectors, for quick inspection
    #   of things, like vergence angle.

    fig1 = plt.figure(figsize=(6, 6))

    # We change settings related to aesthetics.
    # These have no influence on the actual data, but make it easier to see what is going on.
    fig1.canvas.manager.set_window_title("3D Eye State Visualization")
    fig1.patch.set_facecolor("black")

    # These are main axes in the figure that hold the 3D plots of the eye model,
    # as well as the insets that we will be adding.
    eyestate_ax = fig1.add_subplot(111, projection="3d")
    eyestate_ax.set_xlim([-40, 40])
    eyestate_ax.set_ylim([-40, 60])
    eyestate_ax.set_zlim([50, 38])

    # Makes sure that 3D plots have equal aspect ratio on all sides.
    # Adapted from:
    # https://github.com/matplotlib/matplotlib/issues/17172#issuecomment-830139107
    eyestate_ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(eyestate_ax, f"get_{a}lim")() for a in "xyz")]
    )

    eyestate_ax.set_xlabel("X")
    eyestate_ax.set_ylabel("Z")
    eyestate_ax.set_zlabel("Y")
    eyestate_ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    eyestate_ax.grid(False)
    eyestate_ax.set_axis_off()
    eyestate_ax.set_facecolor("black")
    eyestate_ax.view_init(elev=12, azim=90)

    # This inset will display the current eye image.
    # inset_ax_eye_image = inset_axes(
    #     eyestate_ax, width="100%", height="28%", loc="upper left"
    # )
    # inset_ax_eye_image.axis("off")
    # An inset just defines the axes. You then need to plot something into them,
    # in order to have a plot object that will be updated in the animation loop
    # later.
    # eye_image_plot = inset_ax_eye_image.imshow(
    #     np.zeros((192, int(192 * 2), 3), dtype=np.uint8)
    # )  # placeholder

    fig2 = plt.figure(figsize=(6, 6))
    fig2.patch.set_facecolor("black")

    # This inset will display the optical axes of the left and right eye from an overhead view.
    # In other words, it will display the X and Z coordinates of the optical axes.
    # inset_ax_optaxes_xz = inset_axes(
    #     eyestate_ax,
    #     width="25%",
    #     height="22%",
    #     loc="lower left",
    #     bbox_to_anchor=(0.17, 0, 1, 1),
    #     bbox_transform=eyestate_ax.transAxes,
    # )
    inset_ax_optaxes_xz = fig2.add_subplot(121)
    inset_ax_optaxes_xz.set_facecolor("black")
    # inset_ax_optaxes_xz.set_xlim(-55, 55)
    # inset_ax_optaxes_xz.set_ylim(-10, 180)
    inset_ax_optaxes_xz.set(xticklabels=[], yticklabels=[])
    inset_ax_optaxes_xz.set_title("Optical Axes (X,Z)", color="white")
    inset_ax_optaxes_xz.set_xlabel("X")
    inset_ax_optaxes_xz.set_ylabel("Z")
    inset_ax_optaxes_xz.set_aspect("equal")
    inset_ax_optaxes_xz.set_box_aspect(1)
    inset_ax_optaxes_xz.grid(False)
    # inset_ax_optaxes_xz.text(
    #     0.05,
    #     0.15,
    #     "Optical Axes (X,Z)",
    #     color="white",
    #     fontsize=10,
    #     transform=inset_ax_optaxes_xz.transAxes,
    #     va="top",
    # )
    inset_ax_optaxes_xz.tick_params(colors="black")
    for spine in inset_ax_optaxes_xz.spines.values():
        spine.set_edgecolor("white")

    # Again, we need to plot something into the insets,
    # in order to have a plot object that will be updated
    # in the animation loop
    optaxes_left_xz_plot = inset_ax_optaxes_xz.quiver(
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
    optaxes_right_xz_plot = inset_ax_optaxes_xz.quiver(
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

    # This inset will display the optical axes of the left and right eye from a side-profile view.
    # In other words, it will display the Z and Y coordinates of the optical axes.
    # inset_ax_optaxes_zy = inset_axes(
    #     eyestate_ax,
    #     width="25%",
    #     height="22%",
    #     loc="lower right",
    #     bbox_to_anchor=(-0.15, 0, 1, 1),
    #     bbox_transform=eyestate_ax.transAxes,
    # )
    inset_ax_optaxes_zy = fig2.add_subplot(122)
    inset_ax_optaxes_zy.set_facecolor("black")
    # inset_ax_optaxes_zy.set_xlim(25, 100)
    # inset_ax_optaxes_zy.set_ylim(10, -35)
    inset_ax_optaxes_zy.set(xticklabels=[], yticklabels=[])
    inset_ax_optaxes_zy.set_title("Optical Axes (Z,Y)", color="white")
    inset_ax_optaxes_zy.set_xlabel("Z")
    inset_ax_optaxes_zy.set_ylabel("Y")
    inset_ax_optaxes_xz.set_aspect("equal")
    inset_ax_optaxes_xz.set_box_aspect(1)
    inset_ax_optaxes_zy.grid(False)
    # inset_ax_optaxes_zy.text(
    #     0.05,
    #     0.15,
    #     "Optical Axes (Z,Y)",
    #     color="white",
    #     fontsize=10,
    #     transform=inset_ax_optaxes_zy.transAxes,
    #     va="top",
    # )
    inset_ax_optaxes_zy.tick_params(colors="black")
    for spine in inset_ax_optaxes_zy.spines.values():
        spine.set_edgecolor("white")

    # Again, we need to plot something into the insets,
    # in order to have a plot object that will be updated
    # in the animation loop
    optaxes_left_zy_plot = inset_ax_optaxes_zy.quiver(
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
    optaxes_right_zy_plot = inset_ax_optaxes_zy.quiver(
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

    # Create two OpenCV windows for displaying:
    # 1. Scene camera feed with gaze overlay
    # 2. Eye camera feed
    cv2.namedWindow("Scene Camera + Gaze Overlay - Press ESC to quit")
    cv2.namedWindow("Eye Cameras - Press ESC to quit")

    def update2(frame_number):
        # Some matplotlib plotting objects need to be declared as global
        # to be accessible from the `update` function. A linter, such as Ruff,
        # can easily point out which ones need to be declared as global.
        global optaxes_left_xz_plot
        global optaxes_right_xz_plot
        global optaxes_left_zy_plot
        global optaxes_right_zy_plot

        gaze = None
        if not data_queue.empty():
            data = data_queue.get()
            gaze = data["gaze"]

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

            # Plot an overhead view of the optical axes as arrows, using matplotlib's quiver.
            # This can be used to qualitatively inspect vergence, for example.
            optaxis_left_xz = np.array(
                [optical_axis_vector_left[0], optical_axis_vector_left[2]]
            )

            # Normalize and rescale the vector to make it easier to see in the
            # plot.
            optaxis_left_xz /= np.linalg.norm(optaxis_left_xz)
            optaxis_left_xz *= 125

            # Again, some matplotlib objects need to be manually removed, before you
            # draw the next frame. Comment the `remove()` line below to see what happens,
            # if you don't do this.
            optaxes_left_xz_plot.remove()
            optaxes_left_xz_plot = inset_ax_optaxes_xz.quiver(
                eye_center_left[0],
                eye_center_left[2],
                optaxis_left_xz[0],
                optaxis_left_xz[1],
                color=(101 / 255, 188 / 255, 118 / 255),
                scale=1,
                scale_units="xy",
                angles="xy",
                width=0.01,
            )

            # Plot the right optical axis vector in an overhead view.
            optaxis_right_xz = np.array(
                [optical_axis_vector_right[0], optical_axis_vector_right[2]]
            )
            optaxis_right_xz /= np.linalg.norm(optaxis_right_xz)
            optaxis_right_xz *= 125
            optaxes_right_xz_plot.remove()
            optaxes_right_xz_plot = inset_ax_optaxes_xz.quiver(
                eye_center_right[0],
                eye_center_right[2],
                optaxis_right_xz[0],
                optaxis_right_xz[1],
                color=(229 / 255, 111 / 255, 114 / 255),
                scale=1,
                scale_units="xy",
                angles="xy",
                width=0.01,
            )

            # We do similar for the side-profile view of the optical axes.
            optaxis_left_zy = np.array(
                [optical_axis_vector_left[2], optical_axis_vector_left[1]]
            )
            optaxis_left_zy /= np.linalg.norm(optaxis_left_zy)
            optaxis_left_zy *= 50
            optaxes_left_zy_plot.remove()
            optaxes_left_zy_plot = inset_ax_optaxes_zy.quiver(
                eye_center_left[2],
                eye_center_left[1],
                optaxis_left_zy[0],
                optaxis_left_zy[1],
                color=(101 / 255, 188 / 255, 118 / 255),
                scale=1,
                scale_units="xy",
                angles="xy",
                width=0.01,
            )

            optaxis_right_zy = np.array(
                [optical_axis_vector_right[2], optical_axis_vector_right[1]]
            )
            optaxis_right_zy /= np.linalg.norm(optaxis_right_zy)
            optaxis_right_zy *= 50
            optaxes_right_zy_plot.remove()
            optaxes_right_zy_plot = inset_ax_optaxes_zy.quiver(
                eye_center_right[2],
                eye_center_right[1],
                optaxis_right_zy[0],
                optaxis_right_zy[1],
                color=(229 / 255, 111 / 255, 114 / 255),
                scale=1,
                scale_units="xy",
                angles="xy",
                width=0.01,
            )

        return (
            [optaxes_left_xz_plot]
            + [optaxes_right_xz_plot]
            + [optaxes_left_zy_plot]
            + [optaxes_right_zy_plot]
        )

    ani2 = animation.FuncAnimation(fig2, update2, interval=35, blit=True)

    plt.tight_layout()

    # Now, we get to the main animation callback that will be repeatedly called by matplotlib's
    # animation routines for each frame. It essentially:
    #
    # - Gets the latest data in the data_queue, as provided by the data acquisition thread.
    # - Converts the eyestate data into a format that is more easily used for plotting.
    # - Updates the plot objects with the new data.
    # - Finally, the function returns, passing the plot objects onto matplotlib's animation routine, so that it can draw
    #   the next frame to the plot.
    #
    # The frame_number argument is required by matplotlib, even if you do not use it in your function.
    def update1(frame_number):
        # Some matplotlib plotting objects need to be declared as global
        # to be accessible from the `update` function. A linter, such as Ruff,
        # can easily point out which ones need to be declared as global.
        # global optaxes_left_xz_plot
        # global optaxes_right_xz_plot
        # global optaxes_left_zy_plot
        # global optaxes_right_zy_plot

        # Without having a brief pause before updating images, the matplotlib animation stutters.
        # It is also required for properly updating OpenCV windows.
        # Here, we use OpenCV's waitKey function to introduce a small 1ms delay and
        # also check if the user pressed Esc, which will quit the visualization.
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return

        # Prepare variables to hold the latest data
        # and check if the queues have new values.
        # If new values are available, get them from the queues.
        eye_img = None
        gaze = None
        scene_img = None
        if not data_queue.empty():
            data = data_queue.get()
            eye_img = data["eye"]
            gaze = data["gaze"]
            scene_img = data["scene"]

        # if not scene_queue.empty():
        # scene_img = scene_queue.get()

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
            cv2.imshow("Eye Cameras - Press ESC to quit", eye_img)
            # eye_image_plot.set_data(eye_img)

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
            # frame. Otherwise, you will see many overlapping spheres. Comment
            # these two lines to see the effect.
            for artist in list(eyestate_ax.collections):
                artist.remove()

            # Use the ThreeDEyeModel's plot function to draw the full eyestate
            # in 3D!
            eye_left.plot(eyestate_ax, (101 / 255, 188 / 255, 118 / 255))
            eye_right.plot(eyestate_ax, (229 / 255, 111 / 255, 114 / 255))

            # # Plot an overhead view of the optical axes as arrows, using matplotlib's quiver.
            # # This can be used to qualitatively inspect vergence, for example.
            # optaxis_left_xz = np.array(
            #     [optical_axis_vector_left[0], optical_axis_vector_left[2]]
            # )

            # # Normalize and rescale the vector to make it easier to see in the
            # # plot.
            # optaxis_left_xz /= np.linalg.norm(optaxis_left_xz)
            # optaxis_left_xz *= 125

            # # Again, some matplotlib objects need to be manually removed, before you
            # # draw the next frame. Comment the `remove()` line below to see what happens,
            # # if you don't do this.
            # optaxes_left_xz_plot.remove()
            # optaxes_left_xz_plot = inset_ax_optaxes_xz.quiver(
            #     eye_center_left[0],
            #     eye_center_left[2],
            #     optaxis_left_xz[0],
            #     optaxis_left_xz[1],
            #     color=(101 / 255, 188 / 255, 118 / 255),
            #     scale=1,
            #     scale_units="xy",
            #     angles="xy",
            #     width=0.01,
            # )

            # # Plot the right optical axis vector in an overhead view.
            # optaxis_right_xz = np.array(
            #     [optical_axis_vector_right[0], optical_axis_vector_right[2]]
            # )
            # optaxis_right_xz /= np.linalg.norm(optaxis_right_xz)
            # optaxis_right_xz *= 125
            # optaxes_right_xz_plot.remove()
            # optaxes_right_xz_plot = inset_ax_optaxes_xz.quiver(
            #     eye_center_right[0],
            #     eye_center_right[2],
            #     optaxis_right_xz[0],
            #     optaxis_right_xz[1],
            #     color=(229 / 255, 111 / 255, 114 / 255),
            #     scale=1,
            #     scale_units="xy",
            #     angles="xy",
            #     width=0.01,
            # )

            # # We do similar for the side-profile view of the optical axes.
            # optaxis_left_zy = np.array(
            #     [optical_axis_vector_left[2], optical_axis_vector_left[1]]
            # )
            # optaxis_left_zy /= np.linalg.norm(optaxis_left_zy)
            # optaxis_left_zy *= 50
            # optaxes_left_zy_plot.remove()
            # optaxes_left_zy_plot = inset_ax_optaxes_zy.quiver(
            #     eye_center_left[2],
            #     eye_center_left[1],
            #     optaxis_left_zy[0],
            #     optaxis_left_zy[1],
            #     color=(101 / 255, 188 / 255, 118 / 255),
            #     scale=1,
            #     scale_units="xy",
            #     angles="xy",
            #     width=0.01,
            # )

            # optaxis_right_zy = np.array(
            #     [optical_axis_vector_right[2], optical_axis_vector_right[1]]
            # )
            # optaxis_right_zy /= np.linalg.norm(optaxis_right_zy)
            # optaxis_right_zy *= 50
            # optaxes_right_zy_plot.remove()
            # optaxes_right_zy_plot = inset_ax_optaxes_zy.quiver(
            #     eye_center_right[2],
            #     eye_center_right[1],
            #     optaxis_right_zy[0],
            #     optaxis_right_zy[1],
            #     color=(229 / 255, 111 / 255, 114 / 255),
            #     scale=1,
            #     scale_units="xy",
            #     angles="xy",
            #     width=0.01,
            # )

        # Finally, we return a list of plot objects, per
        # matplotlib's conventions.
        return (
            eyestate_ax.collections
            + [eye_left.optical_axis.optical_axis_quiver_plot]
            + [eye_right.optical_axis.optical_axis_quiver_plot]
            # + [eye_image_plot]
            # + [optaxes_left_xz_plot]
            # + [optaxes_right_xz_plot]
            # + [optaxes_left_zy_plot]
            # + [optaxes_right_zy_plot]
        )

    # Create a matplotlib animation function that will run the `update` function
    # every 33ms. This will update our animation at ~30 FPS. We pass `blit=False`,
    # because matplotlib's 3D functionality does not support blitting.
    ani1 = animation.FuncAnimation(fig1, update1, interval=35, blit=False)

    # This helps to make use of the full figure plotting real-estate.
    plt.tight_layout()

    # Finally, show the plot with the animation! :-)
    plt.show()

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # No matter what happens, we need to clean close our connection to Neon
    # and close all OpenCV windows.
    device.close()
    cv2.destroyAllWindows()
