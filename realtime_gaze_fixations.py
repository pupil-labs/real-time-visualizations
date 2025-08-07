import threading
import time
from queue import Queue

import cv2
import numpy as np
from pupil_labs.realtime_api.simple import Device
from pupil_labs.realtime_api.streaming.eye_events import (
    FixationEventData,
)

device = Device(address="192.168.1.35", port=8080)

scene_gaze_queue = Queue(maxsize=1)
fixations_queue = Queue(maxsize=1)


def scene_and_gaze_acquisition():
    while True:
        frame, gaze = device.receive_matched_scene_video_frame_and_gaze()

        data = {
            "frame": frame,
            "gaze": gaze,
        }

        if scene_gaze_queue.full():
            scene_gaze_queue.get_nowait()

        scene_gaze_queue.put_nowait(data)


def fixation_acquisition():
    while True:
        eye_event = device.receive_eye_events(timeout_seconds=0.3)

        if isinstance(eye_event, FixationEventData) and eye_event.event_type == 1:
            fixation_x = eye_event.mean_gaze_x
            fixation_y = eye_event.mean_gaze_y
            now = time.time()

            data = {
                "fixation_x": fixation_x,
                "fixation_y": fixation_y,
                "now": now,
            }

            if fixations_queue.full():
                fixations_queue.get_nowait()

            fixations_queue.put_nowait(data)


scene_and_gaze_thread = threading.Thread(target=scene_and_gaze_acquisition)
scene_and_gaze_thread.start()

fixation_thread = threading.Thread(target=fixation_acquisition)
fixation_thread.start()

draw_fixation = False
fixations_to_draw = []

cv2.namedWindow("Scene Camera - Press ESC to quit", cv2.WINDOW_NORMAL)

while True:
    frame = None
    if not scene_gaze_queue.empty():
        frame_data = scene_gaze_queue.get_nowait()
        frame = frame_data["frame"]
        gaze = frame_data["gaze"]

        print(gaze.x, gaze.y)

        cv2.circle(
            frame.bgr_pixels,
            (int(gaze.x), int(gaze.y)),
            radius=80,
            color=(0, 0, 255),
            thickness=15,
        )

    if not fixations_queue.empty():
        fixation_data = fixations_queue.get_nowait()

        fixation_x = fixation_data["fixation_x"]
        fixation_y = fixation_data["fixation_y"]
        now = fixation_data["now"]
        draw_fixation = True

    if (frame is not None) and draw_fixation:
        if np.abs(time.time() - now) < 3:
            cv2.circle(
                frame.bgr_pixels,
                (int(fixation_x), int(fixation_y)),
                radius=20,
                color=(255, 0, 0),
                thickness=15,
            )
        else:
            draw_fixation = False

    if frame:
        cv2.imshow("Scene Camera - Press ESC to quit", frame.bgr_pixels)
        if cv2.waitKey(1) & 0xFF == 27:
            break
