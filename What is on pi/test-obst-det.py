#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import contextlib
import time
import pyrealsense2 as rs
import obstDistCheckNoRGB2 as scan

# Camera intrinsic parameters for RealSense D435i
camera_matrix3 = np.array([[1384, 0, 960],
                           [0, 1384, 540],
                           [0, 0, 1]], dtype=float)

dist_coeffs3 = np.array((0, 0, 0, 0))

CAMERAS = []  # List to store camera configurations

def createPipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    # Define a source - color camera
    camRgb = pipeline.create(dai.node.ColorCamera)

    camRgb.setPreviewSize(640, 480)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)

    # Create output
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    return pipeline

def scan_for_obstacles():
    """
    Function to scan for obstacles using the `obstDistCheckNoRGB2` module.
    """
    print("Scanning for obstacles...")
    front_distance = scan.detect_obstacles(camera_serial=247122073398, cameras=CAMERAS)
    rear_distance = scan.detect_obstacles(camera_serial=327122073351, cameras=CAMERAS)
    print(f"Front distance: {front_distance}, Rear distance: {rear_distance}")
    return front_distance, rear_distance

try:
    with contextlib.ExitStack() as stack:
        # Initialize RealSense cameras
        ctx = rs.context()
        realSense_devices = ctx.query_devices()

        realsense_pipelines = []
        realsense_profiles = []

        for cam_idx in range(len(realSense_devices)):  # For all connected RealSense cameras
            pipeline = rs.pipeline()
            config = rs.config()
            device_serial = realSense_devices[cam_idx].get_info(rs.camera_info.serial_number)
            print(f"Starting camera with serial number: {device_serial}")
            config.enable_device(device_serial)
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipeline.start(config)
            camera = {
                "pipeline": pipeline,
                "config": config,
                "device_serial": device_serial,
                "is_running": True
                }

            CAMERAS.append(camera)
            realsense_pipelines.append((pipeline, device_serial))
            realsense_profiles.append(profile)

        print("Press Enter to scan for obstacles. Press 'q' to quit.")

        while True:
            color_images = []
            for pipeline, serial_number in realsense_pipelines:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_infrared_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    stream_name = f"realsense-{serial_number}"
                    color_images.append((color_image, stream_name))
                    cv2.imshow(stream_name, color_image)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit if 'q' is pressed
                break
            elif key == 13:  # Enter key pressed
                front_obstacle = scan.detect_obstacles(247122073398, cameras=CAMERAS, visualize = False)
                rear_obstacle = scan.detect_obstacles(327122073351, cameras=CAMERAS, visualize = False)
                print(f"Front obstacle distance: {front_obstacle}")
                print(f"Rear obstacle distance: {rear_obstacle}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Stop all pipelines and close OpenCV windows
    for pipeline, _ in realsense_pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()
