## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.yuyv, 30)

# Start streaming
pipeline_profile = pipeline.start(config)

device = pipeline_profile.get_device()
for sensor in device.query_sensors():
    if sensor.get_info(rs.camera_info.name) == "RGB Camera":
        print("Found RGB Camera")
            
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            print("Auto exposure disabled")
            sensor.set_option(rs.option.exposure, 100)
            sensor.set_option(rs.option.gain, 10)
            sensor.set_option(rs.option.contrast, 50)
            sensor.set_option(rs.option.brightness, 0)
            sensor.set_option(rs.option.gamma, 300)
            sensor.set_option(rs.option.hue, 0)
            sensor.set_option(rs.option.saturation, 64)
            sensor.set_option(rs.option.sharpness, 50)
            sensor.set_option(rs.option.enable_auto_white_balance, 1)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        frame_data = np.asanyarray(color_frame.get_data())
        print("received frame size:", frame_data.size)
        
        rgb_image = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
        
        # Convert images to numpy arrays
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)


        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', rgb_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
