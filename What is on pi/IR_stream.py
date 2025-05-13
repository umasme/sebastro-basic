import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

profile = pipeline.start(config)

device = profile.get_device()
depth_sensor = device.first_depth_sensor()
depth_sensor.set_option(rs.option.emitter_enabled, 0)

try:
    while True:
        frames = pipeline.wait_for_frames()
        ir_frame = frames.get_infrared_frame(1)
        ir_image = np.asanyarray(ir_frame.get_data())
        cv2.imshow("IR(Y8", ir_image)
        
        if cv2.waitKey(1) == 13:
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
     
        if cv2.waitKey(1) == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
