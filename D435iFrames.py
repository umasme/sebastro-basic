import pyrealsense2 as rs
import numpy as np
import sys
import cv2

def get_frame_d4(device_serial):
    
    """
    starts selected d435i camera and displays the depth and infrared stream 
    Alex mentioned that color stream is broken for linux, so we are not using it
    
    Args:
        serial (str): Serial number of the specific camera to use.
    Returns:
        TBD - prob just give camera frame
    """
    
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Use specific device serial provided
        config.enable_device(device_serial)
            
        # Configure streams, infrared can probably be removed
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                
        # Start the pipeline
        profile = pipeline.start(config)
        
        while True:
        
            color_images = []

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_infrared_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                stream_name = f"realsense-{device_serial}"
                mxId = f"realsense-{device_serial}"  # Fake ID, update CAMERA_INFOS if needed
                color_images.append((color_image, stream_name, mxId))
                cv2.imshow(stream_name, color_image)
            
            #not sure why but code breaks if this is not here
            if cv2.waitKey(1) == ord('q'): 
                break
        
    except Exception as e:
        print(f"cooked")
        sys.exit(1)
        
if __name__ == "__main__":
    get_frame_d4("327122073351")
    #get_frame_d4("247122073398")
