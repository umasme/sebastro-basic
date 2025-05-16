import cv2
import numpy as np
import depthai as dai
import contextlib
import time
import pyrealsense2 as rs  # Added RealSense library

CAMERAS = []

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

    # Define sources and outputs
    imu = pipeline.create(dai.node.IMU)
    xlinkOut = pipeline.create(dai.node.XLinkOut)

    xlinkOut.setStreamName("imu")

    # Enable ACCELEROMETER_RAW at 500 Hz rate
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
    # Enable GYROSCOPE_RAW at 400 Hz rate
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)

    # Set batch report thresholds
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    # Link plugins IMU -> XLINK
    imu.out.link(xlinkOut.input)

    return pipeline

with contextlib.ExitStack() as stack:
        deviceInfos = dai.Device.getAllAvailableDevices()
        usbSpeed = dai.UsbSpeed.SUPER
        openVinoVersion = dai.OpenVINO.Version.VERSION_2021_4

        qRgbMap = []
        devices = []

        realsense_pipelines = []
        realsense_profiles = []

        ctx = rs.context()
        realSense_devices = ctx.query_devices()

        for deviceInfo in deviceInfos:
            deviceInfo: dai.DeviceInfo
            device: dai.Device = stack.enter_context(dai.Device(openVinoVersion, deviceInfo, usbSpeed))
            devices.append(device)
            print("===Connected to ", deviceInfo.getMxId())
            mxId = device.getMxId()
            cameras = device.getConnectedCameras()
            usbSpeed = device.getUsbSpeed()
            eepromData = device.readCalibration2().getEepromData()
            print("   >>> MXID:", mxId)
            print("   >>> Num of cameras:", len(cameras))
            print("   >>> USB speed:", usbSpeed)
            if eepromData.boardName != "":
                print("   >>> Board name:", eepromData.boardName)
            if eepromData.productName != "":
                print("   >>> Product name:", eepromData.productName)

            pipeline = createPipeline()
            device.startPipeline(pipeline)

            # Output queue for imu bulk packets
            imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)

            # Output queue will be used to get the rgb frames from the output defined above
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            stream_name = "rgb-" + mxId + "-" + eepromData.productName
            qRgbMap.append((q_rgb, stream_name, mxId))

            # Create resizable windows for each stream
            # cv2.namedWindow(stream_name, cv2.WINDOW_NORMAL)

        for cam_idx in range(1):  # For two RealSense cameras
            pipeline = rs.pipeline()
            config = rs.config()
            serial_number = realSense_devices[cam_idx].get_info(rs.camera_info.serial_number)
            device_serial = serial_number # change this later *yawn*
            print(f"Starting camera with serial number: {serial_number}")
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipeline.start(config)
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
            camera = {
                "pipeline": pipeline,
                "config": config,
                "device_serial": device_serial,
                "is_running": True
                }

            CAMERAS.append(camera)
            realsense_pipelines.append((pipeline, serial_number))
            realsense_profiles.append(profile)
    
        while True:
            
            color_images = []
            for pipeline, serial_number in realsense_pipelines:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_infrared_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    stream_name = f"realsense-{serial_number}"
                    mxId = f"realsense-{serial_number}"  # Fake ID, update CAMERA_INFOS if needed
                    color_images.append((color_image, stream_name, mxId))
                    cv2.imshow(stream_name, color_image)
            
            for q_rgb, stream_name, mxId in qRgbMap:
                if q_rgb.has():
                    color_image = q_rgb.get().getCvFrame()
                    color_images.append((color_image, stream_name, mxId))

            for color_image, stream_name, mxId in color_images:
                # Convert to grayscale for ArUco detection
                if mxId == "realsense-247122073398" or mxId == "realsense-327122073351":
                    gray_image = color_image
                    scaling_factor = 1.75
                else:
                    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                    scaling_factor = 0.71
            cv2.imshow(stream_name, color_image)

            if cv2.waitKey(1) == ord('q'):
                break
