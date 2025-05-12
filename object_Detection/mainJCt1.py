#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import contextlib
import time
import pyrealsense2 as rs  # Added RealSense library
from pysabertooth import Sabertooth
import linearactuator as LA
import obstDistCheckNoRGB2 as scan

''' TODO:
1. check theta returned from aruco detection is accurate
2. add intrinsic parameters for all four cameras
3. test with all four cameras
4. add code for encoders when aruco is not detected
5. add object detection
6. add object detection behavior
7. add object detection behavior to waypoints
8. 
'''
# Camera intrinsic parameters (from Matlab) for (left or right) oak-d-lite
fx1 = 1515.24261837315  # Focal length x
fy1 = 1513.21547841726  # Focal length y
cx1 = 986.009156502993   # Optical center x
cy1 = 551.618039270305   # Optical center y
camera_matrix1 = np.array([[fx1, 0, cx1],
                           [0, fy1, cy1],
                           [0, 0, 1]], dtype=float)

# Distortion coefficients for (left or right) oak-d-lite
dist_coeffs1 = np.array((0.114251294509202,-0.228889968220235,0,0))

relative_position1 = [0.145, 0, 180] # Relative position of the camera with respect to the robot base (X, Y, Theta)
relative_position2 = [-0.145, 0, 0]  # Relative position of the camera with respect to the robot base (X, Y, Theta)

# Camera intrinsic parameters for the other (left or right) oak-d-lite (assumed the same for both cameras for now (03/18/25) update when other cameras are calibrated)
fx2 = 1515.24261837315  # Focal length x
fy2 = 1513.21547841726  # Focal length y
cx2 = 986.009156502993   # Optical center x
cy2 = 551.618039270305   # Optical center y
camera_matrix2 = np.array([[fx2, 0, cx2],
                           [0, fy2, cy2],
                           [0, 0, 1]], dtype=float)

# Distortion coefficients for (left or right) oak-d-lite
dist_coeffs2 = np.array((0.114251294509202,-0.228889968220235,0,0))

# Default camera intrinsic parameters for RealSense D435i (assumed the same for both cameras for now (03/18/25) update when other cameras are calibrated)
camera_matrix3 = np.array([[1384, 0, 960],
                           [0, 1384, 540],
                           [0, 0, 1]], dtype=float) 

dist_coeffs3 = np.array((0, 0, 0, 0))

relative_position3 = [0, 0.145, 90] # Relative position of the camera with respect to the robot base (X, Y, Theta)  
relative_position4 = [0, -0.145, 270] # Relative position of the camera with respect to the robot base (X, Y, Theta)

CAMERAS = []

CAMERA_INFOS = {
 "14442C10911DC5D200" : {"camera_matrix" : camera_matrix1, "dist_coeffs" : dist_coeffs1, "relative_position" : relative_position1},
 "14442C1071EDDFD600" : {"camera_matrix" : camera_matrix2, "dist_coeffs" : dist_coeffs2, "relative_position" : relative_position2},
 "realsense-247122073398": {"camera_matrix": camera_matrix3, "dist_coeffs": dist_coeffs3, "relative_position": relative_position3},
 "realsense-327122073351": {"camera_matrix": camera_matrix3, "dist_coeffs": dist_coeffs3, "relative_position": relative_position4},
}

WAYPOINTS = [[50, -2, "mine"], [1.5, -2, "deposit"], [1, -1, "deposit"]]  # Updated waypoints

marker_size = 0.11

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_ITERATIVE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

# Define ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

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

def timeDeltaToMilliS(delta) -> float:
        return delta.total_seconds() * 1000

def localize(color_images, imuQueue, aruco_detector, marker_size, baseTs, prev_gyroTs, camera_position, pose):
    last_print_time = time.time()  # Initialize time tracking

    imuData = imuQueue.get()  # Blocking call, will wait until new data has arrived
    imuPackets = imuData.packets
    for color_image, stream_name, mxId in color_images:
        # Convert to grayscale for ArUco detection
        if mxId == "realsense-247122073398" or mxId == "realsense-327122073351":
            gray_image = color_image
        else:
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = aruco_detector.detectMarkers(gray_image)

        camera_matrix = CAMERA_INFOS[str(mxId)]["camera_matrix"]
        dist_coeffs = CAMERA_INFOS[str(mxId)]["dist_coeffs"]

        # Process each detected marker and get pose relative to id 2
        if ids is not None and 2 in ids:
            arr = np.where(ids == 2)[0][0]
            corners = np.array(corners[arr])
            ids = np.array(ids[arr])
            # Get the center of the marker
            center = np.mean(corners[0], axis=0).astype(int)

            rvec, tvec, _ = my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

            rvec = np.array(rvec)
            tvec = np.array(tvec)

            # Draw the marker and axes
            cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            rotation_matrix, _ = cv2.Rodrigues(rvec[0])  # Convert rotation vector to rotation matrix using the rodrigues formula
            R_inv = rotation_matrix.T  # Inverse of the rotation matrix
            camera_position = R_inv @ tvec[0]  # @ is the matrix multiplication operator in Python
            theta = np.arcsin(-R_inv[2][0])
            theta = np.degrees(theta)  # Convert radians to degrees for readability

            pose = [camera_position[0][0], camera_position[2][0], theta]  # Pose in the format [x, y, theta]
            pose = pose + np.array(CAMERA_INFOS[str(mxId)]["relative_position"])  # Adjust pose based on relative position of the camera
            pose[2] = pose[2] % 360  # Normalize theta to be between 0 and 360 degrees

        elif camera_position is not None:
            for imuPacket in imuPackets:
                acceleroValues = imuPacket.acceleroMeter
                gyroValues = imuPacket.gyroscope

                acceleroTs = acceleroValues.getTimestampDevice()
                gyroTs = gyroValues.getTimestampDevice()

                if baseTs is None:
                    baseTs = acceleroTs if acceleroTs < gyroTs else gyroTs
                    prev_gyroTs = gyroTs
                    print(baseTs)

                acceleroTs = timeDeltaToMilliS(acceleroTs - baseTs)
                gyroTs = timeDeltaToMilliS(gyroTs - baseTs)

                imuF = "{:.06f}"
                tsF = "{:.03f}"

                # Calculate the time difference between the current and previous gyroscope readings
                dt = (gyroTs - timeDeltaToMilliS(prev_gyroTs - baseTs)) / 1000.0  # Convert milliseconds to seconds
                prev_gyroTs = gyroValues.getTimestampDevice()

                gyroValues = round(gyroValues.x, 2), round(gyroValues.y, 2), round(gyroValues.z, 2)

                # Integrate the gyroscope data to get the angles
                pose[2] += np.degrees(gyroValues[1]) * dt  # Pitch

        current_time = time.time()
        if current_time - last_print_time >= 1 and camera_position is not None:
            print(f"Camera Position: {pose}")
            last_print_time = current_time  # Update last print time

        # Display the output image
        cv2.imshow(stream_name, color_image)

    return pose, baseTs, prev_gyroTs, camera_position

def turn_to(theta):
    if pose[2] - theta > 180:
         turn_left(50)
         #print(f"Turning left to {theta}")
    elif pose[2] - theta < 180:
         turn_right(50)  # Adjust speed as necessary for turning, 20 is an example speed
         #print(f"Turning right to {theta}")

def move_to(current_position, target_position):
    theta = np.degrees(np.arctan2(target_position[1] - current_position[1], target_position[0] - current_position[0]))
    theta -= 90  # Adjust for camera orientation
    theta = theta % 360  # Normalize theta to be between 0 and 360 degrees
    if abs(current_position[2] - theta) > 5:
        turn_to(theta)
    else:
        linear_motion(20)
        print("Moving forward")


def excavate(initial_time):
    lowering_time = 5  # Duration of lowering trencher in seconds
    excavate_time = 10  # Duration of excavation in seconds
    raising_time = 5  # Duration of raising trencher in seconds
    if time.time() - initial_time < lowering_time:
        print("Lowering trencher")
        LA.move(-1)  # Lower the trencher
        return False  # Excavation is still in progress
    elif time.time() - initial_time < (lowering_time + excavate_time):
         LA.stop()
         construction_motors.drive(1, 100)
         construction_motors.drive(2, 10)
         linear_motion(10)  # Move forward while excavating
         print("Excavating")
         return False
    elif time.time() - initial_time < (lowering_time + excavate_time + raising_time):
         print("Raising trencher")
         LA.move(1)  # Raise the trencher
         construction_motors.drive(1, 50)  # continue the excavation motor to deposit remaining regolith
    else: 
        print("Excavation complete")
        LA.stop()
        construction_motors.drive(1, 0)  # Stop the excavation motor
        construction_motors.drive(2, 0)  # Stop the deposition motor
        return True 
    
def deposit(initial_time):
    deposit_time = 5
    if time.time() - initial_time < deposit_time:
        print("Depositing")
        construction_motors.drive(1, 50)	# drive deposition motor
        return False
    else:
        print("Deposit complete")
        stop_all()
        return True
    
def stop_all():
	motor1.stop()			# Turn off both motors
	motor2.stop()	

def linear_motion(speed:int):
	## Motor 1
	motor1.drive(1,speed)	# Turn on motor 1
	motor1.drive(2,speed)	# Turn on motor 2

	time.sleep(0.01)

	## Motor 2
	motor2.drive(1, -speed)	# Turn on motor 1
	motor2.drive(2, -speed)	# Turn orealsense-247122073398"n motor 2

def turn_left(speed:int):
	
	## Motor 1
	motor1.drive(1,-speed)	# Turn on motor 1
	motor1.drive(2,speed)	# Turn on motor 2

	time.sleep(0.01)

	## Motor 2
	motor2.drive(1,-speed)	# Turn on motor 1
	motor2.drive(2,speed)	# Turn on motor 2

def turn_right(speed:int):
    	## Motor 1
	motor1.drive(1, speed)	# Turn on motor 1
	motor1.drive(2,-speed)	# Turn on motor 2

	time.sleep(0.01)

	## Motor 2
	motor2.drive(1,speed)	# Turn on motor 1
	motor2.drive(2,-speed)	# Turn on motor 2


motor1 = Sabertooth("/dev/ttyAMA0", baudrate = 9600, address = 129)	# Init the Motor
motor1.open()								# Open then connection
print(f"Connection Status: {motor1.saber.is_open}")			# Let us know if it is open
motor1.info()								# Get the motor info


## Init up the sabertooth 2, and open the seral connection 
motor2 = Sabertooth("/dev/ttyAMA0", baudrate = 9600, address = 134)	# Init the Motor
motor2.open()								# Open then connection
print(f"Connection Status: {motor2.saber.is_open}")			# Let us know if it is open
motor2.info()								# Get the motor info

construction_motors = Sabertooth("/dev/ttyAMA0", baudrate = 9600, address = 128)	# Init the Motor
construction_motors.open()								# Open then connection
print(f"Connection Status: {construction_motors.saber.is_open}")			# Let us know if it is open
construction_motors.info()								# Get the motor info

LA = LA.linearactuator()		# Init the linear actuator
try:
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

        for cam_idx in range(2):  # For two RealSense cameras
            pipeline = rs.pipeline()
            config = rs.config()
            serial_number = realSense_devices[cam_idx].get_info(rs.camera_info.serial_number)
            device_serial = serial_number # change this later *yawn*
            print(f"Starting camera with serial number: {serial_number}")
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            config.enable_stream(rs.stream.depth, 1, 640, 480, rs.format.z16, 30)
            profile = pipeline.start(config)
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
            camera = {
                "pipeline": pipeline,
                "config": config,
                "device_serial": device_serial,
                "is_running": True
                }

            CAMERAS.append(camera)
            realsense_pipelines.append((pipeline, serial_number))
            realsense_profiles.append(profile)
        
        last_print_time = time.time()

        mining = False
        depositing = False

        baseTs = None
        prev_gyroTs = None
        pose = None  # Initialize pose as a list with [x, y, theta]
        camera_position = None

        # Add these variables to the initialization section before the while loop:

        # Waypoint constants - defining the key waypoints
        MIDPOINT_INDEX = 1
        EXCAVATION_INDEX = 3
        DEPOSITION_INDEX = 5

        # Initial waypoints defined as [x, y, type]
        WAYPOINTS = [
            [0, 0, "temporary"],  # Temporary waypoint for initial position
            [5, 5, "midpoint"],     # Index 0: Midpoint
            [0, 0, "temporary"],  # Temporary waypoint for initial position
            [50, -2, "mine"],       # Index 1: Excavation zone
            [0, 0, "temporary"],  # Temporary waypoint for initial position
            [1.5, -2, "deposit"]    # Index 2: Deposition zone
        ]

        # Navigation state variables
        current_waypoint_index = 0  # Start at midpoint
        destination_index = MIDPOINT_INDEX  # First destination is excavation zone
        last_permanent_waypoint = MIDPOINT_INDEX  # Start at midpoint
        initial_position_reached = False  # Haven't reached the initial midpoint yet

        # Obstacle detection variables
        scanning_in_progress = False
        need_to_scan = True  # Need to scan for obstacles initially
        scan_stage = 0
        at_waypoint = False  # Not at a waypoint initially
        i = 0  # Used for excavation/deposit timing

        # Helper functions for waypoint navigation
        def calculate_angle_to_waypoint(current_pos, target_pos):
            # Calculate angle from current position to target waypoint            theta = np.degrees(np.arctan2(target_pos[1] - current_pos[1], target_pos[0] - current_pos[0]))
            theta -= 90  # Adjust for camera orientation
            theta = theta % 360  # Normalize theta to be between 0 and 360 degrees
            return theta
        
        def calculate_distance_to_waypoint(current_pos, target_pos):
            # Calculate Euclidean distance to waypoint
            return np.sqrt((target_pos[0] - current_pos[0])**2 + (target_pos[1] - current_pos[1])**2)



        while True:
            color_images = []
            for pipeline, serial_number in realsense_pipelines:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_infrared_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    stream_name = f"realsense-{serial_number}"
                    mxId = f"realsense-{serial_number}"
                    color_images.append((color_image, stream_name, mxId))
                    cv2.imshow(stream_name, color_image)
            
            for q_rgb, stream_name, mxId in qRgbMap:
                if q_rgb.has():
                    color_image = q_rgb.get().getCvFrame()
                    color_images.append((color_image, stream_name, mxId))

            # Pass all required arguments to the localize function
            pose, baseTs, prev_gyroTs, camera_position = localize(
                color_images, imuQueue, aruco_detector, marker_size, baseTs, prev_gyroTs, camera_position, pose
            )
            
            if cv2.waitKey(1) == ord('q'):
                break

            if pose is None:
                turn_left(20)  # If pose is None, rotate to find ArUco markers
                print("Pose is None, rotating to find ArUco markers")
                continue
            
            
            # Check if we have reached the current waypoint
            if not at_waypoint:
                print("checking if not at waypoint")
                if abs(pose[0] - WAYPOINTS[current_waypoint_index][0]) <= 0.5 and abs(pose[1] - WAYPOINTS[current_waypoint_index][1]) <= 0.5:
                    print(f"Arrived at waypoint {current_waypoint_index} ({WAYPOINTS[current_waypoint_index][2]}) at position {pose}.")
                    stop_all()
                    at_waypoint = True
                    
                    # If this is a temporary waypoint, remove it and continue to next waypoint
                    if WAYPOINTS[current_waypoint_index][2] == "temporary":
                        print("This was a temporary waypoint, continuing to destination.")
                        current_waypoint_index += 1
                        at_waypoint = False
                continue
            
            # Handle waypoint activities if we're at a waypoint
            if at_waypoint:
                print("At waypoint, checking actions...")
                # Process waypoint actions based on waypoint type
                if WAYPOINTS[current_waypoint_index][2] == "mine":
                    if i == 0:
                        initial_excavation_time = time.time()
                        i += 1
                        print("Starting excavation...")
                        
                    if excavate(initial_excavation_time):
                        at_waypoint = False
                        i = 0
                        
                        # After mining, set destination to midpoint
                        destination_index = MIDPOINT_INDEX
                        need_to_scan = True
                                        
                elif WAYPOINTS[current_waypoint_index][2] == "deposit":
                    if i == 0:
                        initial_deposit_time = time.time()
                        i += 1
                        print("Starting deposition...")

                    if deposit(initial_deposit_time):
                        at_waypoint = False
                        i = 0
                        
                        # After depositing, set destination to midpoint
                        destination_index = MIDPOINT_INDEX
                        need_to_scan = True
                        
                elif WAYPOINTS[current_waypoint_index][2] == "midpoint":
                    # At midpoint, decide where to go next
                    at_waypoint = False
                    need_to_scan = True
                    if not initial_position_reached:
                        # First time reaching midpoint, mark it and go to excavation
                        initial_position_reached = True
                        print("Initial midpoint reached for the first time. Starting navigation cycle.")
                        
                        # Next, go to excavation
                        destination_index = EXCAVATION_INDEX
                        last_permanent_waypoint = MIDPOINT_INDEX
                    
                    elif last_permanent_waypoint == EXCAVATION_INDEX:
                        # We came from excavation, now go to deposition
                        destination_index = DEPOSITION_INDEX
                        print("Going to deposition after excavation.")
                        
                        
                    elif last_permanent_waypoint == DEPOSITION_INDEX:
                        # We came from deposition, go back to excavation
                        destination_index = EXCAVATION_INDEX
                        print("Going to excavation after deposition.")
                        
                continue
            
            
            # Obstacle scanning and avoidance logic for discovery mode
            if need_to_scan:
                if not scanning_in_progress:
                    print("Need to scan for obstacles, starting scan...")
                    # Initialize scan
                    scanning_in_progress = True
                    scan_stage = 0  # 0: initial rotation, 1: front scan, 2: left scan, 3: back scan, 4: create waypoint
                    original_heading = pose[2]
                    target_angle = calculate_angle_to_waypoint(pose, WAYPOINTS[destination_index])
                    print(f"Starting obstacle scan, target angle: {target_angle}")
                    
                # Stage 0: Rotate to face target waypoint
                if scan_stage == 0:
                    print("if scan stage 0")
                    if abs(pose[2] - target_angle) > 5:
                        turn_to(target_angle)
                        print(f"Turning to target angle: {target_angle}, current angle: {pose[2]}")
                    else:
                        print("Now facing target waypoint, checking front obstacles")
                        scan_stage = 1
                
                # Stage 1: Front scan
                elif scan_stage == 1:
                    print("if scan stage 1")
                    # Check front obstacle using the front camera (realsense-327122073351)
                    front_distance = scan.detect_obstacles(247122073398, cameras=CAMERAS)
                    print(f"Front obstacle detection: {front_distance} meters")
                    
                    if front_distance is None or front_distance > 0.5:
                        # Path is clear, calculate safe distance to move
                        if front_distance is None:
                            safe_move_distance = 1.0  # Default distance if no obstacle detected
                        else:
                            safe_move_distance = min(front_distance - 0.3, 1.0)  # Leave 30cm safety margin, max 1m movement
                        
                        # Check distance to target waypoint
                        waypoint_distance = calculate_distance_to_waypoint(pose, WAYPOINTS[destination_index])
                        move_distance = min(safe_move_distance, waypoint_distance)
                        
                        # Don't create temp waypoint if we're close enough to destination
                        if move_distance == waypoint_distance and waypoint_distance < 1.5:
                            # We're close to destination, move directly there
                            current_waypoint_index = destination_index
                            scanning_in_progress = False
                            need_to_scan = False
                            print(f"Close to destination, moving directly to {WAYPOINTS[destination_index]}")
                        else:
                            # Create temporary waypoint directly in front
                            temp_waypoint_x = pose[0] + move_distance * np.cos(np.radians(pose[2]))
                            temp_waypoint_y = pose[1] + move_distance * np.sin(np.radians(pose[2]))
                            
                            # Insert the temporary waypoint
                            temp_waypoint = [temp_waypoint_x, temp_waypoint_y, "temporary"]
                            insert_index = destination_index-1  # Add to end of waypoints list
                            WAYPOINTS[insert_index] = temp_waypoint
                            print(f"Created temporary waypoint in front at {temp_waypoint}")
                            
                            # Move to the temporary waypoint
                            current_waypoint_index = insert_index
                            
                        scanning_in_progress = False
                        need_to_scan = False
                        scan_stage = 0  # Reset for next scan
                    else:
                        # Obstacle in front, need to check left
                        print("Obstacle in front, checking left side...")
                        scan_stage = 2
                        target_angle = (pose[2] + 90) % 360  # Turn 90 degrees CCW
                        
                # Stage 2: Turn and scan left
                elif scan_stage == 2:
                    # First, turn to face left
                    if abs(pose[2] - target_angle) > 5:
                        turn_to(target_angle)
                    else:
                        # Check left obstacle using front camera
                        left_distance = scan.detect_obstacles(247122073398, cameras=CAMERAS)
                        print(f"Left obstacle detection: {left_distance} meters")
                        
                        if left_distance is None or left_distance > 0.5:
                            # Left is clear, create temporary waypoint
                            if left_distance is None:
                                safe_move_distance = 1.0
                            else:
                                safe_move_distance = min(left_distance - 0.3, 1.0)
                            
                            # Create temporary waypoint to the left
                            temp_waypoint_x = pose[0] + safe_move_distance * np.cos(np.radians(pose[2]))
                            temp_waypoint_y = pose[1] + safe_move_distance * np.sin(np.radians(pose[2]))
                            
                            # Insert waypoint
                            temp_waypoint = [temp_waypoint_x, temp_waypoint_y, "temporary"]
                            insert_index = destination_index-1  # Add to end of waypoints list
                            WAYPOINTS[insert_index] = temp_waypoint
                            # Move to the temporary waypoint
                            current_waypoint_index = insert_index
                            scanning_in_progress = False
                            need_to_scan = False
                            scan_stage = 0  # Reset for next scan
                            print(f"Created temporary waypoint to the left at {temp_waypoint}")
                        else:
                            # Left is blocked too, check back
                            print("Left side blocked, checking back...")
                            scan_stage = 3

                
                # Stage 3: Scan back while facing left
                elif scan_stage == 3:
                    # Check back obstacle using back camera
                    back_distance = scan.detect_obstacles(327122073351, cameras=CAMERAS)
                    print(f"Back obstacle detection: {back_distance} meters")
                    
                    if back_distance is None or back_distance > 0.5:
                        # Back is clear, turn to face back direction
                        scan_stage = 4
                        target_angle = (original_heading + 180) % 360  # Turn to face back
                    else:
                        # Back is blocked too, need to check right
                        print("Back direction blocked, SeNDING IT...")
                        scan_stage = 5
                        target_angle = (original_heading - 90) % 360  # Turn CW 90
                        
                        
                # Stage 4: Turn to face back and create waypoint
                elif scan_stage == 4:
                    if abs(pose[2] - target_angle) > 5:
                        turn_to(target_angle)
                    else:
                        # Now facing back, create temporary waypoint
                        if back_distance is None:
                            safe_move_distance = 1.0
                        else:
                            safe_move_distance = min(back_distance - 0.3, 1.0)
                        
                        # Create temporary waypoint
                        temp_waypoint_x = pose[0] + safe_move_distance * np.cos(np.radians(pose[2]))
                        temp_waypoint_y = pose[1] + safe_move_distance * np.sin(np.radians(pose[2]))
                        
                        # Insert waypoint
                        temp_waypoint = [temp_waypoint_x, temp_waypoint_y, "temporary"]
                        insert_index = destination_index-1  # Add to end of waypoints list
                        WAYPOINTS[insert_index] = temp_waypoint
                        
                        # Move to the temporary waypoint
                        current_waypoint_index = insert_index
                        scanning_in_progress = False
                        need_to_scan = False
                        scan_stage = 0  # Reset for next scan
                        print(f"Created temporary waypoint going backward at {temp_waypoint}")
                
                # Stage 5: SeNDING IT...
                elif scan_stage == 4:
                    if abs(pose[2] - target_angle) > 5:
                        turn_to(target_angle)
                    else:
                        # Now facing FRONT, create temporary waypoint
                        safe_move_distance = 1.0
                        # Create temporary waypoint
                        temp_waypoint_x = pose[0] + safe_move_distance * np.cos(np.radians(pose[2]))
                        temp_waypoint_y = pose[1] + safe_move_distance * np.sin(np.radians(pose[2]))
                        
                        # Insert waypoint
                        temp_waypoint = [temp_waypoint_x, temp_waypoint_y, "temporary"]
                        insert_index = destination_index-1  # Add to end of waypoints list
                        WAYPOINTS[insert_index] = temp_waypoint
                        
                        # Move to the temporary waypoint
                        current_waypoint_index = insert_index
                        scanning_in_progress = False
                        need_to_scan = False
                        scan_stage = 0  # Reset for next scan
                        print(f"Created temporary waypoint going backward at {temp_waypoint}")
                
                continue
            
            # Standard waypoint navigation when not using known routes or scanning
            if not scanning_in_progress and not need_to_scan:
                # If we reach the current waypoint, we need to scan for the next move
                if abs(pose[0] - WAYPOINTS[current_waypoint_index][0]) <= 0.5 and abs(pose[1] - WAYPOINTS[current_waypoint_index][1]) <= 0.5:
                    # Current waypoint reached, scan for next move
                    need_to_scan = True
                else:
                    # Continue moving to current waypoint
                    move_to(pose, WAYPOINTS[current_waypoint_index])
                
                current_time = time.time()
                if current_time - last_print_time >= 1:
                    print(f"Moving to waypoint {current_waypoint_index} at position {WAYPOINTS[current_waypoint_index]}, current position {pose}")
                    last_print_time = current_time
            
            

except Exception as e:
    print(f"An error occurred: {e}")
    stop_all()
finally:
     stop_all()
