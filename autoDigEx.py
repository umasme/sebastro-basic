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
4. add code for encoders when aruco is not detected
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

relative_position1 = [0.145, 0, 90] # Relative position of the camera with respect to the robot base (X, Y, Theta)
relative_position2 = [-0.145, 0, 270]  # Relative position of the camera with respect to the robot base (X, Y, Theta)

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
camera_matrix3 = np.array([[393.5206, 0, 323.4011],
                           [0, 394.0078, 241.6593],
                           [0, 0, 1]], dtype=float) 

dist_coeffs3 = np.array((0.0337, -0.0349, 0, 0))

relative_position3 = [0, 0.145, 180] # Relative position of the camera with respect to the robot base (X, Y, Theta)  
relative_position4 = [0, -0.145, 0] # Relative position of the camera with respect to the robot base (X, Y, Theta)

CAMERAS = [] # this can be incorporated with the other stuff but i am so gosh darn tired right now

CAMERA_INFOS = {
 "14442C10911DC5D200" : {"camera_matrix" : camera_matrix1, "dist_coeffs" : dist_coeffs1, "relative_position" : relative_position1},
 "14442C1071EDDFD600" : {"camera_matrix" : camera_matrix2, "dist_coeffs" : dist_coeffs2, "relative_position" : relative_position2},
 "realsense-247122073398": {"camera_matrix": camera_matrix3, "dist_coeffs": dist_coeffs3, "relative_position": relative_position3},
 "realsense-327122073351": {"camera_matrix": camera_matrix3, "dist_coeffs": dist_coeffs3, "relative_position": relative_position4},
}

WAYPOINTS = [[-1.285, 6.84, "mine"], [-3.57, 6.84, "deposit"]]  # Updated waypoints

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
            scaling_factor = 1.75
        else:
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            scaling_factor = 0.71

        # Detect ArUco markers
        corners, ids, _ = aruco_detector.detectMarkers(gray_image)

        camera_matrix = CAMERA_INFOS[str(mxId)]["camera_matrix"]
        dist_coeffs = CAMERA_INFOS[str(mxId)]["dist_coeffs"]
        if ids is None and current_position is None:
            turn_left(30)
            print("turning to ArUco")
        # Process each detected marker and get pose relative to id 2
        if ids is not None and 2 in ids:
            arr = np.where(ids == 2)[0][0]
            corners = np.array(corners[arr])
            ids = np.array(ids[arr])
            # Get the center of the marker
            center = np.mean(corners[0], axis=0).astype(int)

            rvec, tvec, _ = my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

            rvec = np.array(rvec)
            tvec = np.array(tvec) * scaling_factor

            # Draw the marker and axes
            cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            rotation_matrix, _ = cv2.Rodrigues(rvec[0])  # Convert rotation vector to rotation matrix using the rodrigues formula
            R_inv = rotation_matrix.T  # Inverse of the rotation matrix
            camera_position = R_inv @ tvec[0]  # @ is the matrix multiplication operator in Python
            theta = np.arcsin(-R_inv[2][0])
            theta = np.degrees(theta)  # Convert radians to degrees for readability

            pose = [camera_position[0][0], abs(camera_position[2][0]), theta]  # Pose in the format [x, y, theta]
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
        #cv2.imshow(stream_name, color_image)
        #cv2.waitKey(1)

    return pose, baseTs, prev_gyroTs, camera_position

def turn_to(theta, pose):
    # Calculate the shortest turning direction
    delta_theta = (theta - pose[2] + 540) % 360 - 180
    if delta_theta > 0:
        turn_right(30)  # Turn right if the shortest path is clockwise
        # Turn left if the shortest path is counterclockwise
    else:
        turn_left(30)

def move_to(current_position, target_position):
    
    dx = target_position[0] - current_position[0]
    dy = target_position[1] - current_position[1]
    angle_rad = np.arctan2(dy, dx)
    angle_from_y_rad = angle_rad - np.pi / 2
    angle_from_y_deg = np.degrees(angle_from_y_rad)

    if target_position[2] == "deposit":
        theta = (angle_from_y_deg +180) % 360
        speed = -20
    else:
        theta = angle_from_y_deg % 360
        speed = 20
    delta_theta = (theta - pose[2] + 540) % 360 -180
    if abs(delta_theta) > 40:
        turn_to(theta, pose)
    else:
        linear_motion(speed)
        print("Moving forward")

def excavate(initial_time):
    lowering_time = 5  # Duration of lowering trencher in seconds
    excavate_time = 10  # Duration of excavation in seconds
    raising_time = 5  # Duration of raising trencher in seconds
    if time.time() - initial_time < lowering_time:
        print("Lowering trencher")
        LA.move(1)  # Lower the trencher
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
         LA.move(-1)  # Raise the trencher
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
	motor1.drive(1,-speed)	# Turn on motor 1
	motor1.drive(2,-speed)	# Turn on motor 2

	time.sleep(0.01)

	## Motor 2
	motor2.drive(1, speed)	# Turn on motor 1
	motor2.drive(2, speed)	# Turn orealsense-247122073398"n motor 2

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
        
        last_print_time = time.time()

        mining = False
        depositing = False
        baseTs = None
        prev_gyroTs = None
        pose = None  # Initialize pose as a list with [x, y, theta]
        camera_position = None
        i = 0
        waypoint = 0
        at_waypoint = True  # Flag to indicate if the robot has reached the current waypoint\
        initially_turning = True  # Flag to indicate if the robot is initially turning , determines whether to scan for obstacles or not
        current_position = None
    

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
            
            for q_rgb, stream_name, mxId in qRgbMap:
                if q_rgb.has():
                    color_image = q_rgb.get().getCvFrame()
                    color_images.append((color_image, stream_name, mxId))

            # Pass all required arguments to the localize function
            pose, baseTs, prev_gyroTs, camera_position = localize(
                color_images, imuQueue, aruco_detector, marker_size, baseTs, prev_gyroTs, camera_position, pose
            )
            
            if pose is None:
                turn_left(20)  # If pose is None, rotate to find ArUco markers
                #print("rotating to find ArUco markers...")


            elif pose is not None:
                if not at_waypoint:
                    current_position = pose
                    target_position = WAYPOINTS[waypoint]
                    dx = target_position[0] - current_position[0]
                    dy = target_position[1] - current_position[1]
                    angle_rad = np.arctan2(dy, dx)
                    angle_from_y_rad = angle_rad - np.pi / 2
                    angle_from_y_deg = np.degrees(angle_from_y_rad)
                    target_x = target_position[0]
                    target_y = target_position[1]
                    current_x = pose[0]
                    current_y = pose[1]
                    if WAYPOINTS[waypoint][2] == "deposit":
                        target_theta = (angle_from_y_deg + 180) % 360
                    else: target_theta = angle_from_y_deg % 360
                    current_theta = pose[2]
                    delta_theta = (target_theta - pose[2] + 540) % 360 -180
                    if abs(current_x - target_x) > 0.5 or abs(current_y - target_y) > 0.5:
                            if initially_turning and abs(delta_theta) > 5:
                                turn_to(target_theta, pose)
                            elif initially_turning and abs(current_theta - target_theta) <= 5:
                                #stop_all()
                                #front_distance_to_obstacle = scan.detect_obstacles(247122073398, cameras=CAMERAS)
                                front_distance_to_obstacle = None
                                #print(front_distance_to_obstacle)
                                if front_distance_to_obstacle == None:
                                    print("No obstacles detected, moving to waypoint")
                                    initially_turning = False
                                else:
                                    waypoint_x = current_x + (front_distance_to_obstacle - 0.5) * np.sin(np.radians(current_theta))
                                    waypoint_y = current_y + (front_distance_to_obstacle - 0.5) * np.cos(np.radians(current_theta))
                                    WAYPOINTS.insert(waypoint, [waypoint_x, waypoint_y, "obstacle"])
                                    print(f"Obstacle detected, moving to waypoint {waypoint + 1} at {WAYPOINTS[waypoint]}")
                                    initially_turning = False
                            
                            elif initially_turning == False:
                                if j == 0:
                                    initial_pose_time = time.time()
                                    prev_position = pose
                                    j += 1
                                if time.time() - initial_pose_time > 1:
                                    if pose[0] == prev_position[0] and pose[1] == prev_position[1]:
                                        theta_rad = np.radians(pose[2])
                                        pose[0] = pose[0] + 0.25 * np.sin(theta_rad)#add x component
                                        pose[1] = pose[1] + 0.25 * np.cos(theta_rad)#add y component
                                    j = 0
                                move_to(pose, WAYPOINTS[waypoint])

                            current_time = time.time()
                            if current_time - last_print_time >= 1:
                                print(pose)
                                last_print_time = current_time
                                print("Moving to waypoint")
                    
                    else:
                        print(f"Arrived at waypoint {waypoint + 1} at position {pose}.")
                        stop_all()
                        at_waypoint = True  # Set flag to indicate arrival at waypoint
                        initially_turning = True
                
                else:
                    if WAYPOINTS[waypoint][2] == "mine":
                        if i == 0:
                            initial_excavation_time = time.time()
                            i += 1
                            
                        if excavate(initial_excavation_time):
                            at_waypoint = False
                            waypoint = 1
                            i = 0
                    
                    elif WAYPOINTS[waypoint][2] == "deposit":
                        if i == 0:
                            initial_deposit_time = time.time()
                            i += 1

                        if deposit(initial_deposit_time):
                            at_waypoint = False
                            waypoint = 0
                            i = 0
                    
                    elif WAYPOINTS[waypoint][2] == "obstacle":
                        
                        if i == 0:
                            turn_angle = current_theta + 90 # turn 90 degrees
                            i += 1

                        elif abs((turn_angle - pose[2] + 540) % 360 -180) >= 30 and i == 1: # turn to the right
                            turn_to(turn_angle, pose)
                        
                        elif abs(delta_theta) < 30 and i == 1:
                            i += 1

                        elif i == 2:
                            front_distance_to_obstacle = scan.detect_obstacles(247122073398, cameras=CAMERAS)
                            rear_distance_to_obstacle = scan.detect_obstacles(327122073351, cameras=CAMERAS)
                            i += 1

                        elif i == 3:
                            if front_distance_to_obstacle == None:
                                move_direction = 1 # we gonna move forward
                            
                            elif rear_distance_to_obstacle == None:
                                move_direction = -1 # we gonna move backward

                            elif rear_distance_to_obstacle > front_distance_to_obstacle and front_distance_to_obstacle > 0.5:
                                move_direction = 1 # we gonna move forward

                            elif front_distance_to_obstacle > rear_distance_to_obstacle and front_distance_to_obstacle > 0.5:
                                move_direction = -1 # we gonna move backward
                            
                            else:
                                print("Chat, we are cooked.")
                                break
                            
                            waypoint_x = current_x + move_direction * 0.5 * np.cos(np.radians(current_theta))
                            waypoint_y = current_y + move_direction * 0.5 * np.sin(np.radians(current_theta))
                            i += 1
                        
                        elif i == 4:
                            if move_direction == 1 and abs(current_x - waypoint_x) > 0.1 and abs(current_y - waypoint_y) > 0.1:
                                linear_motion(20)
                                print("Avoiding obstacle")
                            
                            elif move_direction == -1 and abs(current_x - waypoint_x) > 0.1 and abs(current_y - waypoint_y) > 0.1:
                                linear_motion(-20)
                                print("Avoiding obstacle")
                            
                            else:
                                print(f"Avoided obstacle, moving to waypoint {waypoint + 1} at {WAYPOINTS[waypoint]}.")
                                stop_all()
                                at_waypoint = False
                                i = 0
                                waypoint += 1
                                initially_turning = True

                            
        # cv2.destroyAllWindows()
        stop_all()
        print(f"Final Pose: {pose}")
except Exception as e:
    print(f"An error occurred: {e}")
    stop_all()
finally:
     stop_all()
