import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import sys




def list_available_cameras():
    """
    List all available RealSense devices.
    
    Returns:
        list: List of device serial numbers.
    """
    context = rs.context()
    devices = context.query_devices()
    return [device.get_info(rs.camera_info.serial_number) for device in devices]

def initialize_camera(device_serial=None):
    """
    Initialize a RealSense camera.
    
    Args:
        device_serial (str, optional): Serial number of the specific camera to use.
                                       If None, the first available camera will be used.
    
    Returns:
        dict: A dictionary containing the pipeline, config, and device_serial
    """
    pipeline = rs.pipeline()
    config = rs.config()
    
    # If a specific device serial is provided, use it
    if device_serial:
        config.enable_device(device_serial)
        
    # Configure only depth stream - removed color stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Start the pipeline
    profile = pipeline.start(config)
    
    # If no serial was provided, get it from the active device
    if not device_serial:
        device = profile.get_device()
        device_serial = device.get_info(rs.camera_info.serial_number)
    
    # Return all necessary components as a "camera" object
    camera = {
        "pipeline": pipeline,
        "config": config,
        "device_serial": device_serial,
        "is_running": True
    }
    
    return camera

def stop_camera(camera):
    """
    Stop the camera pipeline.
    
    Args:
        camera (dict): Camera object returned by initialize_camera()
    """
    if camera["is_running"]:
        camera["pipeline"].stop()
        camera["is_running"] = False

def get_frames(camera):
    """
    Get depth frames from the camera.
    
    Args:
        camera (dict): Camera object returned by initialize_camera()
        
    Returns:
        depth_frame: Depth frame
    """
    if not camera["is_running"]:
        raise RuntimeError("Camera pipeline is not running")
        
    frames = camera["pipeline"].wait_for_frames()
    depth_frame = frames.get_depth_frame()
    
    return depth_frame

def create_point_cloud(camera, depth_frame=None, visualize=False):
    if depth_frame is None:
        depth_frame = get_frames(camera)

    if not depth_frame:
        raise RuntimeError("Failed to capture valid frames")

    # Convert depth frame to Open3D format
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_o3d = o3d.geometry.Image(depth_image)

    # Get RealSense intrinsics
    profile = camera["pipeline"].get_active_profile()
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height,
        intrinsics.fx, intrinsics.fy,
        intrinsics.ppx, intrinsics.ppy
    )

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d,
        camera_intrinsics,
        depth_scale=1000.0,      # For RealSense: millimeters to meters
        depth_trunc=3.0,         # Truncate at 3 meters
        stride=1
    )

    # Flip the orientation if necessary
    '''pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    '''
    if visualize:
        o3d.visualization.draw_geometries([pcd])

    print(f"Generated point cloud with {len(pcd.points)} points")
    return pcd

def process_camera_by_serial(device_serial=None, visualize=False):
    """
    Process a specific camera identified by its serial number.
    
    Args:
        device_serial (str, optional): Serial number of the camera to use.
                                       If None, the first available camera will be used.
        visualize (bool): Whether to visualize the point cloud.
        
    Returns:
        o3d.geometry.PointCloud or None: The created point cloud object, or None if failed.
    """
    try:
        # If no serial provided, try to get the first available camera
        if device_serial is None:
            available_cameras = list_available_cameras()
            if not available_cameras:
                print("No RealSense cameras found!")
                return None
            device_serial = available_cameras[0]
            print(f"No serial specified, using the first available camera: {device_serial}")
            
        # Initialize camera with the specified serial
        print(f"Initializing camera with serial: {device_serial}")
        camera = initialize_camera(device_serial)
        
        # Create point cloud (no save path)
        pcd = create_point_cloud(camera, visualize=visualize)
        
        # Stop the camera when done
        stop_camera(camera)
        
        return pcd
    
    except Exception as e:
        print(f"Error processing camera {device_serial}: {e}")
        return None

def check_obstacles_in_front(pcd, plane_equation, rectangle_width=2, rectangle_length=3, visualize=True):
    """
    Find the closest point within a rectangle in front of the camera (origin)
    after projecting onto the ground plane, with optional visualization.
    
    Args:
        pcd: Point cloud object
        plane_equation: Coefficients [a, b, c, d] of the ground plane (ax + by + cz + d = 0)
        rectangle_width: Width of the rectangle (perpendicular to camera forward direction)
        rectangle_length: Length of the rectangle (along camera forward direction)
        visualize: Whether to show visualization
    
    Returns:
        float: Distance to closest obstacle, or None if no obstacles found
    """
    # Debug visualization flag
    print(f"check_obstacles_in_front called with visualize={visualize}")
    # Get points from point cloud
    points = np.asarray(pcd.points)
    
    # Extract plane equation coefficients
    a, b, c, _ = plane_equation  # Only need normal direction, not 'd'
    normal_vector = np.array([a, b, c])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize
    
    # Define camera's forward direction (assuming z-axis is forward)
    camera_forward = np.array([0, 0, 1])
    
    # Create a coordinate system on the ground plane
    # First basis vector tries to align with camera forward projected onto plane
    forward_projected = camera_forward - np.dot(camera_forward, normal_vector) * normal_vector
    if np.linalg.norm(forward_projected) < 1e-6:  # If forward is perpendicular to plane
        x_axis = np.array([1, 0, 0])
        if abs(np.dot(x_axis, normal_vector)) > 0.9:  # If normal is close to x-axis
            x_axis = np.array([0, 1, 0])
        forward_projected = x_axis - np.dot(x_axis, normal_vector) * normal_vector
        
    x_parallel = forward_projected / np.linalg.norm(forward_projected)  # First basis vector
    y_parallel = np.cross(normal_vector, x_parallel)  # Second basis vector
    y_parallel = y_parallel / np.linalg.norm(y_parallel)  # Normalize
    
    # Create projection matrix
    projection_matrix = np.column_stack([x_parallel, y_parallel])
    
    # Project all points (only for classification)
    points_2d = np.dot(points, projection_matrix)
    
    # Check if any points fall within the rectangle
    in_rectangle = np.logical_and(
        np.logical_and(points_2d[:, 0] >= 0, points_2d[:, 0] <= rectangle_length),
        np.logical_and(points_2d[:, 1] >= -rectangle_width/2, points_2d[:, 1] <= rectangle_width/2)
    )
    
    # Initialize closest point variables
    closest_point = None
    min_distance = float('inf')
    
    # Find closest point in rectangle
    if np.any(in_rectangle):
        # Get points that are in the rectangle
        rectangle_points = points[in_rectangle]
        
        # Calculate distances from origin to each point
        distances = np.linalg.norm(rectangle_points, axis=1)
        
        # Find the closest point
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        closest_point = rectangle_points[min_idx]
    
    # Visualization
    if visualize:
        # Create a copy of the point cloud for visualization
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color the points based on whether they're in the rectangle
        colors = np.zeros((len(points), 3))
        colors[in_rectangle] = [1, 0, 0]  # Red for points in rectangle
        colors[~in_rectangle] = [0.5, 0.5, 0.5]  # Gray for other points
        
        # Highlight the closest point if found
        if closest_point is not None:
            # Find the index of closest point in the original points array
            closest_idx = np.where(np.all(points == closest_point, axis=1))[0][0]
            colors[closest_idx] = [0, 0, 1]  # Blue for closest point
            
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Create rectangle corners in the 2D coordinate system
        rect_corners_2d = np.array([
            [0, -rectangle_width/2],                # Bottom left
            [rectangle_length, -rectangle_width/2], # Bottom right
            [rectangle_length, rectangle_width/2],  # Top right
            [0, rectangle_width/2],                 # Top left
            [0, -rectangle_width/2]                 # Close the loop
        ])
        
        # Convert rectangle corners back to 3D
        # We need to find a point on the plane as reference
        # Using the normal and a point that satisfies ax + by + cz + d = 0
        t = -plane_equation[3] / (a**2 + b**2 + c**2)**0.5  # Distance from origin to plane
        point_on_plane = t * normal_vector  # A point on the plane closest to origin
        
        # Convert 2D rectangle points to 3D
        rect_corners_3d = np.zeros((len(rect_corners_2d), 3))
        for i, corner in enumerate(rect_corners_2d):
            # Convert from 2D plane coordinates to 3D
            rect_corners_3d[i] = point_on_plane + corner[0] * x_parallel + corner[1] * y_parallel
        
        # Create line set for rectangle
        lines = [[i, i+1] for i in range(4)]  # Connect consecutive points
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(rect_corners_3d)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(lines))])  # Green lines
        
        # Create coordinate system for visualization
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        
        # Visualize
        o3d.visualization.draw_geometries([vis_pcd, line_set, coordinate_frame])
    
    if np.any(in_rectangle):
        print(f"Found {np.sum(in_rectangle)} points in the rectangle")
        print(f"Closest point: {closest_point}, distance: {min_distance:.4f}")
        return min_distance
    else:
        print("No obstacles detected in the rectangle")
        return None

def visualize_and_filter_point_cloud(pcd, camera_position, max_height_inches=40, band_width_inches=2, cylinder_radius=5, brown_radius_inches=10, visualize=False):
    """
    Filter and visualize a point cloud with color-coded regions.
    
    Args:
        pcd: Point cloud object
        camera_position: [x, y, z] position of the camera
        max_height_inches: Maximum height threshold in inches
        band_width_inches: Width of the band around the ground plane in inches
        cylinder_radius: Radius of the cylinder for horizontal filtering
        brown_radius_inches: Radius for the brown center region in inches
        visualize: Whether to show visualization
        
    Returns:
        float or tuple: Distance to the closest obstacle, or (None, None) if no obstacles detected
    """
    # Debug visualization flag
    print(f"visualize_and_filter_point_cloud called with visualize={visualize}")
    # Transform the point cloud so that the camera's position becomes the origin
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = -np.asarray(camera_position)  # Translate by negative camera position
    pcd.transform(translation_matrix)

    # Downsample the point cloud for efficiency
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # Perform plane segmentation using RANSAC to find the ground plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

    # Convert max height and band width from inches to meters
    max_height_meters = max_height_inches * 0.0254
    band_width_meters = band_width_inches * 0.0254
    brown_radius_meters = brown_radius_inches * 0.0254

    # Separate ground plane points (inliers) and others (outliers)
    ground_plane_cloud = pcd.select_by_index(inliers)
    non_ground_cloud = pcd.select_by_index(inliers, invert=True)

    # Step 1: Filter out points above the height threshold from both clouds
    all_points = np.asarray(non_ground_cloud.points)
    ground_plane_points = np.asarray(ground_plane_cloud.points)

    distances_to_plane = (a * all_points[:, 0] + b * all_points[:, 1] + c * all_points[:, 2] + d)
    valid_height_indices = np.where(distances_to_plane <= max_height_meters)[0]
    non_ground_cloud = non_ground_cloud.select_by_index(valid_height_indices)

    distances_ground = (a * ground_plane_points[:, 0] + b * ground_plane_points[:, 1] + c * ground_plane_points[:, 2] + d)
    valid_ground_indices = np.where(distances_ground <= max_height_meters)[0]
    ground_plane_cloud = ground_plane_cloud.select_by_index(valid_ground_indices)

    # Step 2: Compute horizontal distances for both clouds
    non_ground_points = np.asarray(non_ground_cloud.points)
    ground_plane_points = np.asarray(ground_plane_cloud.points)

    projection_matrix = np.eye(3) - np.outer([a, b, c], [a, b, c]) / np.dot([a, b, c], [a, b, c])
    horizontal_positions_non_ground = non_ground_points @ projection_matrix.T
    horizontal_positions_ground = ground_plane_points @ projection_matrix.T

    horizontal_distances_non_ground = np.linalg.norm(horizontal_positions_non_ground, axis=1)
    horizontal_distances_ground = np.linalg.norm(horizontal_positions_ground, axis=1)

    # Step 3: Split points into inside and outside horizontal cutoff for both clouds
    inside_indices_non_ground = np.where(horizontal_distances_non_ground <= cylinder_radius)[0]
    outside_indices_non_ground = np.where(horizontal_distances_non_ground > cylinder_radius)[0]

    inside_indices_ground = np.where(horizontal_distances_ground <= cylinder_radius)[0]
    outside_indices_ground = np.where(horizontal_distances_ground > cylinder_radius)[0]

    inside_non_ground_cloud = non_ground_cloud.select_by_index(inside_indices_non_ground)
    outside_non_ground_cloud = non_ground_cloud.select_by_index(outside_indices_non_ground)

    inside_ground_cloud = ground_plane_cloud.select_by_index(inside_indices_ground)
    outside_ground_cloud = ground_plane_cloud.select_by_index(outside_indices_ground)

    # Step 4: Apply exclusive colors
    # Brown: Points within the brown radius (inside the cylinder)
    inside_points = np.asarray(inside_non_ground_cloud.points)
    distances_inside = (a * inside_points[:, 0] + b * inside_points[:, 1] + c * inside_points[:, 2] + d)
    horizontal_distances_inside = horizontal_distances_non_ground[inside_indices_non_ground]

    brown_indices = np.where(horizontal_distances_inside <= brown_radius_meters)[0]
    brown_cloud = inside_non_ground_cloud.select_by_index(brown_indices)

    # Remove brown points from the "yellow", "red", and "blue" groups
    remaining_indices = np.setdiff1d(np.arange(len(inside_points)), brown_indices)
    inside_non_ground_cloud = inside_non_ground_cloud.select_by_index(remaining_indices)

    # Yellow: Points within band_width of the ground plane
    distances_inside_remaining = distances_inside[remaining_indices]
    yellow_indices = np.where(np.abs(distances_inside_remaining) <= band_width_meters)[0]
    yellow_cloud = inside_non_ground_cloud.select_by_index(yellow_indices)

    # Red: Points above the yellow band
    red_indices = np.where(distances_inside_remaining > band_width_meters)[0]
    red_cloud = inside_non_ground_cloud.select_by_index(red_indices)

    # Blue: Points below the yellow band
    blue_indices = np.where(distances_inside_remaining < -band_width_meters)[0]
    blue_cloud = inside_non_ground_cloud.select_by_index(blue_indices)

    # Green: Ground plane points inside the cylinder
    green_cloud = inside_ground_cloud
    # Pink: All points outside the horizontal cutoff (non-ground + ground outside)
    pink_cloud = outside_non_ground_cloud + outside_ground_cloud

    # Color the groups
    brown_cloud.paint_uniform_color([0.6, 0.3, 0])  # Brown
    yellow_cloud.paint_uniform_color([1, 1, 0])  # Yellow
    red_cloud.paint_uniform_color([1, 0, 0])  # Red
    blue_cloud.paint_uniform_color([0, 0, 1])  # Blue
    green_cloud.paint_uniform_color([0, 1, 0])  # Green
    pink_cloud.paint_uniform_color([1, 0.75, 0.8])  # Pink

    print(f"Yellow indices count: {len(yellow_indices)}")

    # Visualize all the groups if requested
    if visualize:
        o3d.visualization.draw_geometries([brown_cloud, yellow_cloud, red_cloud, blue_cloud, green_cloud],
                                        window_name="Point Cloud Visualization",
                                        width=800,
                                        height=600,
                                        left=50,
                                        top=50)

    # Combine non-ground clouds for 2D map
    obs_cloud = red_cloud + blue_cloud
    
    # Check for obstacles in front, using the same visualization setting
    return check_obstacles_in_front(
        obs_cloud, 
        plane_model, 
        visualize=visualize
    )

def detect_obstacles(camera_serial=None, camera_position=[0.0, 0.0, 0.0], visualize=True):
    """
    Complete obstacle detection pipeline that generates point cloud from camera
    and detects obstacles.
    
    Args:
        camera_serial (str, optional): Serial number of camera to use. If None, uses first available.
        camera_position (list): 3D position of camera [x, y, z]
        visualize (bool): Whether to visualize the results (controls ALL visualizations)
        
    Returns:
        float or None: Distance to closest obstacle, or None if no obstacles detected/camera not available
    """
    # Print visualization status to debug
    print(f"Visualization is set to: {visualize}")
    
    # Check if camera serial is provided, otherwise list available cameras
    if camera_serial is None:
        available_cameras = list_available_cameras()
        if not available_cameras:
            print("No RealSense cameras found!")
            return None
        camera_serial = available_cameras[0]
        print(f"Using first available camera: {camera_serial}")
    
    # Process camera to get point cloud
    pcd = process_camera_by_serial(
        device_serial=camera_serial,
        visualize=visualize  # Pass the visualization flag
    )
    
    if pcd:
        # Detect obstacles using the point cloud
        obstacle_range = visualize_and_filter_point_cloud(
            pcd, 
            camera_position,
            max_height_inches=40, 
            band_width_inches=2, 
            cylinder_radius=5, 
            brown_radius_inches=10,
            visualize=visualize  # Pass the visualization flag
        )
        return obstacle_range
    else:
        print("Failed to generate point cloud")
        return None

def main():
    # Check for visualization flag in arguments
    visualize = False
    device_serial = None
    
    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg == "--visualize" or arg == "-v":
            visualize = True
        elif not arg.startswith("-"):
            device_serial = arg
    
    print(f"Starting obstacle detection with visualization={visualize}")
    
    # Call the detect_obstacles function with command line arguments
    obstacle_range = detect_obstacles(
        camera_serial=device_serial, 
        visualize=visualize
    )
    print(f"Obstacle range: {obstacle_range}")

if __name__ == "__main__":
    main()
    




''' EXAMPLE CALLING SCRIPT


    # Import the module
import unified_realsense_obstacle_detection as obstacle_detector

def my_function():
    # Call the detect_obstacles function with your camera serial
    camera_serial = "327122073351"  # Replace with your actual camera serial
    result = obstacle_detector.detect_obstacles(
        camera_serial=camera_serial,
        camera_position=[0.0, 0.0, 0.0],
        visualize=True  # Set to True for visualizations, False to disable them
    )
    
    return result

# Example usage
obstacle_distance = my_function()
if obstacle_distance is not None:
    print(f"Detected obstacle at distance: {obstacle_distance} meters")
else:
    print("No obstacles detected or camera error")
    
    
'''
