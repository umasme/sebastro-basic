import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import sys
import os
import gc  # Garbage collection
from threading import Lock
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global lock for Open3D visualization
visualization_lock = Lock()

def list_available_cameras():
    """List all available RealSense devices."""
    try:
        context = rs.context()
        devices = context.query_devices()
        serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
        logger.info(f"Found {len(serials)} RealSense cameras: {serials}")
        
        # Display additional information about each device
        for i, device in enumerate(devices):
            try:
                logger.debug(f"Device {i+1} info:")
                logger.debug(f"  Name: {device.get_info(rs.camera_info.name)}")
                logger.debug(f"  Serial: {device.get_info(rs.camera_info.serial_number)}")
                logger.debug(f"  Firmware: {device.get_info(rs.camera_info.firmware_version)}")
                
                # List supported stream profiles
                logger.debug("  Supported stream profiles:")
                sensors = device.query_sensors()
                for j, sensor in enumerate(sensors):
                    logger.debug(f"    Sensor {j+1} ({sensor.get_info(rs.camera_info.name)}):")
                    profiles = sensor.get_stream_profiles()
                    for profile in profiles:
                        if profile.is_video_profile():
                            video_profile = profile.as_video_stream_profile()
                            fmt = profile.format()
                            logger.debug(f"      {profile.stream_type()}: {video_profile.width()}x{video_profile.height()} @ {video_profile.fps()}fps (format: {fmt})")
            except Exception as e:
                logger.warning(f"Error getting detailed info for device {i+1}: {e}")
                
        return serials
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        return []

def initialize_camera(device_serial=None, visualize_debug=False):
    """
    Initialize a RealSense camera with simpler settings for low memory environments.
    """
    try:
        logger.info("Creating pipeline")
        pipeline = rs.pipeline()
        config = rs.config()

        # List available cameras to debug
        available_cameras = list_available_cameras()
        if not available_cameras:
            logger.error("No RealSense cameras detected!")
            return None

        # Enable the device with lowest possible settings to ensure compatibility
        if device_serial:
            logger.info(f"Enabling device with serial: {device_serial}")
            config.enable_device(device_serial)
        
        # Very low-resolution depth stream for minimal memory usage
        # Try different format options if one doesn't work
        try_formats = [(rs.format.z16, 30), (rs.format.z16, 30)]
        success = False
        
        for depth_format, fps in try_formats:
            try:
                logger.debug(f"Trying format {depth_format} @ {fps}fps")
                # Reset config for each attempt
                config = rs.config()
                if device_serial:
                    config.enable_device(device_serial)
                
                # Configure with minimal resolution
                logger.info(f"Setting up depth stream with resolution 320x240 @ {fps}fps")
                config.enable_stream(rs.stream.depth, 480, 270, depth_format, fps)
                
                # Try to start the pipeline
                logger.info("Starting pipeline")
                pipeline_profile = pipeline.start(config)
                
                # If we get here, it worked
                success = True
                logger.info("Pipeline started successfully")
                
                # Get device information
                device = pipeline_profile.get_device()
                if not device_serial:
                    device_serial = device.get_info(rs.camera_info.serial_number)
                
                # Debug device info
                logger.debug(f"Using device: {device.get_info(rs.camera_info.name)}")
                logger.debug(f"Serial: {device_serial}")
                logger.debug(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")
                
                # Log active streams
                active_profiles = pipeline_profile.get_streams()
                logger.debug("Active streams:")
                for profile in active_profiles:
                    if profile.is_video_profile():
                        video_profile = profile.as_video_stream_profile()
                        logger.debug(f"  {profile.stream_type()}: {video_profile.width()}x{video_profile.height()} @ {video_profile.fps()}fps (format: {profile.format()})")
                
                break
            except Exception as e:
                logger.warning(f"Failed with settings {depth_format} @ {fps}fps: {e}")
                try:
                    pipeline.stop()
                except:
                    pass
        
        if not success:
            logger.error("Could not initialize camera with any settings")
            return None
            
        # Create a decimation filter for downsampling
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 4)  # More aggressive decimation
        
        # Simple spatial filter with minimal settings
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 1)
        
        # Return camera object
        camera = {
            "pipeline": pipeline,
            "config": config,
            "device_serial": device_serial,
            "is_running": True,
            "filters": {
                "decimation": decimation,
                "spatial": spatial
            }
        }
        
        # Debug first frame if requested
        if visualize_debug:
            try:
                logger.debug("Testing frame acquisition...")
                frames = pipeline.wait_for_frames(5000)  # 5s timeout
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    logger.debug(f"Got frame: {depth_frame.get_width()}x{depth_frame.get_height()}")
                    depth_data = np.asanyarray(depth_frame.get_data())
                    logger.debug(f"Depth range: min={np.min(depth_data)}, max={np.max(depth_data)}")
                else:
                    logger.warning("No depth frame received in test")
            except Exception as e:
                logger.warning(f"Frame test error: {e}")

        return camera
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        raise

def stop_camera(camera):
    """Stop the camera pipeline."""
    if camera is None:
        return
        
    try:
        if camera.get("is_running", False):
            logger.info("Stopping camera pipeline")
            camera["pipeline"].stop()
            camera["is_running"] = False
    except Exception as e:
        logger.error(f"Error stopping camera: {e}")

def get_frames(camera, timeout_ms=5000):
    """Get filtered depth frames from the camera."""
    if camera is None or not camera.get("is_running", False):
        raise RuntimeError("Camera pipeline is not running")

    try:
        logger.info("Waiting for frames")
        frames = camera["pipeline"].wait_for_frames(timeout_ms)
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            logger.warning("No depth frame received")
            return None

        logger.info(f"Depth frame acquired: {depth_frame.get_width()}x{depth_frame.get_height()}")
        
        # Apply filters to reduce data
        filtered_frame = camera["filters"]["decimation"].process(depth_frame)
        filtered_frame = camera["filters"]["spatial"].process(filtered_frame)
        
        logger.debug(f"Frame after filtering: {filtered_frame.get_width()}x{filtered_frame.get_height()}")
        
        return filtered_frame
    except Exception as e:
        logger.error(f"Error getting frames: {e}")
        raise

def create_simple_point_cloud(camera, max_points=10000):
    """Create a very simple point cloud with minimal memory usage."""
    try:
        # Get a frame
        depth_frame = get_frames(camera, timeout_ms=5000)
        if not depth_frame:
            logger.error("Failed to get valid depth frame")
            return None
            
        # Get depth intrinsics for point projection
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        height, width = depth_image.shape
        
        # Subsample the depth image to reduce points
        sample_step = max(1, int(np.sqrt(width * height / max_points)))
        logger.debug(f"Using sample step of {sample_step} to limit points")
        
        # Create arrays for storing points
        points = []
        
        # Sample points from the depth image
        for y in range(0, height, sample_step):
            for x in range(0, width, sample_step):
                depth_value = depth_image[y, x]
                
                # Skip invalid depth values
                if depth_value <= 0 or depth_value > 10000:  # 10m max range
                    continue
                    
                # Convert depth to 3D point
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth_value)
                points.append(depth_point)
                
                # Break early if we have enough points
                if len(points) >= max_points:
                    break
            
            # Break early if we have enough points
            if len(points) >= max_points:
                break
                
        # Create Open3D point cloud if we have points
        if not points:
            logger.warning("No valid points found")
            return None
            
        logger.info(f"Created point cloud with {len(points)} points")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Assign uniform color
        colors = np.ones((len(points), 3)) * 0.5  # Gray
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
        
    except Exception as e:
        logger.error(f"Error creating simple point cloud: {e}")
        return None
    finally:
        # Force garbage collection
        gc.collect()

def find_ground_plane(pcd):
    """Find the ground plane in a point cloud."""
    if pcd is None or len(np.asarray(pcd.points)) == 0:
        logger.warning("Empty point cloud, cannot find ground plane")
        return None
        
    try:
        # Perform plane segmentation to find the ground plane
        logger.info("Performing plane segmentation")
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=100  # Reduced iterations for speed
        )
        
        [a, b, c, d] = plane_model
        logger.info(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        logger.info(f"Found {len(inliers)} points on the ground plane")
        
        return plane_model
        
    except Exception as e:
        logger.error(f"Error finding ground plane: {e}")
        return None

def detect_simple_obstacles(pcd, plane_model, max_distance=3.0):
    """Detect obstacles as points above the ground plane."""
    if pcd is None or plane_model is None:
        return None
        
    try:
        # Extract plane equation coefficients
        a, b, c, d = plane_model
        normal_vector = np.array([a, b, c])
        
        # Get points
        points = np.asarray(pcd.points)
        
        # Check which points are above the plane and not too high
        # Calculate distance to plane
        distances = (a * points[:, 0] + b * points[:, 1] + 
                    c * points[:, 2] + d) / np.linalg.norm(normal_vector)
        
        # Find points that are obstacles (above ground but not too high)
        min_height = 0.05  # 5cm above ground
        max_height = 0.5   # 50cm max height
        obstacle_mask = (distances > min_height) & (distances < max_height)
        
        # Calculate distance from origin (camera)
        point_distances = np.linalg.norm(points, axis=1)
        
        # Only consider points within max_distance
        valid_distance_mask = point_distances < max_distance
        
        # Combine masks
        final_mask = obstacle_mask & valid_distance_mask
        
        # If we have obstacles, find the closest one
        if np.any(final_mask):
            obstacle_points = points[final_mask]
            obstacle_distances = point_distances[final_mask]
            
            min_idx = np.argmin(obstacle_distances)
            min_distance = obstacle_distances[min_idx]
            closest_point = obstacle_points[min_idx]
            
            logger.info(f"Found obstacle at distance: {min_distance:.2f}m")
            logger.info(f"Obstacle point: [{closest_point[0]:.2f}, {closest_point[1]:.2f}, {closest_point[2]:.2f}]")
            
            return min_distance
        else:
            logger.info("No obstacles detected")
            return None
            
    except Exception as e:
        logger.error(f"Error detecting obstacles: {e}")
        return None

def simple_obstacle_detection(camera_serial=None):
    """Simplified obstacle detection for low-memory systems."""
    camera = None
    
    try:
        # List available cameras
        available_cameras = list_available_cameras()
        if not available_cameras:
            logger.error("No RealSense cameras found!")
            return None
            
        # Use specified camera or first available
        if camera_serial is None:
            camera_serial = available_cameras[0]
            logger.info(f"Using first available camera: {camera_serial}")
        
        # Initialize camera with debug info
        camera = initialize_camera(device_serial=camera_serial, visualize_debug=True)
        if camera is None:
            logger.error("Failed to initialize camera")
            return None
            
        # Create minimal point cloud
        logger.info("Creating simple point cloud")
        pcd = create_simple_point_cloud(camera, max_points=5000)
        if pcd is None:
            logger.error("Failed to create point cloud")
            return None
            
        # Find ground plane
        logger.info("Finding ground plane")
        plane_model = find_ground_plane(pcd)
        if plane_model is None:
            logger.error("Failed to find ground plane")
            return None
            
        # Detect obstacles
        logger.info("Detecting obstacles")
        obstacle_distance = detect_simple_obstacles(pcd, plane_model)
        
        return obstacle_distance
        
    except Exception as e:
        logger.error(f"Error in simple obstacle detection: {e}")
        return None
    finally:
        # Stop camera
        if camera:
            stop_camera(camera)
            
        # Force garbage collection
        gc.collect()

def main():
    # Check command line arguments
    device_serial = None
    debug_mode = False
    
    for arg in sys.argv[1:]:
        if arg == "--debug":
            debug_mode = True
            logging.getLogger().setLevel(logging.DEBUG)
        elif not arg.startswith("-"):
            device_serial = arg
    
    logger.info("Starting simple obstacle detection")
    
    try:
        # Force garbage collection before starting
        gc.collect()
        
        # Call the simplified obstacle detection
        obstacle_distance = simple_obstacle_detection(camera_serial=device_serial)
        
        if obstacle_distance is not None:
            print(f"Obstacle detected at: {obstacle_distance:.2f} meters")
        else:
            print("No obstacles detected or detection failed")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Final garbage collection
        gc.collect()

if __name__ == "__main__":
    main()
