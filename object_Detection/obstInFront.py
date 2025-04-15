import numpy as np
import open3d as o3d

def check_obstacles_in_front(point_cloud_file, plane_equation, rectangle_width=2, rectangle_length=3, visualize=True):
    """
    Check if any points fall within a rectangle in front of the camera (origin)
    after projecting onto the ground plane, with optional visualization.
    
    Args:
        point_cloud_file: Path to the PCD file
        plane_equation: Coefficients [a, b, c, d] of the ground plane (ax + by + cz + d = 0)
        rectangle_width: Width of the rectangle (perpendicular to camera forward direction)
        rectangle_length: Length of the rectangle (along camera forward direction)
        visualize: Whether to show visualization
    
    Returns:
        bool: True if any points are detected within the specified rectangle
    """
    # Load the point cloud
    points = np.asarray(point_cloud_file.points)
    
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
    
    has_obstacles = np.any(in_rectangle)
    
    # Visualization
    if visualize:
        # Create a copy of the point cloud for visualization
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color the points based on whether they're in the rectangle
        colors = np.zeros((len(points), 3))
        colors[in_rectangle] = [1, 0, 0]  # Red for points in rectangle
        colors[~in_rectangle] = [0.5, 0.5, 0.5]  # Gray for other points
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
    
    return has_obstacles
'''
# Example usage
if __name__ == "__main__":
    # Example parameters
    pcd_file = "your_point_cloud.pcd"
    ground_plane = [0.0248, 0.9593, 0.2814, -0.5106]  # From your example
    
    result = check_obstacles_in_front(pcd_file, ground_plane, rectangle_width=2, rectangle_length=1, visualize=True)
    print(f"Obstacles detected in front: {result}")
'''
