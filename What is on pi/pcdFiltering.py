from pcdGenNoColorD435i import check_obstacles_in_front
import open3d as o3d
import numpy as np

def visualize_and_filter_point_cloud(pcd_file, camera_position, max_height_inches=40, band_width_inches=2, cylinder_radius=3.0, brown_radius_inches=10):
    # Load the point cloud from file
    pcd = o3d.io.read_point_cloud(pcd_file)

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
    # Brown: Points within 3 inches of the origin (inside the cylinder)
    inside_points = np.asarray(inside_non_ground_cloud.points)
    distances_inside = (a * inside_points[:, 0] + b * inside_points[:, 1] + c * inside_points[:, 2] + d)
    horizontal_distances_inside = horizontal_distances_non_ground[inside_indices_non_ground]

    brown_indices = np.where(horizontal_distances_inside <= brown_radius_meters)[0]
    brown_cloud = inside_non_ground_cloud.select_by_index(brown_indices)

    # Remove brown points from the "yellow", "red", and "blue" groups
    remaining_indices = np.setdiff1d(np.arange(len(inside_points)), brown_indices)
    inside_non_ground_cloud = inside_non_ground_cloud.select_by_index(remaining_indices)

    # Yellow: Points within 1 inch of the ground plane
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


    # Visualize all the groups
    o3d.visualization.draw_geometries([brown_cloud, yellow_cloud, red_cloud, blue_cloud, green_cloud],
                                      window_name="Point Cloud Visualization",
                                      width=800,
                                      height=600,
                                      left=50,
                                      top=50)

    # Save the filtered point cloud
    combined_cloud = brown_cloud + yellow_cloud + red_cloud + blue_cloud + green_cloud + pink_cloud
    obs_cloud = red_cloud + blue_cloud
    output_file = "filtered_point_cloud_with_exclusive_colors.pcd"
    o3d.io.write_point_cloud(output_file, combined_cloud)
    print(f"Filtered point cloud saved to {output_file}")
    

    # Combine non-ground clouds for 2D map
    non_ground_cloud = red_cloud + blue_cloud + yellow_cloud + brown_cloud
    gCloud = green_cloud + yellow_cloud + brown_cloud
    max_threshold = cylinder_radius  # Horizontal cutoff radius
    # Convert plane model to standard float types
    #plane_equation = [float(a), float(b), float(c), float(d)]
    plane_equation = [a, b, c, d]
    # Generate and visualize the 2D matrix map with proper scaling
    result = check_obstacles_in_front(obs_cloud, plane_equation)
    print(f"Obstacles detected in front: {result}")



# Provide your PCD file path and the camera's position (e.g., [x, y, z] in meters)
pcd_file_path = "point_cloud_2.pcd"
camera_position = [0.0, 0.0, 0.0]  # Replace with the actual camera position
visualize_and_filter_point_cloud(pcd_file_path, camera_position)

