# visualization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_results(image, world_points, image_points, projected_points, court):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the original image with points
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.scatter(image_points[:, 0], image_points[:, 1], c="g", label="Selected")
    ax1.scatter(
        projected_points[:, 0, 0], projected_points[:, 0, 1], c="r", label="Projected"
    )
    for i, (img_pt, proj_pt) in enumerate(zip(image_points, projected_points)):
        ax1.plot([img_pt[0], proj_pt[0, 0]], [img_pt[1], proj_pt[0, 1]], "b-")
        ax1.text(img_pt[0], img_pt[1], str(i + 1), fontsize=8, color="white")
    ax1.legend()
    ax1.set_title("Image with Calibration Points")

    # Plot the court schematic with projected points
    court_width = court.court_definition.width
    court_height = court.court_definition.height
    ax2.set_xlim(0, court_width)
    ax2.set_ylim(0, court_height)
    ax2.set_aspect("equal")

    # Draw court outline
    ax2.add_patch(plt.Rectangle((0, 0), court_width, court_height, fill=False))

    # Draw center circle
    center_x, center_y = court_width / 2, court_height / 2
    circle_radius = court.court_definition.circle_diameter / 2
    ax2.add_patch(plt.Circle((center_x, center_y), circle_radius, fill=False))

    # Draw 3-point line
    three_point_radius = court.court_definition.three_point_distance
    ax2.add_patch(plt.Circle((0, center_y), three_point_radius, fill=False))
    ax2.add_patch(plt.Circle((court_width, center_y), three_point_radius, fill=False))

    # Plot world points
    ax2.scatter(world_points[:, 0], world_points[:, 1], c="b", label="World Points")
    for i, point in enumerate(world_points):
        ax2.text(point[0], point[1], str(i + 1), fontsize=8)

    ax2.legend()
    ax2.set_title("Court Schematic with World Points")

    plt.tight_layout()
    plt.show()


def visualize_court_overlay(image, camera_matrix, dist_coeffs, rvec, tvec, court):
    court_def = court.court_definition
    half_width = court_def.width / 2
    half_height = court_def.height / 2

    court_points = np.array(
        [
            [-half_width, -half_height, 0],
            [half_width, -half_height, 0],
            [half_width, half_height, 0],
            [-half_width, half_height, 0],
            [0, 0, 0],
            [court_def.circle_diameter / 2, 0, 0],
            [-half_width + court_def.key_area_length, -court_def.key_area_width / 2, 0],
            [-half_width + court_def.key_area_length, court_def.key_area_width / 2, 0],
            [half_width - court_def.key_area_length, -court_def.key_area_width / 2, 0],
            [half_width - court_def.key_area_length, court_def.key_area_width / 2, 0],
            *[
                [
                    -half_width
                    + court_def.rim_center_offset
                    + court_def.three_point_distance * np.cos(angle),
                    court_def.three_point_distance * np.sin(angle),
                    0,
                ]
                for angle in np.linspace(-np.pi / 2, np.pi / 2, 20)
            ],
            *[
                [
                    half_width
                    - court_def.rim_center_offset
                    - court_def.three_point_distance * np.cos(angle),
                    court_def.three_point_distance * np.sin(angle),
                    0,
                ]
                for angle in np.linspace(-np.pi / 2, np.pi / 2, 20)
            ],
            [-half_width, court_def.three_point_limit - half_height, 0],
            [-half_width, half_height - court_def.three_point_limit, 0],
            [half_width, court_def.three_point_limit - half_height, 0],
            [half_width, half_height - court_def.three_point_limit, 0],
        ],
        dtype=np.float32,
    )

    # Debug: Print court points
    print("Court Points:")
    print(court_points)

    projected_points, _ = cv2.projectPoints(
        court_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)

    # Debug: Print projected points
    print("Projected Points:")
    print(projected_points)

    overlay_image = image.copy()

    def valid_point(point):
        return (
            isinstance(point, (tuple, list, np.ndarray))
            and len(point) == 2
            and all(
                isinstance(coord, (int, float))
                and not np.isnan(coord)
                and not np.isinf(coord)
                for coord in point
            )
        )

    def to_int_tuple(point):
        return tuple(map(int, point))

    def safe_draw_line(img, pt1, pt2, color, thickness):
        if valid_point(pt1) and valid_point(pt2):
            cv2.line(img, to_int_tuple(pt1), to_int_tuple(pt2), color, thickness)

    def safe_draw_circle(img, center, radius, color, thickness):
        if valid_point(center) and isinstance(radius, (int, float)) and radius > 0:
            cv2.circle(img, to_int_tuple(center), int(radius), color, thickness)

    # Debug: Draw all projected points and print coordinates
    for i, point in enumerate(projected_points):
        if valid_point(point):
            print(f"Point {i}: {point}")
            safe_draw_circle(overlay_image, point, 5, (255, 0, 0), -1)

    # Save intermediate results for verification
    cv2.imwrite("data/overlay_image_debug.png", overlay_image)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    plt.title("Court Overlay on Image")
    plt.axis("off")
    plt.show()

    # Draw court outline
    for i in range(4):
        safe_draw_line(
            overlay_image,
            projected_points[i],
            projected_points[(i + 1) % 4],
            (0, 255, 0),
            2,
        )

    # Draw center circle
    if valid_point(projected_points[4]) and valid_point(projected_points[5]):
        center = to_int_tuple(projected_points[4])
        radius_point = to_int_tuple(projected_points[5])
        radius = int(np.linalg.norm(np.array(radius_point) - np.array(center)))
        safe_draw_circle(overlay_image, center, radius, (0, 255, 0), 2)

    # Draw key areas
    for i in range(6, 10, 2):
        safe_draw_line(
            overlay_image, projected_points[i], projected_points[i + 1], (0, 255, 0), 2
        )

    # Draw 3-point lines and arcs
    if all(valid_point(pt) for pt in projected_points[10:30]):
        pts = np.array([to_int_tuple(pt) for pt in projected_points[10:30]])
        cv2.polylines(overlay_image, [pts], False, (0, 255, 0), 2)

    if all(valid_point(pt) for pt in projected_points[30:50]):
        pts = np.array([to_int_tuple(pt) for pt in projected_points[30:50]])
        cv2.polylines(overlay_image, [pts], False, (0, 255, 0), 2)

    # Draw 3-point corners
    for i in range(50, 54, 2):
        safe_draw_line(
            overlay_image, projected_points[i], projected_points[i + 1], (0, 255, 0), 2
        )

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    plt.title("Court Overlay on Image")
    plt.axis("off")
    plt.show()

    # Save the final overlay image
    cv2.imwrite("data/overlay_image_final.png", overlay_image)


def visualize_calibration_points(image, image_points):
    overlay_image = image.copy()
    for point in image_points:
        if point is not None:
            cv2.circle(overlay_image, (int(point.x), int(point.y)), 5, (0, 255, 0), -1)

    cv2.imwrite("data/calibration_points.png", overlay_image)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    plt.title("Calibration Points on Image")
    plt.axis("off")
    plt.show()


def project_court_schematic(image, camera_matrix, dist_coeffs, rvec, tvec, court):
    court_def = court.court_definition
    h, w = court_def.height, court_def.width

    # Define key court points in 3D
    court_points = np.array(
        [
            [0, 0, 0],  # Bottom Left Corner
            [w, 0, 0],  # Bottom Right Corner
            [w, h, 0],  # Top Right Corner
            [0, h, 0],  # Top Left Corner
            [0, 0, 0],  # Close the outline loop
            [w / 2, h / 2, 0],  # Center of Court
            [w / 2 + court_def.circle_diameter / 2, h / 2, 0],  # Center Circle
            [
                court_def.key_area_length,
                h / 2 - court_def.key_area_width / 2,
                0,
            ],  # Left Key Bottom
            [
                court_def.key_area_length,
                h / 2 + court_def.key_area_width / 2,
                0,
            ],  # Left Key Top
            [
                w - court_def.key_area_length,
                h / 2 - court_def.key_area_width / 2,
                0,
            ],  # Right Key Bottom
            [
                w - court_def.key_area_length,
                h / 2 + court_def.key_area_width / 2,
                0,
            ],  # Right Key Top
            [
                court_def.rim_center_offset,
                h / 2 - court_def.three_point_distance,
                0,
            ],  # Top of Left 3-Point Arc
            [
                w - court_def.rim_center_offset,
                h / 2 - court_def.three_point_distance,
                0,
            ],  # Top of Right 3-Point Arc
            [
                court_def.rim_center_offset,
                h / 2 + court_def.three_point_distance,
                0,
            ],  # Bottom of Left 3-Point Arc
            [
                w - court_def.rim_center_offset,
                h / 2 + court_def.three_point_distance,
                0,
            ],  # Bottom of Right 3-Point Arc
        ],
        dtype=np.float32,
    )

    labels = [
        "Bottom Left Corner",
        "Bottom Right Corner",
        "Top Right Corner",
        "Top Left Corner",
        "Close Loop",
        "Center of Court",
        "Center Circle",
        "Left Key Bottom",
        "Left Key Top",
        "Right Key Bottom",
        "Right Key Top",
        "Top of Left 3-Point Arc",
        "Top of Right 3-Point Arc",
        "Bottom of Left 3-Point Arc",
        "Bottom of Right 3-Point Arc",
    ]

    print("Court Points:")
    print(court_points)

    # Project the court points to the image plane
    projected_points, _ = cv2.projectPoints(
        court_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)

    # Convert projected points to integer tuples and check for valid points
    def valid_point(point):
        return all(0 <= p < dim for p, dim in zip(point, image.shape[1::-1]))

    projected_points = [
        tuple(map(int, point)) for point in projected_points if valid_point(point)
    ]

    print("Projected Points:")
    print(projected_points)

    # Draw the court outline on the image
    overlay_image = image.copy()

    def draw_lines(points, color):
        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i + 1]
            cv2.line(overlay_image, pt1, pt2, color, 2)

    def draw_circle(center, radius, color):
        cv2.circle(overlay_image, center, radius, color, 2)

    # Draw court outline
    draw_lines(projected_points[:5], (0, 255, 0))

    # Draw center circle
    center = projected_points[5]
    radius_point = projected_points[6]
    radius = int(np.linalg.norm(np.array(radius_point) - np.array(center)))
    draw_circle(center, radius, (0, 255, 0))

    # Draw key areas
    draw_lines(projected_points[7:9], (0, 255, 0))
    draw_lines(projected_points[9:11], (0, 255, 0))

    # Draw 3-point arcs
    draw_lines(projected_points[11:13], (0, 255, 0))
    draw_lines(projected_points[13:15], (0, 255, 0))

    # Add labels
    for label, point in zip(labels, projected_points):
        cv2.putText(
            overlay_image,
            label,
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Show the image with the court overlay
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    plt.title("Court Overlay on Image")
    plt.axis("off")
    plt.show()

    return overlay_image
