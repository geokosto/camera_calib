import cv2
import numpy as np
from typing import NamedTuple, List, Tuple
import matplotlib.pyplot as plt
import json


class Point2D(NamedTuple):
    x: float
    y: float


class Point3D(NamedTuple):
    x: float
    y: float
    z: float


class CourtDefinition(NamedTuple):
    width: float
    height: float
    circle_diameter: float
    three_point_distance: float
    three_point_limit: float
    key_area_width: float
    key_area_length: float
    board_offset: float
    board_width: float
    board_height: float
    board_elevation: float
    rim_center_offset: float
    rim_height: float
    rim_radius: float
    no_charge_zone_radius: float


court_definitions = {
    "FIBA": CourtDefinition(
        2800.0,
        1500.0,
        360.0,
        675,
        90.0,
        490.0,
        575.0,
        122,
        183,
        106.6,
        290,
        160,
        305,
        23,
        125,
    ),
    "NBA": CourtDefinition(
        2865.1,
        1524.0,
        366.0,
        723.9,
        90.0,
        488.0,
        575.0,
        122,
        183,
        106.6,
        290,
        160,
        305,
        23,
        122,
    ),
    "NCAA": CourtDefinition(
        2865.1,
        1524.0,
        366.0,
        723.9,
        90.0,
        488.0,
        575.0,
        122,
        183,
        106.6,
        290,
        157.5,
        305,
        23,
        122,
    ),
    "NCAAW": CourtDefinition(
        2865.1,
        1524.0,
        366.0,
        632,
        90.0,
        488.0,
        575.0,
        122,
        183,
        106.6,
        290,
        157.5,
        305,
        23,
        122,
    ),
    "NFHS": CourtDefinition(
        2865.1,
        1524.0,
        366.0,
        602,
        160.0,
        488.0,
        574.0,
        122,
        183,
        106.6,
        290,
        157.5,
        305,
        23,
        122,
    ),
}


class Court:
    def __init__(self, rule_type="FIBA"):
        self.court_definition = court_definitions[rule_type]

    def get_world_points(self) -> List[Tuple[str, Point3D]]:
        w, h = self.court_definition.width, self.court_definition.height
        three_point_limit_top = self.court_definition.three_point_limit
        three_point_limit_bottom = h - self.court_definition.three_point_limit
        return [
            ("Bottom Left Corner", Point3D(0, 0, 0)),
            ("Bottom Right Corner", Point3D(w, 0, 0)),
            ("Top Right Corner", Point3D(w, h, 0)),
            ("Top Left Corner", Point3D(0, h, 0)),
            (
                "Left Free Throw Line (Bottom)",
                Point3D(
                    self.court_definition.key_area_length,
                    h / 2 - self.court_definition.key_area_width / 2,
                    0,
                ),
            ),
            (
                "Right Free Throw Line (Bottom)",
                Point3D(
                    w - self.court_definition.key_area_length,
                    h / 2 - self.court_definition.key_area_width / 2,
                    0,
                ),
            ),
            (
                "Left Free Throw Line (Top)",
                Point3D(
                    self.court_definition.key_area_length,
                    h / 2 + self.court_definition.key_area_width / 2,
                    0,
                ),
            ),
            (
                "Right Free Throw Line (Top)",
                Point3D(
                    w - self.court_definition.key_area_length,
                    h / 2 + self.court_definition.key_area_width / 2,
                    0,
                ),
            ),
            (
                "Left 3-Point Corner",
                Point3D(0, three_point_limit_top, 0),
            ),
            (
                "Right 3-Point Corner",
                Point3D(w, three_point_limit_top, 0),
            ),
            (
                "Top Left 3-Point Corner",
                Point3D(0, three_point_limit_bottom, 0),
            ),
            (
                "Top Right 3-Point Corner",
                Point3D(w, three_point_limit_bottom, 0),
            ),
            (
                "Center of Court",
                Point3D(w / 2, h / 2, 0),
            ),
            (
                "Top of Left 3-Point Arc",
                Point3D(
                    self.court_definition.rim_center_offset,
                    h / 2 + self.court_definition.three_point_distance,
                    0,
                ),
            ),
            (
                "Top of Right 3-Point Arc",
                Point3D(
                    w - self.court_definition.rim_center_offset,
                    h / 2 + self.court_definition.three_point_distance,
                    0,
                ),
            ),
            (
                "Center Line Left Intersection",
                Point3D(0, h / 2, 0),
            ),
            (
                "Center Line Right Intersection",
                Point3D(w, h / 2, 0),
            ),
        ]


class CalibrationTool:
    def __init__(self, image_path: str, court_type: str = "FIBA", debug: bool = False):
        self.image = cv2.imread(image_path)
        self.court = Court(court_type)
        self.world_points = self.court.get_world_points()
        self.image_points = [None] * len(self.world_points)
        self.current_point_index = 0
        self.window_name = "Basketball Court Calibration"
        self.window_width = 1600
        self.window_height = 900
        self.display_image = None
        self.scale_factor = 1
        self.debug = debug
        self.calibration_points_file = "data/calibration_points.json"

    def create_court_schematic(self):
        court_def = self.court.court_definition
        margin = 50
        schematic_width = self.window_width // 2 - 2 * margin
        schematic_height = (
            self.window_height - 2 * margin - 80
        )  # More space at the bottom
        scale = min(
            schematic_width / court_def.width, schematic_height / court_def.height
        )

        schematic = (
            np.ones((self.window_height, self.window_width // 2, 3), dtype=np.uint8)
            * 255
        )

        def scale_point(x, y):
            return (
                int(x * scale) + margin,
                int((court_def.height - y) * scale) + margin,  # Flip y-axis
            )

        # Draw court outline
        cv2.rectangle(
            schematic,
            scale_point(0, 0),
            scale_point(court_def.width, court_def.height),
            (0, 0, 0),
            2,
        )

        # Draw center circle
        center = scale_point(court_def.width / 2, court_def.height / 2)
        radius = int(court_def.circle_diameter / 2 * scale)
        cv2.circle(schematic, center, radius, (0, 0, 0), 2)

        # Draw key areas
        cv2.rectangle(
            schematic,
            scale_point(0, court_def.height / 2 - court_def.key_area_width / 2),
            scale_point(
                court_def.key_area_length,
                court_def.height / 2 + court_def.key_area_width / 2,
            ),
            (0, 0, 0),
            2,
        )
        cv2.rectangle(
            schematic,
            scale_point(
                court_def.width - court_def.key_area_length,
                court_def.height / 2 - court_def.key_area_width / 2,
            ),
            scale_point(
                court_def.width, court_def.height / 2 + court_def.key_area_width / 2
            ),
            (0, 0, 0),
            2,
        )

        # Draw 3-point lines and arcs
        left_arc_center = scale_point(court_def.rim_center_offset, court_def.height / 2)
        right_arc_center = scale_point(
            court_def.width - court_def.rim_center_offset, court_def.height / 2
        )
        three_point_radius = int(court_def.three_point_distance * scale)
        cv2.ellipse(
            schematic,
            left_arc_center,
            (three_point_radius, three_point_radius),
            0,
            -90,
            90,
            (0, 0, 0),
            2,
        )
        cv2.ellipse(
            schematic,
            right_arc_center,
            (three_point_radius, three_point_radius),
            0,
            90,
            270,
            (0, 0, 0),
            2,
        )

        # Draw points
        for i, (name, point) in enumerate(self.world_points):
            pt = scale_point(point.x, point.y)
            color = (0, 255, 0) if self.image_points[i] is not None else (0, 0, 255)
            cv2.circle(schematic, pt, 5, color, -1)
            cv2.putText(
                schematic,
                f"{i+1}",
                (pt[0] + 5, pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

        # Highlight current point
        if self.current_point_index < len(self.world_points):
            current_pt = scale_point(
                self.world_points[self.current_point_index][1].x,
                self.world_points[self.current_point_index][1].y,
            )
            cv2.circle(schematic, current_pt, 8, (255, 0, 0), 2)

        return schematic

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_point_index < len(
            self.world_points
        ):
            if x < self.window_width // 2:  # Click on the image
                # Adjust coordinates based on scaling and offset
                scaled_x = int(
                    (x - 10) / self.scale_factor
                )  # Subtract 10 for the left margin
                scaled_y = int((y - self.y_offset) / self.scale_factor)

                # Ensure the point is within the image bounds
                if (
                    0 <= scaled_x < self.image.shape[1]
                    and 0 <= scaled_y < self.image.shape[0]
                ):
                    self.image_points[self.current_point_index] = Point2D(
                        scaled_x, scaled_y
                    )
                    self.current_point_index += 1
            self.update_display()

    def update_display(self):
        display = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        # Calculate scale factor to fit the image in half the window width
        self.scale_factor = (self.window_width // 2 - 20) / self.image.shape[1]
        scaled_height = int(self.image.shape[0] * self.scale_factor)

        # Resize image to fit half the window width
        self.display_image = cv2.resize(
            self.image, (self.window_width // 2 - 20, scaled_height)
        )

        # Create a black background for the image side
        image_background = np.zeros(
            (self.window_height, self.window_width // 2, 3), dtype=np.uint8
        )

        # Calculate vertical offset to center the scaled image
        self.y_offset = (self.window_height - scaled_height) // 2

        # Place the scaled image onto the black background
        image_background[self.y_offset : self.y_offset + scaled_height, 10:-10] = (
            self.display_image
        )

        # Draw selected points on the display image
        for point in self.image_points:
            if point is not None:
                cv2.circle(
                    image_background,
                    (
                        int(point.x * self.scale_factor) + 10,
                        int(point.y * self.scale_factor) + self.y_offset,
                    ),
                    5,
                    (0, 255, 0),
                    -1,
                )

        # Update the display
        display[:, : self.window_width // 2] = image_background
        display[:, self.window_width // 2 :] = self.create_court_schematic()

        # Add a vertical line to separate the image and schematic
        cv2.line(
            display,
            (self.window_width // 2, 0),
            (self.window_width // 2, self.window_height),
            (200, 200, 200),
            2,
        )

        # Add instructions (two lines)
        cv2.putText(
            display,
            "Click on the image to select points.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display,
            "Press 'n' to skip a point. Press 'q' to quit.",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Add point list in 3 columns
        points_per_column = (len(self.world_points) + 2) // 3
        for i, (name, _) in enumerate(self.world_points):
            color = (0, 255, 0) if self.image_points[i] is not None else (0, 0, 255)
            status = "Selected" if self.image_points[i] is not None else "Not Selected"
            column = i // points_per_column
            row = i % points_per_column
            x_offset = self.window_width // 2 + (column * (self.window_width // 6))
            y_offset = 100 + row * 50 + 500
            cv2.putText(
                display,
                f"{i+1}. {name}",
                (x_offset, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
            cv2.putText(
                display,
                f"   {status}",
                (x_offset, y_offset + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

        # Highlight current point
        if self.current_point_index < len(self.world_points):
            cv2.putText(
                display,
                f"Current point: {self.current_point_index + 1}. {self.world_points[self.current_point_index][0]}",
                (20, self.window_height - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # Add progress bar
        selected_points = sum(1 for p in self.image_points if p is not None)
        progress = selected_points / len(self.world_points)
        cv2.rectangle(
            display,
            (20, self.window_height - 60),
            (self.window_width - 20, self.window_height - 30),
            (100, 100, 100),
            -1,
        )
        cv2.rectangle(
            display,
            (20, self.window_height - 60),
            (int(20 + progress * (self.window_width - 40)), self.window_height - 30),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            display,
            f"Progress: {selected_points}/{len(self.world_points)} points",
            (self.window_width // 2 - 100, self.window_height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(self.window_name, display)
        # Add progress bar
        selected_points = sum(1 for p in self.image_points if p is not None)
        progress = selected_points / len(self.world_points)
        cv2.rectangle(
            display,
            (20, self.window_height - 60),
            (self.window_width - 20, self.window_height - 30),
            (100, 100, 100),
            -1,
        )
        cv2.rectangle(
            display,
            (20, self.window_height - 60),
            (int(20 + progress * (self.window_width - 40)), self.window_height - 30),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            display,
            f"Progress: {selected_points}/{len(self.world_points)} points",
            (self.window_width // 2 - 100, self.window_height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(self.window_name, display)

    def save_calibration_points(self):
        data = {"image_points": [(p.x, p.y) if p else None for p in self.image_points]}
        with open(self.calibration_points_file, "w") as f:
            json.dump(data, f)

    def load_calibration_points(self):
        try:
            with open(self.calibration_points_file, "r") as f:
                data = json.load(f)
            self.image_points = [
                Point2D(*p) if p else None for p in data["image_points"]
            ]
            print("Calibration points loaded successfully.")
        except FileNotFoundError:
            print(
                "No saved calibration points found. Please run in non-debug mode first."
            )
            return False
        return True

    def run_calibration(self):
        if self.debug:
            if not self.load_calibration_points():
                print("Please run the calibration tool in non-debug mode first.")
                return
        else:
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)

            while True:
                self.update_display()
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("n"):  # Skip current point
                    self.current_point_index += 1
                    if self.current_point_index >= len(self.world_points):
                        break

            cv2.destroyAllWindows()

            if any(point is not None for point in self.image_points):
                self.save_calibration_points()
            else:
                print("Calibration incomplete. No points were selected.")
                return

        if any(point is not None for point in self.image_points):
            self.perform_calibration()
        else:
            print("Calibration incomplete. No points were selected.")

    def perform_calibration(self):
        valid_points = [
            (w, i)
            for w, i in zip(self.world_points, self.image_points)
            if i is not None
        ]
        if len(valid_points) < 6:
            print("Not enough points for calibration. At least 6 points are required.")
            return

        world_points = np.array(
            [[p.x, p.y, p.z] for (_, p), _ in valid_points], dtype=np.float32
        )
        image_points = np.array([(p.x, p.y) for _, p in valid_points], dtype=np.float32)

        # Use calibrateCamera instead of solvePnP
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            [world_points], [image_points], self.image.shape[:2][::-1], None, None
        )

        if not ret:
            print("Calibration failed.")
            return

        print("Camera Matrix:\n", camera_matrix)
        print("\nDistortion Coefficients:\n", dist_coeffs.ravel())

        # Project 3D points to 2D image plane
        projected_points, _ = cv2.projectPoints(
            world_points, rvecs[0], tvecs[0], camera_matrix, dist_coeffs
        )

        # Calculate reprojection error
        error = cv2.norm(
            image_points, projected_points.reshape(-1, 2), cv2.NORM_L2
        ) / len(projected_points)
        print(f"\nReprojection Error: {error} pixels")

        # Visualize results
        self.visualize_court_overlay(camera_matrix, dist_coeffs, rvecs[0], tvecs[0])

        # Save the calibration results
        np.savez(
            "data/camera_calibration.npz",
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rvecs=rvecs,
            tvecs=tvecs,
        )
        print("Calibration results saved to camera_calibration.npz")

    def visualize_results(self, world_points, image_points, projected_points):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Plot the original image with points
        ax1.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        ax1.scatter(image_points[:, 0], image_points[:, 1], c="g", label="Selected")
        ax1.scatter(
            projected_points[:, 0, 0],
            projected_points[:, 0, 1],
            c="r",
            label="Projected",
        )
        for i, (img_pt, proj_pt) in enumerate(zip(image_points, projected_points)):
            ax1.plot([img_pt[0], proj_pt[0, 0]], [img_pt[1], proj_pt[0, 1]], "b-")
            ax1.text(img_pt[0], img_pt[1], str(i + 1), fontsize=8, color="white")
        ax1.legend()
        ax1.set_title("Image with Calibration Points")

        # Plot the court schematic with projected points
        court_width = self.court.court_definition.width
        court_height = self.court.court_definition.height
        ax2.set_xlim(0, court_width)
        ax2.set_ylim(0, court_height)
        ax2.set_aspect("equal")

        # Draw court outline
        ax2.add_patch(plt.Rectangle((0, 0), court_width, court_height, fill=False))

        # Draw center circle
        center_x, center_y = court_width / 2, court_height / 2
        circle_radius = self.court.court_definition.circle_diameter / 2
        ax2.add_patch(plt.Circle((center_x, center_y), circle_radius, fill=False))

        # Draw 3-point line
        three_point_radius = self.court.court_definition.three_point_distance
        ax2.add_patch(plt.Circle((0, center_y), three_point_radius, fill=False))
        ax2.add_patch(
            plt.Circle((court_width, center_y), three_point_radius, fill=False)
        )

        # Plot world points
        ax2.scatter(world_points[:, 0], world_points[:, 1], c="b", label="World Points")
        for i, point in enumerate(world_points):
            ax2.text(point[0], point[1], str(i + 1), fontsize=8)

        ax2.legend()
        ax2.set_title("Court Schematic with World Points")

        plt.tight_layout()
        plt.show()

    def visualize_court_overlay(self, camera_matrix, dist_coeffs, rvec, tvec):
        court_def = self.court.court_definition
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
                [
                    -half_width + court_def.key_area_length,
                    -court_def.key_area_width / 2,
                    0,
                ],
                [
                    -half_width + court_def.key_area_length,
                    court_def.key_area_width / 2,
                    0,
                ],
                [
                    half_width - court_def.key_area_length,
                    -court_def.key_area_width / 2,
                    0,
                ],
                [
                    half_width - court_def.key_area_length,
                    court_def.key_area_width / 2,
                    0,
                ],
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

        projected_points, _ = cv2.projectPoints(
            court_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)

        overlay_image = self.image.copy()

        def valid_point(point):
            x, y = point
            return not (np.isinf(x) or np.isinf(y) or np.isnan(x) or np.isnan(y))

        def to_int_tuple(point):
            return tuple(map(int, point))

        if all(valid_point(pt) for pt in projected_points[:4]):
            cv2.polylines(
                overlay_image,
                [np.array([to_int_tuple(pt) for pt in projected_points[:4]])],
                True,
                (0, 255, 0),
                2,
            )

        if valid_point(projected_points[4]) and valid_point(projected_points[5]):
            center = to_int_tuple(projected_points[4])
            radius_point = to_int_tuple(projected_points[5])
            radius = int(np.linalg.norm(np.array(radius_point) - np.array(center)))
            cv2.circle(overlay_image, center, radius, (0, 255, 0), 2)

        if all(valid_point(pt) for pt in projected_points[6:10]):
            cv2.line(
                overlay_image,
                to_int_tuple(projected_points[6]),
                to_int_tuple(projected_points[7]),
                (0, 255, 0),
                2,
            )
            cv2.line(
                overlay_image,
                to_int_tuple(projected_points[8]),
                to_int_tuple(projected_points[9]),
                (0, 255, 0),
                2,
            )

        if all(valid_point(pt) for pt in projected_points[10:30]):
            cv2.polylines(
                overlay_image,
                [np.array([to_int_tuple(pt) for pt in projected_points[10:30]])],
                False,
                (0, 255, 0),
                2,
            )
        if all(valid_point(pt) for pt in projected_points[30:50]):
            cv2.polylines(
                overlay_image,
                [np.array([to_int_tuple(pt) for pt in projected_points[30:50]])],
                False,
                (0, 255, 0),
                2,
            )

        if all(valid_point(pt) for pt in projected_points[50:54]):
            cv2.line(
                overlay_image,
                to_int_tuple(projected_points[50]),
                to_int_tuple(projected_points[51]),
                (0, 255, 0),
                2,
            )
            cv2.line(
                overlay_image,
                to_int_tuple(projected_points[52]),
                to_int_tuple(projected_points[53]),
                (0, 255, 0),
                2,
            )

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        plt.title("Improved Court Overlay on Image")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    # image_path = "data/basketball_court.jpg"  # Change this to the path of your image
    image_path = "data/camcourt2_1512419072502_0.png"
    court_type = "FIBA"  # Change this to the appropriate court type
    calibration_tool = CalibrationTool(image_path, court_type=court_type, debug=True)
    calibration_tool.run_calibration()
