import cv2
import numpy as np
import json
from point import Point2D
from court import Court
from visualization import (
    project_court_schematic,
    visualize_calibration_points,
    visualize_results,
    visualize_court_overlay,
)


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
        schematic_height = self.window_height - 2 * margin - 80

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
                int((court_def.height - y) * scale) + margin,
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
                scaled_x = int((x - 10) / self.scale_factor)
                scaled_y = int((y - self.y_offset) / self.scale_factor)

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

        self.scale_factor = (self.window_width // 2 - 20) / self.image.shape[1]
        scaled_height = int(self.image.shape[0] * self.scale_factor)

        self.display_image = cv2.resize(
            self.image, (self.window_width // 2 - 20, scaled_height)
        )

        image_background = np.zeros(
            (self.window_height, self.window_width // 2, 3), dtype=np.uint8
        )

        self.y_offset = (self.window_height - scaled_height) // 2

        image_background[self.y_offset : self.y_offset + scaled_height, 10:-10] = (
            self.display_image
        )

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

        display[:, : self.window_width // 2] = image_background
        display[:, self.window_width // 2 :] = self.create_court_schematic()

        cv2.line(
            display,
            (self.window_width // 2, 0),
            (self.window_width // 2, self.window_height),
            (200, 200, 200),
            2,
        )

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
                elif key == ord("n"):
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
            # Visualize calibration points
            # visualize_calibration_points(self.image, self.image_points)
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

        print("World Points for Calibration:")
        print(world_points)
        print("Image Points for Calibration:")
        print(image_points)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-8)
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            [world_points],
            [image_points],
            self.image.shape[:2][::-1],
            None,
            None,
            criteria,
        )

        if not ret:
            print("Calibration failed.")
            return

        print("Camera Matrix:\n", camera_matrix)
        print("\nDistortion Coefficients:\n", dist_coeffs.ravel())
        print("\nRotation Vectors:\n", rvecs)
        print("\nTranslation Vectors:\n", tvecs)

        projected_points, _ = cv2.projectPoints(
            world_points, rvecs[0], tvecs[0], camera_matrix, dist_coeffs
        )

        error = cv2.norm(
            image_points, projected_points.reshape(-1, 2), cv2.NORM_L2
        ) / len(projected_points)
        print(f"\nReprojection Error: {error} pixels")

        # try:
        #     visualize_court_overlay(
        #         self.image, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], self.court
        #     )
        # except Exception as e:
        #     print(f"Error visualizing court overlay: {e}")

        # project_court_schematic(image, camera_matrix, dist_coeffs, rvec, tvec, court)
        # project_court_schematic(
        #     self.image, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], self.court
        # )

        np.savez(
            "data/camera_calibration.npz",
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rvecs=rvecs,
            tvecs=tvecs,
        )
        print("Calibration results saved to camera_calibration.npz")

        # Visualize calibration results
        from deepsport_utilities.deepsport_utilities.court import Court
        from calib3d.calib3d.calib import Calib

        calib = Calib(
            width=self.image.shape[1],
            height=self.image.shape[0],
            T=tvecs[0],
            R=cv2.Rodrigues(rvecs[0])[0],  # Convert rotation vector to matrix
            K=camera_matrix,
            # kc=dist_coeffs,
        )

        court_rule_type = "FIBA"
        court = Court(rule_type=court_rule_type)
        court.draw_lines(self.image, calib)

        # Display the image with the court drawn on it
        cv2.imshow("Court Calibration", cv2.resize(self.image, (0, 0), fx=0.7, fy=0.7))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the image with court lines
        cv2.imwrite("data/court_with_lines.png", self.image)