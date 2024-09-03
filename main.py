# main.py
from calibration_tool import CalibrationTool


def main():
    # Set the path to your image file
    image_path = "data/camcourt1_1512416912203_40.png"

    # Set the court type (FIBA, NBA, NCAA, NCAAW, or NFHS)
    court_type = "FIBA"

    # Set debug mode (True or False)
    debug_mode = True

    # Create and run the calibration tool
    calibration_tool = CalibrationTool(
        image_path, court_type=court_type, debug=debug_mode
    )
    calibration_tool.run_calibration()


if __name__ == "__main__":
    main()
