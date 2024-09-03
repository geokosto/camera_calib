# main.py
import sys
from calibration_tool import CalibrationTool

def main():
    # Check if an image path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    # Get the image path from the command-line argument
    image_path = sys.argv[1]

    # Set the court type (FIBA, NBA, NCAA, NCAAW, or NFHS)
    court_type = "FIBA"

    # Set debug mode (True or False)
    debug_mode = False

    # Create and run the calibration tool
    calibration_tool = CalibrationTool(
        image_path, court_type=court_type, debug=debug_mode
    )
    calibration_tool.run_calibration()

if __name__ == "__main__":
    main()