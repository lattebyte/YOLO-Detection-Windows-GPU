# Quick solutions during test, not recommended for production
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import sys
import time
import argparse
from lattebyte_cm16a_camera import initialize_camera, configure_camera
from yolo_gpu_setting import configure_yolo

class CustomArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser to override default error message."""
    def error(self, message):
        sys.stderr.write(f"Error: {message}\n\n")
        self.print_help()  # Show the help message on error
        sys.exit(2)

def main(model_name):
    """Main function to initialize, configure camera, and use the provided model."""
    print(f"Using model: {model_name}")

    # Initialize the camera
    lattebyte_cm16a = initialize_camera()

    # Configure camera settings
    configure_camera(lattebyte_cm16a)

    # Configure yolo settings
    yolo_gpu = configure_yolo(model_name)

    # Loop through the frames
    while True:
        start_time = time.time()
        
        ret, frame = lattebyte_cm16a.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform object detection
        results = yolo_gpu(frame, verbose=True)  # Perform inference
        
        # Loop through detected objects and annotate
        for result in results:
            boxes = result.boxes  # Bounding boxes
            for box in boxes:
                # Get the coordinates of the box
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Top-left (x1, y1) and bottom-right (x2, y2) coordinates
                
                # Calculate the center of the box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Calculate the width and height of the box
                width = int(x2 - x1)
                height = int(y2 - y1)
                
                # Annotate the center coordinates on the frame
                center_text = f"Center: ({center_x}, {center_y})"
                cv2.putText(frame, center_text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Annotate the width and height on the frame
                size_text = f"Width: {width}px, Height: {height}px"
                cv2.putText(frame, size_text, (center_x, center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw the bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Calculate and display FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("YOLO Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    lattebyte_cm16a.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Use the custom argument parser
    parser = CustomArgumentParser(description="Run camera with specific YOLO model.")    
    parser.add_argument(
        "model_name", 
        type=str, 
        help="The model name to use (e.g., yolo11n, yolov8n, yolov5nu, and more)"
    )

    # Parse the arguments
    args = parser.parse_args()
    
    main(args.model_name)  # Pass the argument to the main function