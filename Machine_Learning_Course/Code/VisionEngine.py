import cv2
import numpy as np
import threading
import time
from cv2 import aruco

class ThreadedCamera:
    """
    A threaded camera class that uses the Producer-Consumer pattern.
    The Producer (background thread) continuously reads frames from the camera.
    The Consumer (main thread) reads the latest frame instantly.
    """
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Verify camera opened
        if not self.capture.isOpened():
             print(f"Warning: Unable to open camera source {src}")

        self.lock = threading.Lock()
        self._current_frame = None
        self.is_running = True

        # Read the first frame to ensure we have something to return
        success, frame = self.capture.read()
        if success:
            self._current_frame = frame

        # Start the producer thread as a daemon
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """
        Producer Loop: continuously reads frames from the camera
        and updates the shared _current_frame variable.
        """
        while self.is_running:
            if self.capture.isOpened():
                success, frame = self.capture.read()
                if success:
                    with self.lock:
                        self._current_frame = frame
                else:
                    # If we can't read, maybe the camera disconnected or stream ended
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def read(self):
        """
        Consumer: returns the latest frame instantly.
        """
        with self.lock:
            return self._current_frame

    def stop(self):
        """
        Stops the thread and releases resources.
        """
        self.is_running = False
        self.thread.join()
        self.capture.release()

def estimate_camera_matrix(frame_width, frame_height):
    """
    Estimates the camera matrix based on frame dimensions.
    Focal Length = Frame Width
    Center X = Frame Width / 2
    Center Y = Frame Height / 2
    """
    focal_length = frame_width
    center_x = frame_width / 2
    center_y = frame_height / 2

    camera_matrix = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)

    return camera_matrix

def draw_3d_axis(image, corners, ids, camera_matrix, dist_coeffs):
    """
    Draws a 3D bounding box on the detected ArUco markers.
    """
    if ids is None or corners is None:
        return image

    # Define a 3D box in the marker's local coordinate space
    # z=0 is the surface, z=0.1 is protruding out (assuming marker length 1.0 logic roughly)
    # Adjust scale as needed for visualization.
    # Let's assume marker size is roughly 1 unit for the box definition or normalized.
    # The user asked for "z=0 to z=0.1".
    # Standard corners are often normalized, but let's define a box relative to the marker square.
    # Marker corners in local space are often: (-0.5, 0.5, 0), (0.5, 0.5, 0), (0.5, -0.5, 0), (-0.5, -0.5, 0) if centered
    # OR (0,0,0), (1,0,0), (1,1,0), (0,1,0) depending on convention.
    # solvePnP generic object points for a square marker:
    marker_length = 0.05 # 5cm example, or just unitless 1.0.
    # The visual size depends on the object points defined here matching the physical size in tvec units,
    # OR we just use a consistent relative scale.
    # Let's use a normalized coordinate system for the marker:
    # Top-Left: (0,0,0), Top-Right: (1,0,0), Bottom-Right: (1,1,0), Bottom-Left: (0,1,0)
    # The `corners` from detectMarkers are typically TopLeft, TopRight, BottomRight, BottomLeft.

    obj_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    # We iterate through each detected marker
    for i in range(len(ids)):
        # Get corners for this marker
        # corners[i] shape is (1, 4, 2)
        current_corners = corners[i].reshape((4, 2))

        # Calculate Pose
        success, rvec, tvec = cv2.solvePnP(obj_points, current_corners, camera_matrix, dist_coeffs)

        if success:
            # Define 3D Box points to project
            # Base (z=0) is already the corners. Top (z=-0.5) - Z axis usually points OUT or IN?
            # In OpenCV, Z is forward. For a flat marker on a table, Z is usually up/down relative to camera.
            # Usually we define the box protruding 'up' from the marker.
            # If Z points into the scene, 'up' from the marker surface might be negative Z in local space?
            # Let's try drawing a box from z=0 to z=-1 (negative usually towards camera in local marker frame if Z is down? No.)
            # Standard: X right, Y down, Z forward (camera).
            # For marker: X right, Y up, Z forward (out of board)?
            # Let's stick to the prompt: "z=0 to z=0.1".

            # The prompt says: "z=0 to z=0.1". I will follow that literally.

            box_points_3d = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],       # Base
                [0, 0, 0.1], [1, 0, 0.1], [1, 1, 0.1], [0, 1, 0.1] # Top
            ], dtype=np.float32)

            # Project points
            img_points, _ = cv2.projectPoints(box_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
            img_points = np.int32(img_points).reshape(-1, 2)

            # Draw Base
            cv2.drawContours(image, [img_points[:4]], -1, (0, 255, 0), 2)

            # Draw Pillars
            for j in range(4):
                cv2.line(image, tuple(img_points[j]), tuple(img_points[j+4]), (255, 0, 0), 2)

            # Draw Top
            cv2.drawContours(image, [img_points[4:]], -1, (0, 0, 255), 2)

    return image

def main():
    # Initialize Camera
    # Try index 0, then 1 if needed. The user used 1 in their legacy code.
    # I'll default to 0, but user can change.
    camera_src = 0
    camera = ThreadedCamera(src=camera_src)

    # Initialize ArUco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # FPS Variables
    prev_time = time.time()
    fps_history = []

    print("Starting Vision Engine...")
    print("Press 'q' to quit.")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.1)
                continue

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time

            # Avoid division by zero
            if dt > 0:
                fps = 1.0 / dt
                fps_history.append(fps)
                if len(fps_history) > 30: # Rolling average window
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)
            else:
                avg_fps = 0

            # Process Frame
            height, width = frame.shape[:2]

            # Estimate Camera Matrix
            camera_matrix = estimate_camera_matrix(width, height)
            dist_coeffs = np.zeros((4, 1)) # Assuming no distortion for basic estimation

            # Detect Markers
            # Detect markers needs grayscale? detectMarkers handles it usually, but let's be safe.
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Actually ArucoDetector.detectMarkers can take color, but gray is standard.
            corners, ids, rejected = detector.detectMarkers(frame)

            # Visualization
            if ids is not None:
                # Draw standard markers
                aruco.drawDetectedMarkers(frame, corners, ids)

                # Draw 3D Boxes
                frame = draw_3d_axis(frame, corners, ids, camera_matrix, dist_coeffs)

            # Draw FPS
            cv2.putText(frame, f"FPS: {int(avg_fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show Frame
            cv2.imshow('Vision Engine - 3D Threaded', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
