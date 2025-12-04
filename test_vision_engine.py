import sys
import os
import numpy as np
import cv2

# Add the directory to path to import VisionEngine
sys.path.append('Machine_Learning_Course/Code')

try:
    import VisionEngine
    print("Successfully imported VisionEngine")
except ImportError as e:
    print(f"Failed to import VisionEngine: {e}")
    sys.exit(1)

def test_camera_matrix():
    print("Testing estimate_camera_matrix...")
    w, h = 640, 480
    matrix = VisionEngine.estimate_camera_matrix(w, h)

    expected = np.array([
        [640, 0, 320],
        [0, 640, 240],
        [0, 0, 1]
    ], dtype=np.float32)

    if np.allclose(matrix, expected):
        print("Camera matrix estimation: PASS")
    else:
        print(f"Camera matrix estimation: FAIL\nExpected:\n{expected}\nGot:\n{matrix}")

def test_draw_3d_axis():
    print("Testing draw_3d_axis...")
    # Create a dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Dummy corners for a marker
    corners = [np.array([[[100, 100], [200, 100], [200, 200], [100, 200]]], dtype=np.float32)]
    ids = np.array([[1]])

    matrix = VisionEngine.estimate_camera_matrix(640, 480)
    dist = np.zeros((4, 1))

    try:
        res_img = VisionEngine.draw_3d_axis(img, corners, ids, matrix, dist)
        print("draw_3d_axis execution: PASS")
    except Exception as e:
        print(f"draw_3d_axis execution: FAIL with error {e}")

def test_threaded_camera_init():
    print("Testing ThreadedCamera init (expecting warning for no camera)...")
    try:
        # This might fail if no /dev/video0, but code handles it gracefully
        cam = VisionEngine.ThreadedCamera(src=0) # invalid index
        cam.stop()
        print("ThreadedCamera init and stop: PASS")
    except Exception as e:
        print(f"ThreadedCamera init/stop failed: {e}")

if __name__ == "__main__":
    test_camera_matrix()
    test_draw_3d_axis()
    test_threaded_camera_init()
