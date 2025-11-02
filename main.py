import sys
from Machine_Learning_Course.Code.ArUco_Markers.Detecting_an_aruco_marker.DetectArucoLive import run_piano

if __name__ == "__main__":
    # Set the camera index (0 for the default camera)
    camera_index = 0
    run_piano(camera_index)
