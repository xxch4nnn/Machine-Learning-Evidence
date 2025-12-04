import cv2
import numpy as np
import threading
import time
from cv2 import aruco

# --- DIGITAL TWIN CONFIGURATION (MUST MATCH GENERATOR) ---
CONFIG = {
    'MARKER_SIZE': 200.0,    # 3D Unit = 1 Pixel
    'SAFETY_GAP': 50.0,
    'BORDER_WIDTH': 20.0,
    'KEY_WIDTH': 100.0,
    'KEY_HEIGHT': 400.0,
    'NUM_KEYS': 7
}

class ThreadedCamera:
    """ Producer-Consumer Threaded Video Capture """
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        self.lock = threading.Lock()
        self._frame = None
        self.running = True
        
        success, frame = self.capture.read()
        if success: self._frame = frame
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    with self.lock:
                        self._frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()

class VisionEngine:
    def __init__(self, src=0):
        self.cam = ThreadedCamera(src)
        
        # --- ARUCO CONFIG ---
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.params)
        
        # --- 3D OBJECT POINTS DEFINITION ---
        
        # 1. Marker Object Points (The Anchor)
        # Defined counter-clockwise from Top-Left to match ArUco detection order
        # Origin (0,0,0) is the Top-Left corner of the marker
        s = CONFIG['MARKER_SIZE']
        self.marker_points = np.array([
            [0, 0, 0],    # Top-Left
            [s, 0, 0],    # Top-Right
            [s, s, 0],    # Bot-Right
            [0, s, 0]     # Bot-Left
        ], dtype=np.float32)
        
        # 2. Piano Grid Points (Relative to Marker Origin)
        self.piano_points, self.key_lines = self._generate_piano_points()

    def _generate_piano_points(self):
        """ Replicates the Generator's layout math in 3D space (Z=0) """
        
        # Calculate X Offset: Marker + Gap + Border
        start_x = CONFIG['MARKER_SIZE'] + CONFIG['SAFETY_GAP'] + CONFIG['BORDER_WIDTH']
        
        # Calculate Width
        total_width = CONFIG['KEY_WIDTH'] * CONFIG['NUM_KEYS']
        end_x = start_x + total_width
        
        # Y Dimensions (Top aligned with marker top)
        y_top = 0.0
        y_bot = CONFIG['KEY_HEIGHT']
        
        # A. The Bounding Box (Green Border)
        outline = np.array([
            [start_x, y_top, 0],  # TL
            [end_x, y_top, 0],    # TR
            [end_x, y_bot, 0],    # BR
            [start_x, y_bot, 0]   # BL
        ], dtype=np.float32)
        
        # B. Vertical Key Separators
        lines = []
        for i in range(1, CONFIG['NUM_KEYS']):
            x = start_x + (i * CONFIG['KEY_WIDTH'])
            lines.append([x, y_top, 0])
            lines.append([x, y_bot, 0])
            
        return outline, np.array(lines, dtype=np.float32)

    def estimate_intrinsics(self, w, h):
        """ Basic assumption for uncalibrated camera """
        f = w  # Focal length approx width
        return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)

    def run(self):
        print("Starting Vision Engine (Digital Twin Mode)...")
        print("Looking for Left Marker (ID 0)...")
        
        prev_time = time.time()
        fps_log = []
        
        try:
            while True:
                frame = self.cam.read()
                if frame is None: continue
                
                h, w = frame.shape[:2]
                cam_mat = self.estimate_intrinsics(w, h)
                dist = np.zeros((4,1))
                
                # Detect
                corners, ids, _ = self.detector.detectMarkers(frame)
                
                status = "Searching..."
                color = (0, 0, 255) # Red
                
                if ids is not None:
                    # Draw visual debug for all markers
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    ids_flat = ids.flatten()
                    if 0 in ids_flat:
                        # FOUND ANCHOR
                        status = "Tracking (Locked)"
                        color = (0, 255, 0) # Green
                        
                        idx = np.where(ids_flat == 0)[0][0]
                        c0 = corners[idx].reshape((4, 2))
                        
                        # 1. Get Pose of Marker 0 relative to Camera
                        success, rvec, tvec = cv2.solvePnP(self.marker_points, c0, cam_mat, dist)
                        
                        if success:
                            # 2. Project Piano Outline
                            img_pts_border, _ = cv2.projectPoints(self.piano_points, rvec, tvec, cam_mat, dist)
                            img_pts_border = np.int32(img_pts_border).reshape(-1, 2)
                            
                            # Draw Green Box
                            cv2.polylines(frame, [img_pts_border], True, (0, 255, 0), 2)
                            
                            # 3. Project & Draw Key Lines
                            if len(self.key_lines) > 0:
                                key_pts, _ = cv2.projectPoints(self.key_lines, rvec, tvec, cam_mat, dist)
                                key_pts = np.int32(key_pts).reshape(-1, 2)
                                
                                for k in range(0, len(key_pts), 2):
                                    cv2.line(frame, tuple(key_pts[k]), tuple(key_pts[k+1]), (0, 255, 0), 1)
                                    
                            # 4. Draw Axis on Anchor (Visual Verification)
                            cv2.drawFrameAxes(frame, cam_mat, dist, rvec, tvec, 100.0)

                # UI Overlay
                curr = time.time()
                dt = curr - prev_time
                prev_time = curr
                if dt > 0:
                    fps_log.append(1.0/dt)
                    if len(fps_log) > 30: fps_log.pop(0)
                    fps = sum(fps_log)/len(fps_log)
                else: fps = 0
                
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow("Vision Engine", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        finally:
            self.cam.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    VisionEngine().run()