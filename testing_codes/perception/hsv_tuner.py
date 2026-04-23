import pyrealsense2 as rs
import cv2
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

def nothing(x): pass

cv2.namedWindow("Tuner")
cv2.createTrackbar("H Low",  "Tuner", 0,   179, nothing)
cv2.createTrackbar("H High", "Tuner", 179, 179, nothing)
cv2.createTrackbar("S Low",  "Tuner", 0,   255, nothing)
cv2.createTrackbar("S High", "Tuner", 255, 255, nothing)
cv2.createTrackbar("V Low",  "Tuner", 0,   255, nothing)
cv2.createTrackbar("V High", "Tuner", 255, 255, nothing)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hl = cv2.getTrackbarPos("H Low",  "Tuner")
        hh = cv2.getTrackbarPos("H High", "Tuner")
        sl = cv2.getTrackbarPos("S Low",  "Tuner")
        sh = cv2.getTrackbarPos("S High", "Tuner")
        vl = cv2.getTrackbarPos("V Low",  "Tuner")
        vh = cv2.getTrackbarPos("V High", "Tuner")

        mask   = cv2.inRange(hsv, np.array([hl,sl,vl]), np.array([hh,sh,vh]))
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("Original", frame)
        cv2.imshow("Mask",     mask)
        cv2.imshow("Result",   result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            print(f"lower_hsv = np.array([{hl}, {sl}, {vl}])")
            print(f"upper_hsv = np.array([{hh}, {sh}, {vh}])")
        if key == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()