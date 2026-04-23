#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, TwistStamped
from collections import deque
import tf2_ros
import tf2_geometry_msgs
from filterpy.kalman import KalmanFilter


# ── Kalman filter: state = [x, y, z, vx, vy, vz] ──────────────────────────
def make_kalman(dt=1/30.0):
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.F = np.array([
        [1,0,0,dt,0,0],
        [0,1,0,0,dt,0],
        [0,0,1,0,0,dt],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
    ], dtype=float)
    kf.H  = np.eye(3, 6)
    kf.R  *= 0.01    # measurement noise  — tune if position jumps
    kf.Q  *= 0.001   # process noise      — tune if velocity is laggy
    kf.P  *= 10.0
    return kf


class BallPerception:
    def __init__(self):
        rospy.init_node("ball_perception")

        # ── Parameters (set these) ──────────────────────────────────────────
        self.lower_hsv     = np.array([5,  150, 150])   # from Step 3
        self.upper_hsv     = np.array([20, 255, 255])   # from Step 3
        self.ball_radius_m = 0.05                        # real radius in metres
        self.lower_hsv = np.array([18, 83, 138])
        self.upper_hsv = np.array([49, 241, 255])
        self.min_radius_px = 10                          # ignore tiny detections
        self.target_frame  = "base_link"                 # your robot's base frame

        # ── Internal state ──────────────────────────────────────────────────
        self.bridge      = CvBridge()
        self.depth_image = None
        self.cam_info    = None
        self.kf          = make_kalman()
        self.kf_init     = False

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ── Subscribers ─────────────────────────────────────────────────────
        rospy.Subscriber("/camera/color/image_raw",
                         Image, self.color_cb)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw",
                         Image, self.depth_cb)
        rospy.Subscriber("/camera/color/camera_info",
                         CameraInfo, self.info_cb)

        # ── Publishers ──────────────────────────────────────────────────────
        self.pos_pub = rospy.Publisher("/ball/position", PointStamped,  queue_size=1)
        self.vel_pub = rospy.Publisher("/ball/velocity", TwistStamped,  queue_size=1)
        self.vis_pub = rospy.Publisher("/ball/debug_image", Image,      queue_size=1)

        rospy.loginfo("Ball perception node ready.")
        rospy.spin()

    # ── Callbacks ────────────────────────────────────────────────────────────
    def info_cb(self, msg):
        self.cam_info = msg

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

    def color_cb(self, msg):
        if self.depth_image is None or self.cam_info is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process(frame, msg.header.stamp)

    # ── Detection ────────────────────────────────────────────────────────────
    def detect_ball(self, frame):
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        k    = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask,  k, iterations=1)
        mask = cv2.dilate(mask, k, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        c = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(c)

        if radius < self.min_radius_px:
            return None, mask

        return (int(cx), int(cy), radius), mask

    # ── Depth sampling ───────────────────────────────────────────────────────
    def sample_depth(self, cx, cy):
        patch = self.depth_image[
            max(0, cy-5):cy+5,
            max(0, cx-5):cx+5
        ]
        valid = patch[patch > 0]
        return float(np.median(valid)) if len(valid) > 0 else None

    # ── Backprojection ───────────────────────────────────────────────────────
    def pixel_to_3d(self, cx, cy, depth_mm):
        fx  = self.cam_info.K[0]
        fy  = self.cam_info.K[4]
        ppx = self.cam_info.K[2]
        ppy = self.cam_info.K[5]

        z = depth_mm / 1000.0 + self.ball_radius_m   # correct to sphere centre
        x = (cx - ppx) * z / fx
        y = (cy - ppy) * z / fy
        return np.array([x, y, z])

    # ── TF transform ─────────────────────────────────────────────────────────
    def to_world(self, pos_camera, stamp):
        pt = PointStamped()
        pt.header.stamp    = stamp
        pt.header.frame_id = "camera_color_optical_frame"
        pt.point.x, pt.point.y, pt.point.z = pos_camera

        try:
            return self.tf_buffer.transform(pt, self.target_frame,
                                            rospy.Duration(0.1))
        except Exception as e:
            rospy.logwarn(f"TF failed: {e}")
            return None

    # ── Main pipeline ────────────────────────────────────────────────────────
    def process(self, frame, stamp):
        detection, mask = self.detect_ball(frame)

        # ── Debug visualisation ─────────────────────────────────────────────
        vis = frame.copy()
        if detection:
            cx, cy, r = detection
            cv2.circle(vis, (cx, cy), int(r), (0,255,0), 2)
            cv2.circle(vis, (cx, cy), 3,      (0,0,255), -1)
        self.vis_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

        if detection is None:
            return

        cx, cy, _ = detection

        # ── Depth & 3D position in camera frame ────────────────────────────
        depth_mm = self.sample_depth(cx, cy)
        if depth_mm is None:
            return

        pos_cam = self.pixel_to_3d(cx, cy, depth_mm)

        # ── Transform to world/robot frame ──────────────────────────────────
        pt_world = self.to_world(pos_cam, stamp)
        if pt_world is None:
            return

        pos = np.array([pt_world.point.x,
                        pt_world.point.y,
                        pt_world.point.z])
        
        # ── Kalman filter update ────────────────────────────────────────────
        if not self.kf_init:
            self.kf.x[:3] = pos.reshape(3,1)
            self.kf_init  = True

        self.kf.predict()
        self.kf.update(pos)

        smooth_pos = self.kf.x[:3].flatten()
        smooth_vel = self.kf.x[3:].flatten()

        # ── Publish position ────────────────────────────────────────────────
        pos_msg = PointStamped()
        pos_msg.header.stamp    = stamp
        pos_msg.header.frame_id = self.target_frame
        pos_msg.point.x, pos_msg.point.y, pos_msg.point.z = smooth_pos
        self.pos_pub.publish(pos_msg)

        # ── Publish velocity ────────────────────────────────────────────────
        vel_msg = TwistStamped()
        vel_msg.header.stamp    = stamp
        vel_msg.header.frame_id = self.target_frame
        vel_msg.twist.linear.x  = smooth_vel[0]
        vel_msg.twist.linear.y  = smooth_vel[1]
        vel_msg.twist.linear.z  = smooth_vel[2]
        self.vel_pub.publish(vel_msg)


if __name__ == "__main__":
    BallPerception()