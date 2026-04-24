#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, TwistStamped
from std_msgs.msg import Float64
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


class BallPerception(Node):
    def __init__(self):
        super().__init__("ball_perception")

        # ── Parameters (set these) ──────────────────────────────────────────
        self.lower_hsv     = np.array([18, 83,  138])   # from hsv_tuner.py
        self.upper_hsv     = np.array([49, 241, 255])   # from hsv_tuner.py
        self.ball_radius_m = 0.012                        # real radius in metres
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
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Subscribers ─────────────────────────────────────────────────────
        self.create_subscription(Image,      "/camera/camera/color/image_raw",
                                 self.color_cb, 10)
        self.create_subscription(Image,      "/camera/camera/aligned_depth_to_color/image_raw",
                                 self.depth_cb, 10)
        self.create_subscription(CameraInfo, "/camera/camera/color/camera_info",
                                 self.info_cb,  10)

        # ── Publishers ──────────────────────────────────────────────────────
        self.pos_pub  = self.create_publisher(PointStamped, "/ball/position",    1)
        self.vel_pub  = self.create_publisher(TwistStamped, "/ball/velocity",    1)
        self.vis_pub  = self.create_publisher(Image,        "/ball/debug_image", 1)
        self.dist_pub = self.create_publisher(Float64,      "/ball/distance",    1)

        self.get_logger().info("Ball perception node ready.")

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
        fx  = self.cam_info.k[0]
        fy  = self.cam_info.k[4]
        ppx = self.cam_info.k[2]
        ppy = self.cam_info.k[5]

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
                                            rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f"TF failed: {e}")
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
            depth_mm_preview = self.sample_depth(cx, cy)
            if depth_mm_preview is not None:
                dist_m = depth_mm_preview / 1000.0
                cv2.putText(vis, f"{dist_m:.2f} m", (cx + int(r) + 5, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
        vel_msg.twist.linear.x  = float(smooth_vel[0])
        vel_msg.twist.linear.y  = float(smooth_vel[1])
        vel_msg.twist.linear.z  = float(smooth_vel[2])
        self.vel_pub.publish(vel_msg)

        # ── Publish distance from camera origin ─────────────────────────────
        dist_msg = Float64()
        dist_msg.data = float(np.linalg.norm(smooth_pos))
        self.dist_pub.publish(dist_msg)


if __name__ == "__main__":
    rclpy.init()
    node = BallPerception()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
