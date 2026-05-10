#!/usr/bin/env python3

import os

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError as exc:
    cv2 = None
    CV2_IMPORT_ERROR = str(exc)
else:
    CV2_IMPORT_ERROR = ""


NODE_NAME = "webcam_node"


class WebcamNode(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)

        self.bridge = CvBridge()
        self.publisher = rospy.Publisher("/usb_cam/image_raw", Image, queue_size=1)

        self.video_device = self._param("video_device", "/dev/video0")
        self.image_width = int(self._param("image_width", 640))
        self.image_height = int(self._param("image_height", 480))
        self.target_fps = float(self._param("fps", 30.0))
        self.camera_frame_id = self._param("camera_frame_id", "usb_cam")
        self.swap_rb_channels = bool(
            self._param("swap_rb_channels", True)
        )

        self.capture = None

        if CV2_IMPORT_ERROR:
            raise RuntimeError("opencv-python import failed: {0}".format(CV2_IMPORT_ERROR))

        self._open_camera()

        rospy.on_shutdown(self._shutdown)
        rospy.loginfo(
            "%s publishing %sx%s frames from %s to /usb_cam/image_raw",
            NODE_NAME,
            self.image_width,
            self.image_height,
            self.video_device,
        )
        rospy.loginfo(
            "webcam swap_rb_channels=%s",
            self.swap_rb_channels,
        )

    def _param(self, key, default):
        return rospy.get_param("~{0}".format(key), default)

    def _video_capture_source(self):
        basename = os.path.basename(str(self.video_device))
        if basename.startswith("video"):
            suffix = basename[len("video"):]
            if suffix.isdigit():
                return int(suffix)

        device_string = str(self.video_device).strip()
        if device_string.isdigit():
            return int(device_string)

        return device_string

    def _open_camera(self):
        capture_source = self._video_capture_source()
        self.capture = cv2.VideoCapture(capture_source)

        if not self.capture.isOpened():
            raise RuntimeError(
                "failed to open webcam device {0}".format(self.video_device)
            )

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)

    def spin(self):
        rate_hz = self.target_fps if self.target_fps > 0.0 else 30.0
        rate = rospy.Rate(rate_hz)

        while not rospy.is_shutdown():
            ok, frame = self.capture.read()
            if not ok or frame is None:
                rospy.logwarn_throttle(
                    5.0,
                    "failed to read frame from webcam device %s",
                    self.video_device,
                )
                rate.sleep()
                continue

            prepared_frame = self._prepare_frame(frame)
            image_message = self.bridge.cv2_to_imgmsg(prepared_frame, encoding="bgr8")
            image_message.header.stamp = rospy.Time.now()
            image_message.header.frame_id = self.camera_frame_id
            self.publisher.publish(image_message)
            rate.sleep()

    def _prepare_frame(self, frame):
        prepared = frame

        # The VM webcam path needed RB swapping; keep it togglable because the
        # target robot camera may already publish correct color ordering.
        if self.swap_rb_channels:
            prepared = prepared[:, :, ::-1].copy()

        return prepared

    def _shutdown(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def main():
    webcam_node = WebcamNode()
    webcam_node.spin()


if __name__ == "__main__":
    main()
