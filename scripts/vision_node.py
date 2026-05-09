#!/usr/bin/env python3

import os
import threading
from collections import Counter
from collections import deque

import rospy
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from sensor_msgs.msg import Image

from chef_robot_assistant.srv import DetectIngredients
from chef_robot_assistant.srv import DetectIngredientsResponse

try:
    import cv2
except ImportError as exc:
    cv2 = None
    CV2_IMPORT_ERROR = str(exc)
else:
    CV2_IMPORT_ERROR = ""

try:
    import numpy as np
except ImportError as exc:
    np = None
    NUMPY_IMPORT_ERROR = str(exc)
else:
    NUMPY_IMPORT_ERROR = ""

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None
    YOLO_IMPORT_ERROR = str(exc)
else:
    YOLO_IMPORT_ERROR = ""


NODE_NAME = "vision_node"


class VisionNode(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)

        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.model_lock = threading.Lock()

        self.camera_topic = self._param("camera_topic", "/camera/image_raw")
        self.camera_frame_topic_out = self._param(
            "camera_frame_topic_out",
            "/chef_robot_assistant/image_annotated"
        )
        self.detection_confidence_threshold = float(
            self._param("detection_confidence_threshold", 0.5)
        )
        self.capture_duration_sec = float(self._param("capture_duration_sec", 3.0))
        self.vision_model_path = os.path.expanduser(
            self._param("vision_model_path", "")
        )
        self.vision_sample_frame_count = int(
            self._param("vision_sample_frame_count", 10)
        )
        self.vision_min_frame_hits = int(self._param("vision_min_frame_hits", 2))
        self.vision_frame_stale_timeout_sec = float(
            self._param("vision_frame_stale_timeout_sec", 2.0)
        )

        self.latest_frame = None
        self.latest_frame_stamp = None
        self.latest_frame_error = ""
        self.last_inference_error = ""
        self.detection_history = deque(
            maxlen=max(self.vision_sample_frame_count * 6, 60)
        )

        self.model = None
        self.model_error = ""

        self.debug_publisher = rospy.Publisher(
            self.camera_frame_topic_out,
            Image,
            queue_size=1
        )
        self.image_subscriber = rospy.Subscriber(
            self.camera_topic,
            Image,
            self.handle_camera_frame,
            queue_size=1,
            buff_size=2 ** 24
        )
        self.detect_service = rospy.Service(
            "detect_ingredients",
            DetectIngredients,
            self.handle_detect
        )

        self._ensure_model_loaded(log_error=True)

        rospy.loginfo(
            "%s listening on %s and publishing debug frames to %s",
            NODE_NAME,
            self.camera_topic,
            self.camera_frame_topic_out
        )

    def _param(self, key, default):
        return rospy.get_param("/chef_robot_assistant/{0}".format(key), default)

    def _ensure_model_loaded(self, log_error):
        with self.model_lock:
            if self.model is not None:
                return self.model

            dependency_error = self._dependency_error_message()
            if dependency_error:
                self.model_error = dependency_error
            elif not self.vision_model_path:
                self.model_error = "vision_model_path is not set"
            elif not os.path.isfile(self.vision_model_path):
                self.model_error = (
                    "vision model file does not exist: {0}".format(
                        self.vision_model_path
                    )
                )
            else:
                try:
                    self.model = YOLO(self.vision_model_path)
                    self.model_error = ""
                    rospy.loginfo(
                        "Loaded YOLO ingredient model from %s",
                        self.vision_model_path
                    )
                except Exception as exc:
                    self.model = None
                    self.model_error = (
                        "failed to load YOLO model from {0}: {1}".format(
                            self.vision_model_path,
                            exc
                        )
                    )

            if self.model is None and log_error and self.model_error:
                rospy.logerr(self.model_error)

            return self.model

    def _dependency_error_message(self):
        errors = []
        if CV2_IMPORT_ERROR:
            errors.append("opencv-python import failed: {0}".format(CV2_IMPORT_ERROR))
        if NUMPY_IMPORT_ERROR:
            errors.append("numpy import failed: {0}".format(NUMPY_IMPORT_ERROR))
        if YOLO_IMPORT_ERROR:
            errors.append("ultralytics import failed: {0}".format(YOLO_IMPORT_ERROR))
        return "; ".join(errors)

    def handle_camera_frame(self, ros_image):
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="bgr8")
        except CvBridgeError as exc:
            with self.frame_lock:
                self.latest_frame_error = "failed to convert camera frame: {0}".format(
                    exc
                )
            rospy.logwarn_throttle(5.0, self.latest_frame_error)
            return

        stamp_sec = self._header_to_sec(ros_image.header)

        with self.frame_lock:
            self.latest_frame = frame.copy()
            self.latest_frame_stamp = stamp_sec
            self.latest_frame_error = ""

        model = self._ensure_model_loaded(log_error=False)
        if model is None:
            return

        detections, annotated_frame, error_message = self._run_inference(model, frame)
        if error_message:
            with self.frame_lock:
                self.last_inference_error = error_message
            rospy.logwarn_throttle(5.0, error_message)
            return

        with self.frame_lock:
            self.last_inference_error = ""

        self._publish_debug_frame(annotated_frame, ros_image.header)

        frame_labels = sorted({item["label"] for item in detections})
        with self.frame_lock:
            self.detection_history.append({
                "stamp": stamp_sec,
                "labels": frame_labels,
            })

    def _header_to_sec(self, header):
        if header.stamp and header.stamp.to_sec() > 0.0:
            return header.stamp.to_sec()
        return rospy.Time.now().to_sec()

    def _run_inference(self, model, frame):
        try:
            result = model(frame, verbose=False)[0]
        except Exception as exc:
            return [], None, "vision inference failed: {0}".format(exc)

        detections = []
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})

        if boxes is not None:
            xyxy_values = boxes.xyxy.cpu().tolist()
            confidence_values = boxes.conf.cpu().tolist()
            class_values = boxes.cls.cpu().tolist()

            for xyxy, confidence, class_id in zip(
                xyxy_values,
                confidence_values,
                class_values
            ):
                if confidence < self.detection_confidence_threshold:
                    continue

                label = self._class_name_from_id(names, class_id)
                detections.append({
                    "label": label,
                    "confidence": float(confidence),
                    "xyxy": [int(value) for value in xyxy],
                })

        annotated_frame = self._annotate_frame(frame, detections)
        return detections, annotated_frame, ""

    def _class_name_from_id(self, names, class_id):
        class_index = int(class_id)
        if isinstance(names, dict):
            return str(names.get(class_index, class_index)).strip()
        if isinstance(names, list) and class_index < len(names):
            return str(names[class_index]).strip()
        return str(class_index)

    def _annotate_frame(self, frame, detections):
        if cv2 is None or np is None:
            return frame

        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["xyxy"]
            label_text = "{0} {1:.2f}".format(
                detection["label"],
                detection["confidence"]
            )

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                label_text,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        return annotated

    def _publish_debug_frame(self, frame, header):
        try:
            debug_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(
                5.0,
                "failed to publish annotated debug frame: {0}".format(exc)
            )
            return

        debug_image.header = header
        self.debug_publisher.publish(debug_image)

    def handle_detect(self, _request):
        model = self._ensure_model_loaded(log_error=True)
        if model is None:
            return DetectIngredientsResponse(
                success=False,
                ingredients=[],
                message=self.model_error
            )

        fresh_frame, frame_error = self._has_fresh_frame()
        if not fresh_frame:
            return DetectIngredientsResponse(
                success=False,
                ingredients=[],
                message=frame_error
            )

        capture_start = rospy.Time.now().to_sec()
        capture_end = capture_start + self.capture_duration_sec
        sleep_interval = self.capture_duration_sec / max(self.vision_sample_frame_count, 1)

        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < capture_end:
            rospy.sleep(max(sleep_interval, 0.05))

        snapshots = self._window_snapshots(capture_start, capture_end)
        if not snapshots:
            with self.frame_lock:
                inference_error = self.last_inference_error

            return DetectIngredientsResponse(
                success=False,
                ingredients=[],
                message=inference_error or (
                    "no detections were captured during the {0:.1f}s window".format(
                        self.capture_duration_sec
                    )
                )
            )

        sampled_snapshots = snapshots[-self.vision_sample_frame_count:]
        label_counts = Counter()
        for snapshot in sampled_snapshots:
            label_counts.update(snapshot["labels"])

        stable_items = [
            item for item in label_counts.items()
            if item[1] >= self.vision_min_frame_hits
        ]
        stable_items.sort(key=lambda item: (-item[1], item[0]))
        ingredients = [item[0] for item in stable_items]

        if not ingredients:
            return DetectIngredientsResponse(
                success=False,
                ingredients=[],
                message=(
                    "no stable ingredient detections found across {0} sampled frames".format(
                        len(sampled_snapshots)
                    )
                )
            )

        return DetectIngredientsResponse(
            success=True,
            ingredients=ingredients,
            message=(
                "detected {0} stable ingredients across {1} sampled frames".format(
                    len(ingredients),
                    len(sampled_snapshots)
                )
            )
        )

    def _has_fresh_frame(self):
        with self.frame_lock:
            latest_frame_stamp = self.latest_frame_stamp
            latest_frame_error = self.latest_frame_error

        if latest_frame_error:
            return False, latest_frame_error

        if latest_frame_stamp is None:
            return False, "no camera frames received on {0}".format(self.camera_topic)

        frame_age = rospy.Time.now().to_sec() - latest_frame_stamp
        if frame_age > self.vision_frame_stale_timeout_sec:
            return False, (
                "latest camera frame is stale ({0:.2f}s old)".format(frame_age)
            )

        return True, ""

    def _window_snapshots(self, capture_start, capture_end):
        with self.frame_lock:
            snapshots = list(self.detection_history)

        return [
            snapshot for snapshot in snapshots
            if capture_start <= snapshot["stamp"] <= capture_end
        ]


def main():
    VisionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
