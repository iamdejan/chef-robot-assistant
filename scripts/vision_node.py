#!/usr/bin/env python3

import os
import threading
from collections import Counter

import rospy
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from dotenv import dotenv_values
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
    from inference_sdk import InferenceHTTPClient
except ImportError as exc:
    InferenceHTTPClient = None
    ROBOFLOW_IMPORT_ERROR = str(exc)
else:
    ROBOFLOW_IMPORT_ERROR = ""


NODE_NAME = "vision_node"


class VisionNode(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)

        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.client_lock = threading.Lock()

        self.camera_topic = self._param("camera_topic", "/camera/image_raw")
        self.camera_frame_topic_out = self._param(
            "camera_frame_topic_out",
            "/chef_robot_assistant/image_annotated"
        )
        self.detection_confidence_threshold = float(
            self._param("detection_confidence_threshold", 0.5)
        )
        self.capture_duration_sec = float(self._param("capture_duration_sec", 3.0))
        self.vision_workspace_name = self._param(
            "vision_workspace_name",
            "lee-zhi-yang"
        )
        self.vision_workflow_id = self._param(
            "vision_workflow_id",
            "ingredient-detection-api"
        )
        self.vision_sample_frame_count = int(
            self._param("vision_sample_frame_count", 10)
        )
        self.vision_min_frame_hits = max(
            1,
            int(self._param("vision_min_frame_hits", 2))
        )
        self.vision_frame_stale_timeout_sec = float(
            self._param("vision_frame_stale_timeout_sec", 2.0)
        )
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dotenv_path = os.path.join(self.repo_root, ".env")
        self.roboflow_api_key = ""
        self.latest_frame = None
        self.latest_frame_header = None

        self.latest_frame_stamp = None
        self.latest_frame_error = ""
        self.last_inference_error = ""

        self.client = None
        self.client_error = ""

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

        self._ensure_client_loaded(log_error=True)

        rospy.loginfo(
            "%s listening on %s and publishing debug frames to %s",
            NODE_NAME,
            self.camera_topic,
            self.camera_frame_topic_out
        )

    def _param(self, key, default):
        return rospy.get_param("/chef_robot_assistant/{0}".format(key), default)

    def _ensure_client_loaded(self, log_error):
        with self.client_lock:
            if self.client is not None:
                return self.client

            dependency_error = self._dependency_error_message()
            if dependency_error:
                self.client_error = dependency_error
            elif not self.vision_workspace_name:
                self.client_error = "vision_workspace_name is not set"
            elif not self.vision_workflow_id:
                self.client_error = "vision_workflow_id is not set"
            else:
                self.roboflow_api_key, self.client_error = self._read_roboflow_api_key()
                if self.roboflow_api_key:
                    try:
                        self.client = InferenceHTTPClient(
                            api_url="https://serverless.roboflow.com",
                            api_key=self.roboflow_api_key
                        )
                        self.client_error = ""
                        rospy.loginfo(
                            "Configured Roboflow workflow client for %s/%s",
                            self.vision_workspace_name,
                            self.vision_workflow_id
                        )
                    except Exception as exc:
                        self.client = None
                        self.client_error = (
                            "failed to initialize Roboflow workflow client for {0}/{1}: {2}".format(
                                self.vision_workspace_name,
                                self.vision_workflow_id,
                                exc
                            )
                        )

            if self.client is None and log_error and self.client_error:
                rospy.logerr(self.client_error)

            return self.client

    def _dependency_error_message(self):
        errors = []
        if CV2_IMPORT_ERROR:
            errors.append("opencv-python import failed: {0}".format(CV2_IMPORT_ERROR))
        if NUMPY_IMPORT_ERROR:
            errors.append("numpy import failed: {0}".format(NUMPY_IMPORT_ERROR))
        if ROBOFLOW_IMPORT_ERROR:
            errors.append(
                "roboflow inference-sdk import failed: {0}".format(ROBOFLOW_IMPORT_ERROR)
            )
        return "; ".join(errors)

    def _read_roboflow_api_key(self):
        if not os.path.isfile(self.dotenv_path):
            return "", "dotenv file not found at {0}".format(self.dotenv_path)

        try:
            dotenv_map = dotenv_values(self.dotenv_path)
        except Exception as exc:
            return "", "failed to read dotenv file {0}: {1}".format(
                self.dotenv_path,
                exc
            )

        api_key = str(dotenv_map.get("ROBOFLOW_API_KEY", "")).strip()
        if not api_key:
            return "", "ROBOFLOW_API_KEY is missing from {0}".format(
                self.dotenv_path
            )

        return api_key, ""

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
            self.latest_frame_header = ros_image.header
            self.latest_frame_stamp = stamp_sec
            self.latest_frame_error = ""

    def _header_to_sec(self, header):
        if header.stamp and header.stamp.to_sec() > 0.0:
            return header.stamp.to_sec()
        return rospy.Time.now().to_sec()

    def _run_inference(self, frame):
        client = self._ensure_client_loaded(log_error=False)
        if client is None:
            return [], None, self.client_error

        try:
            result = client.run_workflow(
                workspace_name=self.vision_workspace_name,
                workflow_id=self.vision_workflow_id,
                images={"image": frame},
                use_cache=True
            )
        except Exception as exc:
            return [], None, "workflow inference failed for {0}/{1}: {2}".format(
                self.vision_workspace_name,
                self.vision_workflow_id,
                exc
            )

        predictions = self._extract_workflow_predictions(result)
        detections = []

        for prediction in predictions:
            confidence = float(prediction.get("confidence", 0.0))
            if confidence < self.detection_confidence_threshold:
                continue

            x_center = float(prediction.get("x", 0.0))
            y_center = float(prediction.get("y", 0.0))
            width = float(prediction.get("width", 0.0))
            height = float(prediction.get("height", 0.0))

            x1 = int(x_center - (width / 2.0))
            y1 = int(y_center - (height / 2.0))
            x2 = int(x_center + (width / 2.0))
            y2 = int(y_center + (height / 2.0))

            label = str(prediction.get("class", "")).strip().lower()
            if not label:
                continue

            detections.append({
                "label": label,
                "confidence": confidence,
                "xyxy": [x1, y1, x2, y2],
            })

        annotated_frame = self._annotate_frame(frame, detections)
        return detections, annotated_frame, ""

    def _extract_workflow_predictions(self, payload):
        direct_predictions = self._coerce_predictions(payload)
        if direct_predictions:
            return direct_predictions

        queue = [payload]
        while queue:
            current = queue.pop(0)

            if isinstance(current, dict):
                nested_predictions = self._coerce_predictions(current)
                if nested_predictions:
                    return nested_predictions
                queue.extend(current.values())
            elif isinstance(current, list):
                queue.extend(current)

        return []

    def _coerce_predictions(self, payload):
        if not isinstance(payload, dict):
            return []

        predictions = payload.get("predictions", [])
        if not isinstance(predictions, list):
            return []

        normalized = []
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            if self._looks_like_detection_prediction(prediction):
                normalized.append(prediction)

        return normalized

    def _looks_like_detection_prediction(self, prediction):
        required_keys = ("class", "confidence", "x", "y", "width", "height")
        return all(key in prediction for key in required_keys)

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
        client = self._ensure_client_loaded(log_error=True)
        if client is None:
            return DetectIngredientsResponse(
                success=False,
                ingredients=[],
                message=self.client_error
            )

        sample_count = max(self.vision_sample_frame_count, 1)
        frame_wait_timeout_sec = max(
            self.capture_duration_sec / float(sample_count),
            0.5
        )
        sampled_snapshots = []
        previous_stamp = None

        for sample_index in range(sample_count):
            if rospy.is_shutdown():
                break

            frame_ok, frame_or_error = self._wait_for_next_frame(
                previous_stamp,
                frame_wait_timeout_sec
            )
            if not frame_ok:
                if not sampled_snapshots:
                    return DetectIngredientsResponse(
                        success=False,
                        ingredients=[],
                        message=frame_or_error
                    )
                rospy.logwarn(
                    "Stopping ingredient sampling early after %d/%d frames: %s",
                    len(sampled_snapshots),
                    sample_count,
                    frame_or_error
                )
                break

            frame, header, frame_stamp = frame_or_error
            previous_stamp = frame_stamp
            detections, annotated_frame, error_message = self._run_inference(frame)
            if error_message:
                with self.frame_lock:
                    self.last_inference_error = error_message
                rospy.logwarn_throttle(5.0, error_message)
            else:
                with self.frame_lock:
                    self.last_inference_error = ""
                self._publish_debug_frame(annotated_frame, header)
                labels = sorted({item["label"] for item in detections})
                rospy.loginfo(
                    "Vision sample %d/%d captured %d detections above %.2f: %s",
                    sample_index + 1,
                    sample_count,
                    len(detections),
                    self.detection_confidence_threshold,
                    ", ".join(labels) if labels else "none"
                )
                sampled_snapshots.append({
                    "labels": labels
                })

        if not sampled_snapshots:
            with self.frame_lock:
                inference_error = self.last_inference_error or self.client_error

            return DetectIngredientsResponse(
                success=False,
                ingredients=[],
                message=inference_error or (
                    "no detections were captured during the {0:.1f}s window".format(
                        self.capture_duration_sec
                    )
                )
            )

        required_hits = min(self.vision_min_frame_hits, sample_count)
        label_counts = Counter()
        for snapshot in sampled_snapshots:
            label_counts.update(snapshot["labels"])

        stable_items = [
            item for item in label_counts.items()
            if item[1] >= required_hits
        ]
        stable_items.sort(key=lambda item: (-item[1], item[0]))
        ingredients = [item[0] for item in stable_items]

        if not ingredients:
            return DetectIngredientsResponse(
                success=False,
                ingredients=[],
                message=(
                    "no stable ingredient detections found across {0} sampled frames "
                    "(required hits: {1}, label hits: {2})".format(
                        len(sampled_snapshots),
                        required_hits,
                        self._format_label_counts(label_counts)
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

    def _wait_for_next_frame(self, previous_stamp, timeout_sec):
        deadline = rospy.Time.now().to_sec() + max(timeout_sec, 0.0)
        last_error = "no camera frames received on {0}".format(self.camera_topic)

        while not rospy.is_shutdown():
            frame_ok, frame_or_error = self._get_fresh_frame()
            if frame_ok:
                frame, header, stamp = frame_or_error
                if previous_stamp is None or stamp > previous_stamp:
                    return True, (frame, header, stamp)
                last_error = (
                    "camera frames on {0} are not updating".format(self.camera_topic)
                )
            else:
                last_error = frame_or_error

            if rospy.Time.now().to_sec() >= deadline:
                break

            rospy.sleep(0.02)

        return False, last_error

    def _format_label_counts(self, label_counts):
        if not label_counts:
            return "none"

        ordered_items = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
        return ", ".join(
            "{0}:{1}".format(label, count) for label, count in ordered_items
        )

    def _get_fresh_frame(self):
        with self.frame_lock:
            latest_frame = None if self.latest_frame is None else self.latest_frame.copy()
            latest_frame_header = self.latest_frame_header
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

        if latest_frame is None or latest_frame_header is None:
            return False, "camera frame buffer is empty on {0}".format(self.camera_topic)

        return True, (latest_frame, latest_frame_header, latest_frame_stamp)


def main():
    VisionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
