#!/usr/bin/env python3

import base64
import json
import os
import re
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
    import requests
except ImportError as exc:
    requests = None
    REQUESTS_IMPORT_ERROR = str(exc)
else:
    REQUESTS_IMPORT_ERROR = ""


NODE_NAME = "vision_node"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class VisionNode(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)

        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.client_lock = threading.Lock()

        self.camera_topic = self._param("camera_topic", "/usb_cam/image_raw")
        self.camera_frame_topic_out = self._param(
            "camera_frame_topic_out",
            "/chef_robot_assistant/image_annotated"
        )
        self.detection_confidence_threshold = float(
            self._param("detection_confidence_threshold", 0.5)
        )
        self.capture_duration_sec = float(self._param("capture_duration_sec", 3.0))
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
        self.openai_vision_model = str(
            self._param("openai_vision_model", "gpt-5.4")
        ).strip() or "gpt-5.4"
        self.openai_vision_api_timeout_sec = float(
            self._param("openai_vision_api_timeout_sec", 30.0)
        )
        self.openai_vision_image_detail = str(
            self._param("openai_vision_image_detail", "high")
        ).strip().lower() or "high"
        self.openai_vision_max_output_tokens = int(
            self._param("openai_vision_max_output_tokens", 1200)
        )
        self.openai_vision_temperature = float(
            self._param("openai_vision_temperature", 0.0)
        )
        self.openai_vision_jpeg_quality = int(
            self._param("openai_vision_jpeg_quality", 85)
        )
        self.openai_vision_prompt = str(
            self._param("openai_vision_prompt", "")
        ).strip()
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dotenv_path = os.path.join(self.repo_root, ".env")
        self.openai_api_key = ""
        self.latest_frame = None
        self.latest_frame_header = None

        self.latest_frame_stamp = None
        self.latest_frame_error = ""
        self.last_inference_error = ""

        self.client_ready = False
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
            "%s listening on %s and publishing debug frames to %s using OpenAI model %s",
            NODE_NAME,
            self.camera_topic,
            self.camera_frame_topic_out,
            self.openai_vision_model,
        )

    def _param(self, key, default):
        return rospy.get_param("/chef_robot_assistant/{0}".format(key), default)

    def _ensure_client_loaded(self, log_error):
        with self.client_lock:
            if self.client_ready:
                return True

            dependency_error = self._dependency_error_message()
            if dependency_error:
                self.client_error = dependency_error
            else:
                self.openai_api_key, self.client_error = self._read_openai_api_key()
                if self.openai_api_key:
                    self.client_ready = True
                    self.client_error = ""
                    rospy.loginfo(
                        "Configured OpenAI ingredient detector with model %s",
                        self.openai_vision_model,
                    )

            if not self.client_ready and log_error and self.client_error:
                rospy.logerr(self.client_error)

            return self.client_ready

    def _dependency_error_message(self):
        errors = []
        if CV2_IMPORT_ERROR:
            errors.append("opencv-python import failed: {0}".format(CV2_IMPORT_ERROR))
        if NUMPY_IMPORT_ERROR:
            errors.append("numpy import failed: {0}".format(NUMPY_IMPORT_ERROR))
        if REQUESTS_IMPORT_ERROR:
            errors.append("requests import failed: {0}".format(REQUESTS_IMPORT_ERROR))
        return "; ".join(errors)

    def _read_openai_api_key(self):
        env_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if env_key:
            return env_key, ""

        if not os.path.isfile(self.dotenv_path):
            return "", "OPENAI_API_KEY is not set and dotenv file was not found at {0}".format(
                self.dotenv_path
            )

        try:
            dotenv_map = dotenv_values(self.dotenv_path)
        except Exception as exc:
            return "", "failed to read dotenv file {0}: {1}".format(
                self.dotenv_path,
                exc
            )

        api_key = str(dotenv_map.get("OPENAI_API_KEY", "")).strip()
        if not api_key:
            return "", "OPENAI_API_KEY is missing from environment and {0}".format(
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
        if not self._ensure_client_loaded(log_error=False):
            return [], None, self.client_error

        image_data_url, encode_error = self._encode_frame_as_data_url(frame)
        if encode_error:
            return [], None, encode_error

        payload = self._build_openai_payload(frame, image_data_url)
        headers = {
            "Authorization": "Bearer {0}".format(self.openai_api_key),
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                OPENAI_RESPONSES_URL,
                headers=headers,
                json=payload,
                timeout=self.openai_vision_api_timeout_sec,
            )
        except requests.exceptions.Timeout:
            return [], None, "OpenAI vision request timed out"
        except requests.exceptions.RequestException as exc:
            return [], None, "OpenAI vision request failed: {0}".format(exc)

        parsed_response, response_error = self._parse_openai_http_response(response)
        if response_error:
            return [], None, response_error

        detections, parse_error = self._parse_openai_detections(parsed_response, frame)
        if parse_error:
            return [], None, parse_error

        annotated_frame = self._annotate_frame(frame, detections)
        return detections, annotated_frame, ""

    def _encode_frame_as_data_url(self, frame):
        if cv2 is None:
            return "", "opencv-python is unavailable; cannot encode image for OpenAI"

        quality = max(50, min(100, int(self.openai_vision_jpeg_quality)))
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            return "", "failed to encode camera frame as JPEG"

        image_base64 = base64.b64encode(buffer.tobytes()).decode("ascii")
        return "data:image/jpeg;base64,{0}".format(image_base64), ""

    def _build_openai_payload(self, frame, image_data_url):
        height, width = frame.shape[:2]
        prompt = self._build_detection_prompt(width, height)
        text_format = {
            "format": {
                "type": "json_schema",
                "name": "ingredient_detection_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "detections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "class": {
                                        "type": "string",
                                        "description": "Lowercase ingredient name, for example egg, tomato, onion, rice, bread."
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0
                                    },
                                    "x": {
                                        "type": "number",
                                        "description": "Bounding box center x in pixels."
                                    },
                                    "y": {
                                        "type": "number",
                                        "description": "Bounding box center y in pixels."
                                    },
                                    "width": {
                                        "type": "number",
                                        "description": "Bounding box width in pixels."
                                    },
                                    "height": {
                                        "type": "number",
                                        "description": "Bounding box height in pixels."
                                    }
                                },
                                "required": ["class", "confidence", "x", "y", "width", "height"]
                            }
                        }
                    },
                    "required": ["detections"]
                }
            }
        }

        return {
            "model": self.openai_vision_model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url,
                            "detail": self.openai_vision_image_detail,
                        },
                    ],
                }
            ],
            "text": text_format,
            "temperature": self.openai_vision_temperature,
            "max_output_tokens": self.openai_vision_max_output_tokens,
        }

    def _build_detection_prompt(self, image_width, image_height):
        custom_prompt = ""
        if self.openai_vision_prompt:
            custom_prompt = "\nAdditional user guidance: {0}\n".format(
                self.openai_vision_prompt
            )

        return (
            "You are the vision module for a ROS Noetic chef robot. "
            "Detect visible raw food ingredients in the image. "
            "Return only real edible ingredients that are clearly visible. "
            "Do not return plates, bowls, hands, table surfaces, packaging, utensils, background objects, or cooked dish names. "
            "Use simple lowercase singular class names such as egg, tomato, onion, garlic, chicken, fish, rice, bread, carrot, potato, lettuce, cucumber, beef, cheese, milk, lemon, chili, mushroom. "
            "For each ingredient, estimate a bounding box in pixel coordinates for the full image size {0}x{1}. "
            "The fields x and y must be the bounding box center, matching Roboflow's output style. "
            "If you are not confident about an object, omit it instead of guessing. "
            "Return JSON that matches the requested schema exactly."
            "{2}"
        ).format(image_width, image_height, custom_prompt)

    def _parse_openai_http_response(self, response):
        try:
            response_json = response.json()
        except ValueError:
            response_json = None

        if response.status_code >= 400:
            error_message = ""
            if isinstance(response_json, dict):
                error_message = str(
                    response_json.get("error", {}).get("message", "") or ""
                ).strip()
            if not error_message:
                error_message = response.text.strip() or "unknown OpenAI API error"
            return None, "OpenAI API returned {0}: {1}".format(
                response.status_code,
                error_message,
            )

        if not isinstance(response_json, dict):
            return None, "OpenAI API returned a non-JSON response"

        output_text = str(response_json.get("output_text", "") or "").strip()
        if not output_text:
            output_text = self._extract_output_text(response_json)

        if not output_text:
            return None, "OpenAI API returned no output text"

        try:
            return json.loads(output_text), ""
        except ValueError as exc:
            return None, "OpenAI detection output was not valid JSON: {0}; output was: {1}".format(
                exc,
                output_text[:500],
            )

    def _extract_output_text(self, response_json):
        text_parts = []
        for output_item in response_json.get("output", []):
            if not isinstance(output_item, dict):
                continue
            for content_item in output_item.get("content", []):
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in ("output_text", "text"):
                    text = content_item.get("text", "")
                    if text:
                        text_parts.append(text)
        return "\n".join(text_parts).strip()

    def _parse_openai_detections(self, payload, frame):
        if not isinstance(payload, dict):
            return [], "OpenAI detection JSON must be an object"

        raw_detections = payload.get("detections", [])
        if not isinstance(raw_detections, list):
            return [], "OpenAI detection JSON field 'detections' must be a list"

        frame_height, frame_width = frame.shape[:2]
        detections = []
        for raw in raw_detections:
            detection, error = self._normalize_detection(raw, frame_width, frame_height)
            if error:
                rospy.logwarn_throttle(5.0, "Skipping malformed OpenAI detection: %s", error)
                continue
            if detection["confidence"] < self.detection_confidence_threshold:
                continue
            detections.append(detection)

        return detections, ""

    def _normalize_detection(self, raw, frame_width, frame_height):
        if not isinstance(raw, dict):
            return None, "detection is not an object"

        label = self._normalize_label(raw.get("class", ""))
        if not label:
            return None, "missing class label"

        try:
            confidence = float(raw.get("confidence", 0.0))
            x_center = float(raw.get("x", 0.0))
            y_center = float(raw.get("y", 0.0))
            width = float(raw.get("width", 0.0))
            height = float(raw.get("height", 0.0))
        except (TypeError, ValueError) as exc:
            return None, "non-numeric confidence or bounding box: {0}".format(exc)

        if width <= 0.0 or height <= 0.0:
            return None, "non-positive bounding box size"

        confidence = max(0.0, min(1.0, confidence))
        x_center = max(0.0, min(float(frame_width - 1), x_center))
        y_center = max(0.0, min(float(frame_height - 1), y_center))
        width = max(1.0, min(float(frame_width), width))
        height = max(1.0, min(float(frame_height), height))

        x1 = int(round(x_center - (width / 2.0)))
        y1 = int(round(y_center - (height / 2.0)))
        x2 = int(round(x_center + (width / 2.0)))
        y2 = int(round(y_center + (height / 2.0)))

        x1 = max(0, min(frame_width - 1, x1))
        y1 = max(0, min(frame_height - 1, y1))
        x2 = max(0, min(frame_width - 1, x2))
        y2 = max(0, min(frame_height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return None, "empty bounding box after clipping"

        return {
            "label": label,
            "confidence": confidence,
            "xyxy": [x1, y1, x2, y2],
            "roboflow_like": {
                "class": label,
                "confidence": confidence,
                "x": x_center,
                "y": y_center,
                "width": width,
                "height": height,
            },
        }, ""

    def _normalize_label(self, label):
        cleaned = re.sub(r"[^a-z0-9\s_-]", " ", str(label).strip().lower())
        cleaned = cleaned.replace("_", " ").replace("-", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

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
        if not self._ensure_client_loaded(log_error=True):
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
