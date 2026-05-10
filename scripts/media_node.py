#!/usr/bin/env python3

import os
import shutil
import subprocess
import textwrap
import time

import numpy as np
import rospy

from chef_robot_assistant.srv import GenerateDishImage
from chef_robot_assistant.srv import GenerateDishImageResponse
from chef_robot_assistant.srv import SpeakText
from chef_robot_assistant.srv import SpeakTextResponse

try:
    import cv2
except ImportError as exc:
    cv2 = None
    CV2_IMPORT_ERROR = str(exc)
else:
    CV2_IMPORT_ERROR = ""

try:
    from huggingface_hub import InferenceClient
except ImportError as exc:
    InferenceClient = None
    HF_HUB_IMPORT_ERROR = str(exc)
else:
    HF_HUB_IMPORT_ERROR = ""

try:
    from dotenv import dotenv_values
except ImportError as exc:
    dotenv_values = None
    DOTENV_IMPORT_ERROR = str(exc)
else:
    DOTENV_IMPORT_ERROR = ""


NODE_NAME = "media_node"
class MediaNode(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)

        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dotenv_path = os.path.join(self.repo_root, ".env")

        self.tts_voice = str(self._param("tts_voice", "en")).strip() or "en"
        self.tts_speed_wpm = int(self._param("tts_speed_wpm", 165))
        self.display_window_enabled = bool(
            self._param("display_window_enabled", True)
        )
        self.recipe_window_name = str(
            self._param("display_recipe_window_name", "Chef Robot Recipe")
        ).strip() or "Chef Robot Recipe"
        self.image_window_name = str(
            self._param("display_image_window_name", "Chef Robot Dish Image")
        ).strip() or "Chef Robot Dish Image"
        self.display_window_width = int(self._param("display_window_width", 1280))
        self.display_window_height = int(self._param("display_window_height", 720))
        self.image_model_name = str(
            self._param("image_model_name", "Tongyi-MAI/Z-Image-Turbo")
        ).strip() or "Tongyi-MAI/Z-Image-Turbo"
        self.image_api_timeout_sec = float(
            self._param("image_api_timeout_sec", 60.0)
        )
        self.image_output_dir = str(
            self._param(
                "image_output_dir",
                os.path.join(self.repo_root, "generated_media"),
            )
        ).strip() or os.path.join(self.repo_root, "generated_media")
        self.image_width = int(self._param("image_width", 1024))
        self.image_height = int(self._param("image_height", 1024))

        self.espeak_path = shutil.which("espeak")
        self.current_recipe_data = None
        self.current_spoken_text = ""
        self.current_image = None
        self.current_recipe_is_stale = False
        self.recipe_window_dirty = False
        self.image_window_dirty = False

        self._initialize_display_support()
        os.makedirs(self.image_output_dir, exist_ok=True)

        rospy.Service("generate_dish_image", GenerateDishImage, self.handle_image)
        rospy.Service("speak_text", SpeakText, self.handle_speak)
        rospy.loginfo(
            "%s ready with tts_voice=%s image_model=%s display_enabled=%s",
            NODE_NAME,
            self.tts_voice,
            self.image_model_name,
            self.display_window_enabled,
        )

    def _param(self, key, default):
        return rospy.get_param("/chef_robot_assistant/{0}".format(key), default)

    def _initialize_display_support(self):
        if not self.display_window_enabled:
            return

        if CV2_IMPORT_ERROR:
            rospy.logwarn("Display disabled because cv2 is unavailable: %s", CV2_IMPORT_ERROR)
            self.display_window_enabled = False
            return

        if not os.environ.get("DISPLAY"):
            rospy.logwarn("Display disabled because DISPLAY is not set.")
            self.display_window_enabled = False
            return

        try:
            cv2.namedWindow(self.recipe_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.recipe_window_name,
                self.display_window_width,
                self.display_window_height,
            )
        except cv2.error as exc:
            rospy.logwarn("Display disabled because OpenCV window init failed: %s", exc)
            self.display_window_enabled = False

    def _dependency_error_message(self):
        errors = []
        if HF_HUB_IMPORT_ERROR:
            errors.append(
                "huggingface_hub import failed: {0}".format(HF_HUB_IMPORT_ERROR)
            )
        if DOTENV_IMPORT_ERROR:
            errors.append("python-dotenv import failed: {0}".format(DOTENV_IMPORT_ERROR))
        if CV2_IMPORT_ERROR:
            errors.append("opencv-python import failed: {0}".format(CV2_IMPORT_ERROR))
        return "; ".join(errors)

    def _read_hf_token(self):
        dependency_error = self._dependency_error_message()
        if dependency_error:
            return "", dependency_error

        if not os.path.isfile(self.dotenv_path):
            return "", "dotenv file not found at {0}".format(self.dotenv_path)

        try:
            dotenv_map = dotenv_values(self.dotenv_path)
        except Exception as exc:
            return "", "failed to read dotenv file {0}: {1}".format(
                self.dotenv_path,
                exc,
            )

        token = str(dotenv_map.get("HF_TOKEN", "")).strip()
        if not token:
            return "", "HF_TOKEN is missing from {0}".format(self.dotenv_path)
        return token, ""

    def _load_recipe_display_data(self):
        base_key = "/chef_robot_assistant/last_recipe"
        dish_name = str(rospy.get_param(base_key + "/dish_name", "")).strip()
        spoken_summary = str(
            rospy.get_param(base_key + "/spoken_summary", "")
        ).strip()
        full_recipe_text = str(
            rospy.get_param(base_key + "/full_recipe_text", "")
        ).strip()
        missing_ingredients = rospy.get_param(base_key + "/missing_ingredients", [])
        if not isinstance(missing_ingredients, list):
            missing_ingredients = []
        missing_ingredients = [
            str(item).strip() for item in missing_ingredients if str(item).strip()
        ]

        if not any((dish_name, spoken_summary, full_recipe_text, missing_ingredients)):
            return None

        return {
            "dish_name": dish_name,
            "spoken_summary": spoken_summary,
            "full_recipe_text": full_recipe_text,
            "missing_ingredients": missing_ingredients,
        }

    def _wrap_text(self, text, width_chars):
        lines = []
        for raw_line in str(text).splitlines() or [""]:
            wrapped = textwrap.wrap(
                raw_line.strip(),
                width=max(20, width_chars),
                break_long_words=False,
                break_on_hyphens=False,
            )
            if wrapped:
                lines.extend(wrapped)
            else:
                lines.append("")
        return lines

    def _render_recipe_canvas(self, spoken_text):
        canvas = np.full(
            (self.display_window_height, self.display_window_width, 3),
            245,
            dtype=np.uint8,
        )
        color = (30, 30, 30)
        accent = (30, 90, 180)

        recipe_data = self._load_recipe_display_data()
        if recipe_data is not None:
            self.current_recipe_data = recipe_data
            self.current_recipe_is_stale = False
        else:
            recipe_data = self.current_recipe_data
            self.current_recipe_is_stale = recipe_data is not None

        y = 50
        margin = 40
        max_width_chars = max(50, int((self.display_window_width - margin * 2) / 16))

        title = "Chef Robot Assistant"
        if recipe_data and recipe_data.get("dish_name"):
            title = recipe_data["dish_name"]

        if spoken_text:
            cv2.putText(
                canvas,
                "Robot Says",
                (margin, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                accent,
                2,
                cv2.LINE_AA,
            )
            y += 34
            for line in self._wrap_text(spoken_text, max_width_chars):
                cv2.putText(
                    canvas,
                    line,
                    (margin, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                y += 28
            y += 20

        if self.current_recipe_is_stale and recipe_data:
            cv2.putText(
                canvas,
                "Previous Dish and Recipe",
                (margin, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 180),
                2,
                cv2.LINE_AA,
            )
            y += 36

        title_label = "Dish Name"
        if self.current_recipe_is_stale and recipe_data:
            title_label = "Previous Dish Name"

        cv2.putText(
            canvas,
            title_label,
            (margin, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            accent,
            2,
            cv2.LINE_AA,
        )
        y += 34

        cv2.putText(
            canvas,
            title,
            (margin, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            accent,
            2,
            cv2.LINE_AA,
        )
        y += 45

        if recipe_data and recipe_data.get("missing_ingredients"):
            missing_text = "Need to buy: {0}".format(
                ", ".join(recipe_data["missing_ingredients"])
            )
            for line in self._wrap_text(missing_text, max_width_chars):
                cv2.putText(
                    canvas,
                    line,
                    (margin, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 180),
                    2,
                    cv2.LINE_AA,
                )
                y += 28
            y += 10

        if recipe_data and recipe_data.get("full_recipe_text"):
            recipe_label = "Recipe"
            if self.current_recipe_is_stale:
                recipe_label = "Previous Recipe"
            cv2.putText(
                canvas,
                recipe_label,
                (margin, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                accent,
                2,
                cv2.LINE_AA,
            )
            y += 36
            for line in self._wrap_text(recipe_data["full_recipe_text"], max_width_chars):
                if y > self.display_window_height - 20:
                    break
                cv2.putText(
                    canvas,
                    line,
                    (margin, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                y += 24

        return canvas

    def _refresh_recipe_window(self):
        if not self.display_window_enabled:
            return

        canvas = self._render_recipe_canvas(self.current_spoken_text)
        try:
            cv2.imshow(self.recipe_window_name, canvas)
            cv2.waitKey(1)
        except cv2.error as exc:
            rospy.logwarn("Disabling display after recipe window failure: %s", exc)
            self.display_window_enabled = False

    def _refresh_image_window(self):
        if not self.display_window_enabled or self.current_image is None:
            return

        try:
            cv2.namedWindow(self.image_window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(self.image_window_name, self.current_image)
            cv2.waitKey(1)
        except cv2.error as exc:
            rospy.logwarn("Failed to show generated image window: %s", exc)

    def _pump_windows(self):
        if not self.display_window_enabled:
            return

        if self.recipe_window_dirty:
            self._refresh_recipe_window()
            self.recipe_window_dirty = False

        if self.image_window_dirty:
            self._refresh_image_window()
            self.image_window_dirty = False

        try:
            cv2.waitKey(1)
        except cv2.error:
            pass

    def _speak_text(self, text):
        if not self.espeak_path:
            raise RuntimeError("espeak is not available on this machine.")

        command = [
            self.espeak_path,
            "-v",
            self.tts_voice,
            "-s",
            str(self.tts_speed_wpm),
            text,
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(10.0, len(text) / 8.0),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("text-to-speech playback timed out") from exc
        except subprocess.CalledProcessError as exc:
            error_output = exc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(error_output or "text-to-speech playback failed") from exc

    def _build_image_prompt(self, dish_name, description):
        return (
            "A realistic plated food photograph of {0}. "
            "{1} "
            "appetizing, natural lighting, detailed meal presentation, restaurant style."
        ).format(dish_name.strip(), description.strip())

    def _request_image_bytes(self, token, prompt):
        dependency_error = self._dependency_error_message()
        if dependency_error:
            return b"", dependency_error

        try:
            client = InferenceClient(
                provider="auto",
                api_key=token,
                timeout=self.image_api_timeout_sec,
            )
            image = client.text_to_image(
                prompt,
                model=self.image_model_name,
                width=self.image_width,
                height=self.image_height,
            )
        except Exception as exc:
            return b"", "Hugging Face image request failed: {0}".format(exc)

        if image is None:
            return b"", "Hugging Face image request returned no image"

        try:
            import io

            image_buffer = io.BytesIO()
            image.save(image_buffer, format="PNG")
        except Exception as exc:
            return b"", "failed to encode generated image: {0}".format(exc)

        image_bytes = image_buffer.getvalue()
        if not image_bytes:
            return b"", "Hugging Face image request returned an empty image payload"

        return image_bytes, ""

    def _decode_image_bytes(self, image_bytes):
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("generated image payload could not be decoded")
        return image

    def _save_generated_image(self, image_bytes, dish_name):
        safe_name = "".join(
            char if char.isalnum() else "_" for char in dish_name.strip().lower()
        ).strip("_") or "dish"
        file_name = "{0}_{1}.png".format(safe_name, int(time.time()))
        output_path = os.path.join(self.image_output_dir, file_name)
        with open(output_path, "wb") as output_file:
            output_file.write(image_bytes)
        return output_path

    def handle_image(self, request):
        dish_name = str(request.dish_name).strip()
        description = str(request.description).strip()
        rospy.loginfo(
            "Image request for dish='%s' description='%s'",
            dish_name,
            description,
        )

        if not dish_name:
            return GenerateDishImageResponse(
                success=False,
                image_path="",
                message="dish name is required for image generation",
            )

        token, token_error = self._read_hf_token()
        if token_error:
            return GenerateDishImageResponse(
                success=False,
                image_path="",
                message=token_error,
            )

        prompt = self._build_image_prompt(dish_name, description)
        image_bytes, image_error = self._request_image_bytes(token, prompt)
        if image_error:
            return GenerateDishImageResponse(
                success=False,
                image_path="",
                message=image_error,
            )

        try:
            image = self._decode_image_bytes(image_bytes)
            output_path = self._save_generated_image(image_bytes, dish_name)
        except Exception as exc:
            rospy.logwarn("Failed to decode or save generated image: %s", exc)
            return GenerateDishImageResponse(
                success=False,
                image_path="",
                message=str(exc),
            )

        self.current_image = image
        self.image_window_dirty = True
        return GenerateDishImageResponse(
            success=True,
            image_path=output_path,
            message="dish image generation succeeded",
        )

    def handle_speak(self, request):
        text = str(request.text).strip()
        rospy.loginfo("Speech output request: %s", text)
        self.current_spoken_text = text
        self.recipe_window_dirty = True

        try:
            self._speak_text(text)
        except RuntimeError as exc:
            rospy.logwarn("Speech output failed: %s", exc)
            return SpeakTextResponse(
                success=False,
                message=str(exc),
            )

        return SpeakTextResponse(
            success=True,
            message="speech output succeeded",
        )


def main():
    node = MediaNode()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        node._pump_windows()
        rate.sleep()


if __name__ == "__main__":
    main()
