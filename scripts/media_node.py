#!/usr/bin/env python3

import io
import os
import shutil
import subprocess
import time

import numpy as np
import rospy

from chef_robot_assistant.srv import GenerateDishImage
from chef_robot_assistant.srv import GenerateDishImageResponse
from chef_robot_assistant.srv import SpeakText
from chef_robot_assistant.srv import SpeakTextResponse

try:
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
except ImportError as exc:
    tk = None
    ttk = None
    Image = None
    ImageTk = None
    TK_IMPORT_ERROR = str(exc)
else:
    TK_IMPORT_ERROR = ""

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

        # Tkinter UI attributes
        self.root = None
        self.recipe_window = None
        self.image_window = None
        self.recipe_text_widget = None
        self.image_label = None
        self.current_photo_image = None  # Keep reference to prevent garbage collection

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

        if TK_IMPORT_ERROR:
            rospy.logwarn("Display disabled because tkinter/PIL is unavailable: %s", TK_IMPORT_ERROR)
            self.display_window_enabled = False
            return

        if not os.environ.get("DISPLAY"):
            rospy.logwarn("Display disabled because DISPLAY is not set.")
            self.display_window_enabled = False
            return

        try:
            self.root = tk.Tk()
            self.root.withdraw()  # Hide the root window

            # Create recipe window
            self.recipe_window = tk.Toplevel(self.root)
            self.recipe_window.title(self.recipe_window_name)
            self.recipe_window.geometry(f"{self.display_window_width}x{self.display_window_height}")
            self.recipe_window.protocol("WM_DELETE_WINDOW", self._on_recipe_window_close)

            # Create a frame with scrollbar for recipe content
            recipe_frame = ttk.Frame(self.recipe_window)
            recipe_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            self.recipe_text_widget = tk.Text(
                recipe_frame,
                wrap=tk.WORD,
                font=("DejaVu Sans", 11),
                bg="#f5f5f5",
                fg="#1e1e1e",
                padx=10,
                pady=10,
            )
            self.recipe_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            recipe_scrollbar = ttk.Scrollbar(recipe_frame, orient=tk.VERTICAL, command=self.recipe_text_widget.yview)
            recipe_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.recipe_text_widget.config(yscrollcommand=recipe_scrollbar.set)

            # Configure text tags for styling
            self.recipe_text_widget.tag_configure("title", font=("DejaVu Sans", 16, "bold"), foreground="#1e5eb4")
            self.recipe_text_widget.tag_configure("section", font=("DejaVu Sans", 12, "bold"), foreground="#1e5eb4")
            self.recipe_text_widget.tag_configure("body", font=("DejaVu Sans", 11), foreground="#1e1e1e")
            self.recipe_text_widget.tag_configure("accent", font=("DejaVu Sans", 11, "bold"), foreground="#1e5eb4")
            self.recipe_text_widget.tag_configure("warning", font=("DejaVu Sans", 11), foreground="#b40000")
            self.recipe_text_widget.tag_configure("stale", font=("DejaVu Sans", 10, "italic"), foreground="#888888")

            # Create image window
            self.image_window = tk.Toplevel(self.root)
            self.image_window.title(self.image_window_name)
            self.image_window.geometry(f"{self.display_window_width}x{self.display_window_height}")
            self.image_window.protocol("WM_DELETE_WINDOW", self._on_image_window_close)

            self.image_label = ttk.Label(self.image_window)
            self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

            # Initial update to show windows
            self.root.update_idletasks()

        except Exception as exc:
            rospy.logwarn("Display disabled because tkinter window init failed: %s", exc)
            self.display_window_enabled = False

    def _on_recipe_window_close(self):
        self.recipe_window.withdraw()

    def _on_image_window_close(self):
        self.image_window.withdraw()

    def _dependency_error_message(self):
        errors = []
        if HF_HUB_IMPORT_ERROR:
            errors.append(
                "huggingface_hub import failed: {0}".format(HF_HUB_IMPORT_ERROR)
            )
        if DOTENV_IMPORT_ERROR:
            errors.append("python-dotenv import failed: {0}".format(DOTENV_IMPORT_ERROR))
        if TK_IMPORT_ERROR:
            errors.append("tkinter/PIL import failed: {0}".format(TK_IMPORT_ERROR))
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

    def _render_recipe_content(self, spoken_text):
        recipe_data = self._load_recipe_display_data()
        if recipe_data is not None:
            self.current_recipe_data = recipe_data
            self.current_recipe_is_stale = False
        else:
            recipe_data = self.current_recipe_data
            self.current_recipe_is_stale = recipe_data is not None

        # Build content for tkinter Text widget
        self.recipe_text_widget.config(state=tk.NORMAL)
        self.recipe_text_widget.delete(1.0, tk.END)

        # App title
        self.recipe_text_widget.insert(tk.END, "Chef Robot Assistant\n\n", "title")

        # Spoken text
        if spoken_text:
            self.recipe_text_widget.insert(tk.END, "Robot Says\n", "section")
            self.recipe_text_widget.insert(tk.END, spoken_text + "\n\n", "body")

        if self.current_recipe_is_stale and recipe_data:
            self.recipe_text_widget.insert(tk.END, "Previous Dish and Recipe\n", "warning")
            self.recipe_text_widget.insert(tk.END, "\n")

        title_label = "Dish Name"
        if self.current_recipe_is_stale and recipe_data:
            title_label = "Previous Dish Name"

        self.recipe_text_widget.insert(tk.END, f"{title_label}\n", "section")

        dish_title = "No dish generated yet"
        if recipe_data and recipe_data.get("dish_name"):
            dish_title = recipe_data["dish_name"]

        self.recipe_text_widget.insert(tk.END, f"{dish_title}\n\n", "accent")

        if recipe_data and recipe_data.get("missing_ingredients"):
            missing_text = f"Need to buy: {', '.join(recipe_data['missing_ingredients'])}"
            self.recipe_text_widget.insert(tk.END, f"{missing_text}\n\n", "warning")

        if recipe_data and recipe_data.get("full_recipe_text"):
            recipe_label = "Recipe"
            if self.current_recipe_is_stale:
                recipe_label = "Previous Recipe"
            self.recipe_text_widget.insert(tk.END, f"{recipe_label}\n", "section")
            self.recipe_text_widget.insert(tk.END, f"{recipe_data['full_recipe_text']}\n", "body")

        if self.current_recipe_is_stale:
            self.recipe_text_widget.insert(tk.END, "\n(Showing previous recipe - new recipe not yet available)", "stale")

        self.recipe_text_widget.config(state=tk.DISABLED)
        self.recipe_text_widget.see(1.0)

    def _refresh_recipe_window(self):
        if not self.display_window_enabled or self.recipe_text_widget is None:
            return

        try:
            self._render_recipe_content(self.current_spoken_text)
            self.root.update_idletasks()
        except Exception as exc:
            rospy.logwarn("Disabling display after recipe window failure: %s", exc)
            self.display_window_enabled = False

    def _refresh_image_window(self):
        if not self.display_window_enabled or self.current_image is None or self.image_label is None:
            return

        try:
            # The image is stored as RGB numpy array from PIL
            pil_image = Image.fromarray(self.current_image)
            photo = ImageTk.PhotoImage(pil_image)
            self.current_photo_image = photo  # Keep reference
            self.image_label.config(image=photo)
            self.root.update_idletasks()
        except Exception as exc:
            rospy.logwarn("Failed to show generated image window: %s", exc)

    def _pump_windows(self):
        if not self.display_window_enabled or self.root is None:
            return

        if self.recipe_window_dirty:
            self._refresh_recipe_window()
            self.recipe_window_dirty = False

        if self.image_window_dirty:
            self._refresh_image_window()
            self.image_window_dirty = False

        try:
            self.root.update()
        except Exception:
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
                provider="replicate", # can use pay-as-you-go credits
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
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="PNG")
        except Exception as exc:
            return b"", "failed to encode generated image: {0}".format(exc)

        image_bytes = image_buffer.getvalue()
        if not image_bytes:
            return b"", "Hugging Face image request returned an empty image payload"

        return image_bytes, ""

    def _decode_image_bytes(self, image_bytes):
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image.load()  # Force load to verify integrity
            # Convert to RGB numpy array for storage
            image_array = np.array(pil_image.convert("RGB"))
            return image_array
        except Exception as exc:
            raise RuntimeError("generated image payload could not be decoded: {0}".format(exc))

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
