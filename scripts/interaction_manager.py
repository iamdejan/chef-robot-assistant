#!/usr/bin/env python3

import rospy

from chef_robot_assistant.srv import DetectIngredients
from chef_robot_assistant.srv import GenerateDishImage
from chef_robot_assistant.srv import GenerateRecipe
from chef_robot_assistant.srv import SpeakText
from chef_robot_assistant.srv import TranscribeSpeech


NODE_NAME = "interaction_manager"
VALID_CUISINES = ("malay", "western", "chinese")
YES_WORDS = ("yes", "yeah", "yep")
NO_WORDS = ("no", "nope")


class InteractionManager(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)

        self.state = "IDLE"
        self.retry_count = 0
        self.cycle_in_progress = False
        self.detected_ingredients = []
        self.selected_cuisine = ""
        self.recipe_result = {}
        self.image_requested = False

        self.camera_topic = self._param("camera_topic", "/camera/image_raw")
        self.camera_frame_topic_out = self._param(
            "camera_frame_topic_out",
            "/chef_robot_assistant/image_raw"
        )
        self.detection_confidence_threshold = self._param(
            "detection_confidence_threshold",
            0.5
        )
        self.capture_duration_sec = self._param("capture_duration_sec", 3.0)
        self.min_ingredient_count = self._param("min_ingredient_count", 2)
        self.speech_retry_limit = self._param("speech_retry_limit", 3)
        self.default_cuisine = self._param("default_cuisine", "malay").lower()
        self.idle_restart_delay_sec = float(
            self._param("idle_restart_delay_sec", 3.0)
        )
        self.image_generation_enabled = self._param(
            "image_generation_enabled",
            True
        )

        self._wait_for_service("detect_ingredients")
        self._wait_for_service("transcribe_speech")
        self._wait_for_service("generate_recipe")
        self._wait_for_service("generate_dish_image")
        self._wait_for_service("speak_text")

        self.detect_ingredients = rospy.ServiceProxy(
            "detect_ingredients",
            DetectIngredients
        )
        self.transcribe_speech = rospy.ServiceProxy(
            "transcribe_speech",
            TranscribeSpeech
        )
        self.generate_recipe = rospy.ServiceProxy(
            "generate_recipe",
            GenerateRecipe
        )
        self.generate_dish_image = rospy.ServiceProxy(
            "generate_dish_image",
            GenerateDishImage
        )
        self.speak_text = rospy.ServiceProxy("speak_text", SpeakText)
        self._clear_recipe_display_state()

        rospy.loginfo("%s ready for continuous interaction flow", NODE_NAME)
        self._schedule_next_cycle(1.0)

    def _param(self, key, default):
        return rospy.get_param("/chef_robot_assistant/{0}".format(key), default)

    def _wait_for_service(self, service_name):
        rospy.loginfo("Waiting for service %s", service_name)
        rospy.wait_for_service(service_name, timeout=10.0)

    def _run_once(self, _event):
        if self.cycle_in_progress or rospy.is_shutdown():
            return

        self.cycle_in_progress = True
        try:
            self.run_foundation_cycle()
        except rospy.ServiceException as exc:
            rospy.logerr("Service call failed: %s", exc)
            self._reset_to_idle()
        except rospy.ROSException as exc:
            rospy.logerr("ROS runtime error: %s", exc)
            self._reset_to_idle()
        finally:
            self.cycle_in_progress = False
            if self.state == "IDLE" and not rospy.is_shutdown():
                self._schedule_next_cycle(self.idle_restart_delay_sec)

    def _schedule_next_cycle(self, delay_sec):
        rospy.loginfo(
            "Scheduling next interaction cycle in %.1f seconds",
            max(0.0, delay_sec),
        )
        rospy.Timer(
            rospy.Duration(max(0.0, delay_sec)),
            self._run_once,
            oneshot=True,
        )

    def _set_state(self, new_state):
        self.state = new_state
        rospy.loginfo("State -> %s", new_state)

    def _say(self, text):
        rospy.loginfo("Robot says: %s", text)
        response = self.speak_text(text)
        if not response.success:
            rospy.logwarn("SpeakText returned failure: %s", response.message)

    def _publish_recipe_display_state(self):
        base_key = "/chef_robot_assistant/last_recipe"
        rospy.set_param(base_key + "/dish_name", self.recipe_result.get("dish_name", ""))
        rospy.set_param(
            base_key + "/spoken_summary",
            self.recipe_result.get("spoken_summary", ""),
        )
        rospy.set_param(
            base_key + "/full_recipe_text",
            self.recipe_result.get("full_recipe_text", ""),
        )
        rospy.set_param(
            base_key + "/missing_ingredients",
            list(self.recipe_result.get("missing_ingredients", [])),
        )

    def _clear_recipe_display_state(self):
        base_key = "/chef_robot_assistant/last_recipe"
        rospy.set_param(base_key + "/dish_name", "")
        rospy.set_param(base_key + "/spoken_summary", "")
        rospy.set_param(base_key + "/full_recipe_text", "")
        rospy.set_param(base_key + "/missing_ingredients", [])

    def _handle_retry_exhausted_shutdown(self):
        self._say("Retry limit exceeded. I'm shutting down. Please try again.")
        self._reset_to_idle()

    def _retry_service_call(
        self,
        service_name,
        attempt_fn,
        retry_notice,
    ):
        max_attempts = max(1, int(self.speech_retry_limit))
        last_response = None

        for attempt_index in range(max_attempts):
            response = attempt_fn()
            last_response = response
            if response.success:
                return response

            if attempt_index < max_attempts - 1:
                rospy.logwarn("%s failed: %s", service_name, response.message)
                self._say(retry_notice)

        if last_response is not None:
            rospy.logwarn("%s failed after %d attempts: %s", service_name, max_attempts, last_response.message)
        return last_response

    def _is_retryable_detection_failure(self, message):
        normalized = str(message or "").strip().lower()
        retryable_markers = (
            "no stable ingredient detections found",
            "no detections were captured during",
        )
        return any(marker in normalized for marker in retryable_markers)

    def run_foundation_cycle(self):
        self._set_state("PROMPT_PLACE_INGREDIENTS")
        self._say("Please place at least 2 supported ingredients in front of me.")

        while not rospy.is_shutdown():
            self._set_state("DETECT_INGREDIENTS")
            detect_response = self.detect_ingredients()
            if not detect_response.success:
                rospy.logwarn("Detection failed: %s", detect_response.message)
                if self._is_retryable_detection_failure(detect_response.message):
                    self._say(
                        "No supported ingredients detected. Please place ingredients in front of me."
                    )
                    rospy.sleep(1.0)
                    continue
                self._reset_to_idle()
                return

            self.detected_ingredients = list(detect_response.ingredients)

            self._set_state("CHECK_INGREDIENT_COUNT")
            if len(self.detected_ingredients) >= self.min_ingredient_count:
                break

            self._say(
                "Only {0} ingredient detected. Please add more ingredients.".format(
                    len(self.detected_ingredients)
                )
            )
            rospy.sleep(1.0)

        self._set_state("ASK_CUISINE")
        self._say(
            "I am generating a recipe for you. "
            "Do you want Malay, Western or Chinese food?"
        )

        self._set_state("LISTEN_CUISINE")
        transcript = self._capture_transcript()
        if transcript is None:
            self._handle_retry_exhausted_shutdown()
            return

        self._set_state("VALIDATE_CUISINE")
        self.selected_cuisine = self._parse_cuisine(transcript)
        if self.selected_cuisine not in VALID_CUISINES:
            self.selected_cuisine = self.default_cuisine
            self._say(
                "Sorry, I cannot fulfill that requirement. "
                "I will generate a recipe based on the default setting."
            )
        else:
            self._say("Got it, please wait for a moment.")

        self._set_state("GENERATE_RECIPE")
        recipe_response = self._retry_service_call(
            "Recipe generation",
            lambda: self.generate_recipe(
                self.detected_ingredients,
                self.selected_cuisine
            ),
            "Recipe generation failed. I will try again."
        )
        if not recipe_response.success:
            self._handle_retry_exhausted_shutdown()
            return

        self.recipe_result = {
            "dish_name": recipe_response.dish_name,
            "spoken_summary": recipe_response.spoken_summary,
            "full_recipe_text": recipe_response.full_recipe_text,
            "missing_ingredients": list(recipe_response.missing_ingredients),
        }

        self._set_state("PRESENT_RECIPE")
        summary = self.recipe_result["spoken_summary"]
        missing = self.recipe_result["missing_ingredients"]
        self._publish_recipe_display_state()
        if missing:
            summary = "{0} You still need to buy: {1}.".format(
                summary,
                ", ".join(missing)
            )
        self._say(summary)
        rospy.loginfo("Full recipe output:\n%s", self.recipe_result["full_recipe_text"])

        if not self.image_generation_enabled:
            self._say("Thank you for using me.")
            self._reset_to_idle()
            return

        self._set_state("ASK_IMAGE_OPTION")
        self._say("Do you want me to generate an image of the dish?")

        self._set_state("LISTEN_IMAGE_OPTION")
        image_transcript = self._capture_transcript()
        self.image_requested = self._parse_yes_no(image_transcript)

        if not self.image_requested:
            self._say("Thank you for using me.")
            self._reset_to_idle()
            return

        self._set_state("GENERATE_IMAGE")
        image_response = self._retry_service_call(
            "Image generation",
            lambda: self.generate_dish_image(
                self.recipe_result["dish_name"],
                self.recipe_result["spoken_summary"]
            ),
            "Image generation failed. I will try again."
        )
        if not image_response.success:
            self._handle_retry_exhausted_shutdown()
            return

        self._set_state("PRESENT_IMAGE")
        rospy.loginfo("Generated image reference: %s", image_response.image_path)
        self._say("Here is an example image of the dish. Thank you for using me.")
        self._reset_to_idle()

    def _capture_transcript(self):
        self.retry_count = 0
        while self.retry_count < self.speech_retry_limit and not rospy.is_shutdown():
            response = self.transcribe_speech()
            transcript = response.transcript.strip().lower()
            if response.success and transcript:
                return transcript

            self.retry_count += 1
            if self.retry_count < self.speech_retry_limit:
                self._say("Sorry, I cannot hear you. Please say it again.")

        return None

    def _parse_cuisine(self, transcript):
        if transcript is None:
            return ""

        for cuisine in VALID_CUISINES:
            if cuisine in transcript:
                return cuisine
        return ""

    def _parse_yes_no(self, transcript):
        if transcript is None:
            return False

        if any(word in transcript for word in YES_WORDS):
            return True

        if any(word in transcript for word in NO_WORDS):
            return False

        return False

    def _reset_to_idle(self):
        self.state = "IDLE"
        self.retry_count = 0
        self.detected_ingredients = []
        self.selected_cuisine = ""
        self.recipe_result = {}
        self.image_requested = False
        self._clear_recipe_display_state()
        rospy.loginfo("State -> IDLE")


def main():
    InteractionManager()
    rospy.spin()


if __name__ == "__main__":
    main()
