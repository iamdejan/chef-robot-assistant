#!/usr/bin/env python3

import rospy
from std_msgs.msg import String, Bool
from enum import Enum, auto


NODE_NAME = "interaction_manager"

# Topics
USER_COMMAND_TOPIC = "/chef_robot_assistant/user_command"
TTS_TOPIC = "/chef_robot_assistant/tts"
TTS_DONE_TOPIC = "/chef_robot_assistant/tts_done"

# Affirmative responses for "yes"
AFFIRMATIVE_RESPONSES = {"yes", "yea", "yeah", "ya"}

# Questions
QUESTION_ACTIVATE_SCANNING = (
    "Would you like to activate ingredient scanning?"
)
QUESTION_PLACE_INGREDIENTS = (
    "Please place your ingredients in front of the camera."
)


class State(Enum):
    IDLE = auto()
    PROMPT_PLACE_INGREDIENTS = auto()
    DETECT_INGREDIENTS = auto()
    CHECK_INGREDIENT_COUNT = auto()
    ASK_CUISINE = auto()
    LISTEN_CUISINE = auto()
    VALIDATE_CUISINE = auto()
    GENERATE_RECIPE = auto()
    PRESENT_RECIPE = auto()
    ASK_IMAGE_OPTION = auto()
    LISTEN_IMAGE_OPTION = auto()
    GENERATE_IMAGE = auto()
    PRESENT_IMAGE = auto()
    RESET_TO_IDLE = auto()


class SubState(Enum):
    SPEAKING = auto()
    LISTENING = auto()
    PROCESSING = auto()


class InteractionManager:
    def __init__(self):
        rospy.init_node(NODE_NAME)

        self.state = State.IDLE
        self.sub_state = SubState.PROCESSING
        self.last_user_response = None
        self.tts_done = False

        # Publishers
        self.tts_pub = rospy.Publisher(
            TTS_TOPIC,
            String,
            queue_size=10
        )

        # Subscribers
        self.user_command_sub = rospy.Subscriber(
            USER_COMMAND_TOPIC,
            String,
            self.on_user_command
        )

        self.tts_done_sub = rospy.Subscriber(
            TTS_DONE_TOPIC,
            Bool,
            self.on_tts_done
        )

        rospy.loginfo(f"{NODE_NAME} started. Current state: {self.state.name}")
        self.run()

    def on_user_command(self, msg: String):
        if self.sub_state == SubState.LISTENING:
            self.last_user_response = msg.data.strip().lower()
            rospy.loginfo(f"Heard user: '{self.last_user_response}'")
            self.sub_state = SubState.PROCESSING

    def on_tts_done(self, msg: Bool):
        if self.sub_state == SubState.SPEAKING:
            rospy.loginfo("TTS done")
            self.tts_done = True

    def speak(self, text: str):
        rospy.loginfo(f"Speaking: {text}")
        self.tts_done = False
        self.sub_state = SubState.SPEAKING
        msg = String(data=text)

        # Wait for at least one subscriber to connect so the message
        # isn't silently dropped.
        wait_start = rospy.Time.now()
        while self.tts_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            if (rospy.Time.now() - wait_start).to_sec() > 5.0:
                rospy.logwarn("No TTS subscriber connected after 5s, publishing anyway")
                break
            rospy.sleep(0.1)

        self.tts_pub.publish(msg)
        rospy.loginfo("Speaking is done.")

    def wait_for_tts(self):
        while not rospy.is_shutdown() and not self.tts_done:
            rospy.sleep(0.1)
        self.tts_done = False

    def wait_for_user_response(self):
        self.last_user_response = None
        self.sub_state = SubState.LISTENING
        while not rospy.is_shutdown() and self.last_user_response is None:
            rospy.sleep(0.1)
        return self.last_user_response

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            rospy.loginfo(f"[run] state: {self.state}, sub-state: {self.sub_state}")

            if self.sub_state != SubState.PROCESSING:
                rate.sleep()
                continue

            if self.state == State.IDLE:
                self.handle_idle()
            elif self.state == State.PROMPT_PLACE_INGREDIENTS:
                self.handle_prompt_place_ingredients()
            elif self.state == State.DETECT_INGREDIENTS:
                self.handle_detect_ingredients()
            elif self.state == State.CHECK_INGREDIENT_COUNT:
                self.handle_check_ingredient_count()
            elif self.state == State.ASK_CUISINE:
                self.handle_ask_cuisine()
            elif self.state == State.LISTEN_CUISINE:
                self.handle_listen_cuisine()
            elif self.state == State.VALIDATE_CUISINE:
                self.handle_validate_cuisine()
            elif self.state == State.GENERATE_RECIPE:
                self.handle_generate_recipe()
            elif self.state == State.PRESENT_RECIPE:
                self.handle_present_recipe()
            elif self.state == State.ASK_IMAGE_OPTION:
                self.handle_ask_image_option()
            elif self.state == State.LISTEN_IMAGE_OPTION:
                self.handle_listen_image_option()
            elif self.state == State.GENERATE_IMAGE:
                self.handle_generate_image()
            elif self.state == State.PRESENT_IMAGE:
                self.handle_present_image()
            elif self.state == State.RESET_TO_IDLE:
                self.handle_reset_to_idle()

            rate.sleep()

    # ===========================
    # State Handlers
    # ===========================

    def handle_idle(self):
        self.speak(QUESTION_ACTIVATE_SCANNING)
        self.wait_for_tts()
        response = self.wait_for_user_response()

        if response in AFFIRMATIVE_RESPONSES:
            rospy.loginfo("User wants to activate ingredient scanning")
            self.state = State.PROMPT_PLACE_INGREDIENTS
        else:
            rospy.loginfo("User declined or gave unclear answer, re-asking")
            # Stay in IDLE; loop will re-run handle_idle

    def handle_prompt_place_ingredients(self):
        self.speak(QUESTION_PLACE_INGREDIENTS)
        self.wait_for_tts()
        rospy.loginfo("Transitioning to DETECT_INGREDIENTS (placeholder)")
        self.state = State.DETECT_INGREDIENTS

    def handle_detect_ingredients(self):
        rospy.loginfo("DETECT_INGREDIENTS: Placeholder for vision integration")
        rospy.sleep(2)
        self.state = State.CHECK_INGREDIENT_COUNT

    def handle_check_ingredient_count(self):
        rospy.loginfo("CHECK_INGREDIENT_COUNT: Placeholder")
        rospy.sleep(1)
        self.state = State.ASK_CUISINE

    def handle_ask_cuisine(self):
        self.speak("Which type of cuisine would you like to cook?")
        self.wait_for_tts()
        self.state = State.LISTEN_CUISINE

    def handle_listen_cuisine(self):
        response = self.wait_for_user_response()
        rospy.loginfo(f"User chose cuisine: {response}")
        self.state = State.VALIDATE_CUISINE

    def handle_validate_cuisine(self):
        self.speak("Got it. Let me generate a recipe for you.")
        self.wait_for_tts()
        self.state = State.GENERATE_RECIPE

    def handle_generate_recipe(self):
        rospy.loginfo("GENERATE_RECIPE: Placeholder for recipe generation")
        rospy.sleep(2)
        self.state = State.PRESENT_RECIPE

    def handle_present_recipe(self):
        self.speak("Here is your recipe.")
        self.wait_for_tts()
        self.state = State.ASK_IMAGE_OPTION

    def handle_ask_image_option(self):
        self.speak("Would you like me to generate an image of the dish?")
        self.wait_for_tts()
        self.state = State.LISTEN_IMAGE_OPTION

    def handle_listen_image_option(self):
        response = self.wait_for_user_response()
        if response in AFFIRMATIVE_RESPONSES:
            self.state = State.GENERATE_IMAGE
        else:
            self.state = State.RESET_TO_IDLE

    def handle_generate_image(self):
        rospy.loginfo("GENERATE_IMAGE: Placeholder for image generation")
        rospy.sleep(2)
        self.state = State.PRESENT_IMAGE

    def handle_present_image(self):
        self.speak("Here is the image of your dish.")
        self.wait_for_tts()
        self.state = State.RESET_TO_IDLE

    def handle_reset_to_idle(self):
        rospy.loginfo("Resetting to IDLE")
        self.speak("Thank you. Let me know if you need anything else.")
        self.wait_for_tts()
        self.state = State.IDLE


if __name__ == "__main__":
    InteractionManager()
