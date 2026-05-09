#!/usr/bin/env python3

import rospy

from chef_robot_assistant.srv import GenerateDishImage
from chef_robot_assistant.srv import GenerateDishImageResponse
from chef_robot_assistant.srv import SpeakText
from chef_robot_assistant.srv import SpeakTextResponse


NODE_NAME = "media_node"


class MediaNode(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)
        rospy.Service("generate_dish_image", GenerateDishImage, self.handle_image)
        rospy.Service("speak_text", SpeakText, self.handle_speak)
        rospy.loginfo("%s stub services are ready", NODE_NAME)

    def handle_image(self, request):
        rospy.loginfo(
            "Stub image request for dish='%s' description='%s'",
            request.dish_name,
            request.description
        )
        return GenerateDishImageResponse(
            success=True,
            image_path="stub://generated_dish_image",
            message="Foundation stub image response"
        )

    def handle_speak(self, request):
        rospy.loginfo("Stub speech output: %s", request.text)
        return SpeakTextResponse(
            success=True,
            message="Foundation stub speech response"
        )


def main():
    MediaNode()
    rospy.spin()


if __name__ == "__main__":
    main()
