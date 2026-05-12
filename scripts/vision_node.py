#!/usr/bin/env python3

import rospy

from chef_robot_assistant.srv import DetectIngredients
from chef_robot_assistant.srv import DetectIngredientsResponse


NODE_NAME = "vision_node"


class VisionNode(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)
        self.stub_detected_ingredients = rospy.get_param(
            "~stub_detected_ingredients",
            ["tomato", "egg"]
        )
        rospy.Service("detect_ingredients", DetectIngredients, self.handle_detect)
        rospy.loginfo("%s stub service is ready", NODE_NAME)

    def handle_detect(self, _request):
        ingredients = [item.strip().lower() for item in self.stub_detected_ingredients]
        ingredients = [item for item in ingredients if item]
        rospy.loginfo("Stub detection returned: %s", ingredients)
        return DetectIngredientsResponse(
            success=bool(ingredients),
            ingredients=ingredients,
            message="Foundation stub ingredient detection response"
        )


def main():
    VisionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
