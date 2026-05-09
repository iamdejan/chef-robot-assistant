#!/usr/bin/env python3

import rospy

from chef_robot_assistant.srv import TranscribeSpeech
from chef_robot_assistant.srv import TranscribeSpeechResponse


NODE_NAME = "speech_node"


class SpeechNode(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)
        self.default_transcript = rospy.get_param("~stub_transcript", "malay")
        rospy.Service("transcribe_speech", TranscribeSpeech, self.handle_transcribe)
        rospy.loginfo("%s stub service is ready", NODE_NAME)

    def handle_transcribe(self, _request):
        transcript = self.default_transcript.strip().lower()
        rospy.loginfo("Stub transcript returned: %s", transcript)
        return TranscribeSpeechResponse(
            success=bool(transcript),
            transcript=transcript,
            message="Foundation stub speech response"
        )


def main():
    SpeechNode()
    rospy.spin()


if __name__ == "__main__":
    main()
