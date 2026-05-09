#!/usr/bin/env python3

import os
import subprocess
import tempfile
import rospy
from std_msgs.msg import String, Bool
from huggingface_hub import InferenceClient


NODE_NAME = "text_to_speech_node"
TTS_TOPIC = "/chef_robot_assistant/tts"
TTS_DONE_TOPIC = "/chef_robot_assistant/tts_done"

# Model configuration
TTS_MODEL = "hexgrad/Kokoro-82M"
TTS_PROVIDER = "fal-ai"


class TextToSpeechNode:
    def __init__(self):
        rospy.init_node(NODE_NAME)

        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            rospy.logfatal("HF_TOKEN environment variable is not set. TTS node cannot start.")
            rospy.signal_shutdown("Missing HF_TOKEN")
            return

        self.client = InferenceClient(
            provider=TTS_PROVIDER,
            api_key=api_key,
        )

        self.sub = rospy.Subscriber(
            TTS_TOPIC,
            String,
            self.on_tts_request
        )

        self.pub = rospy.Publisher(
            TTS_DONE_TOPIC,
            Bool,
            queue_size=10
        )

        rospy.loginfo(f"{NODE_NAME} is ready")
        rospy.spin()

    def play_audio(self, audio_bytes: bytes) -> bool:
        """Play audio bytes using system audio player."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            if subprocess.run(["which", "paplay"], capture_output=True).returncode == 0:
                subprocess.run(["paplay", temp_path], check=True)
            elif subprocess.run(["which", "ffplay"], capture_output=True).returncode == 0:
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", temp_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                rospy.logerr("No audio player found (paplay or ffplay). Audio saved to: %s", temp_path)
                return False
        except subprocess.CalledProcessError as e:
            rospy.logerr(f"Error playing audio: {e}")
            return False
        finally:
            os.remove(temp_path)

        return True

    def on_tts_request(self, msg: String):
        text = msg.data.strip()
        if not text:
            rospy.logwarn("Received empty TTS request, skipping")
            self.pub.publish(Bool(data=True))
            return

        rospy.loginfo(f"Generating speech for: {text}")
        try:
            audio = self.client.text_to_speech(
                text,
                model=TTS_MODEL,
            )
            rospy.loginfo("Speech generated, playing audio...")
            success = self.play_audio(audio)
            self.pub.publish(Bool(data=success))
        except Exception as e:
            rospy.logerr(f"TTS generation failed: {e}")
            self.pub.publish(Bool(data=False))


if __name__ == "__main__":
    TextToSpeechNode()
