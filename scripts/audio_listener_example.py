#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os

from transformers import pipeline


class AudioListenerNode:
    def __init__(self):
        rospy.init_node("audio_listener_example")
        self.publisher = rospy.Publisher(
            "/user_command",
            String,
            queue_size=10
        )

        rospy.loginfo("Loading Whisper model...")

        # Whisper via HuggingFace
        self.asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            device=0 if self._gpu_available() else -1
        )

        rospy.loginfo("Audio Listener Node Ready")

        self.sample_rate = 16000
        self.record_seconds = 5

        self.run()

    def _gpu_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception as e:
            rospy.logwarn(f"error when checking CUDA availability: {e}")
            return False

    def record_audio(self):
        rospy.loginfo("Listening... Speak now")
        audio = sd.rec(
            int(self.record_seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )

        sd.wait()
        return audio

    def transcribe_audio(self, audio):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            filename = f.name
            write(filename, self.sample_rate, audio)

        rospy.loginfo("Transcribing audio...")
        result = self.asr(filename, generate_kwargs={"language": "en", "forced_decoder_ids": None})
        os.remove(filename)
        return result["text"]

    def run(self):
        rate = rospy.Rate(0.2)  # every 5 seconds
        while not rospy.is_shutdown():
            try:
                audio = self.record_audio()
                text = self.transcribe_audio(audio)
                rospy.loginfo(f"Recognized: {text}")

                msg = String()
                msg.data = text

                self.publisher.publish(msg)

            except Exception as e:
                rospy.logerr(f"Error: {e}")

            rate.sleep()


if __name__ == "__main__":
    AudioListenerNode()
