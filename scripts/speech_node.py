#!/usr/bin/env python3

import math
import os
import re
import shutil
import subprocess
import tempfile
import wave

import rospy

from chef_robot_assistant.srv import TranscribeSpeech
from chef_robot_assistant.srv import TranscribeSpeechResponse


try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


NODE_NAME = "speech_node"


class SpeechNode(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)

        self.input_device = self._param("speech_input_device", "default")
        self.record_seconds = float(self._param("speech_record_seconds", 4.0))
        self.sample_rate_hz = int(self._param("speech_sample_rate_hz", 16000))
        self.channels = int(self._param("speech_channels", 1))
        self.silence_rms_threshold = float(
            self._param("speech_silence_rms_threshold", 150.0)
        )
        self.model_name = self._param("speech_model_name", "small")
        self.compute_type = self._param("speech_compute_type", "int8")
        self.language = self._param("speech_language", "").strip().lower()
        self.beam_size = int(self._param("speech_beam_size", 1))

        self.arecord_path = shutil.which("arecord")
        self.model = None

        rospy.Service("transcribe_speech", TranscribeSpeech, self.handle_transcribe)
        rospy.loginfo(
            "%s ready with model=%s compute_type=%s input_device=%s",
            NODE_NAME,
            self.model_name,
            self.compute_type,
            self.input_device,
        )

    def _param(self, key, default):
        return rospy.get_param("/chef_robot_assistant/{0}".format(key), default)

    def _normalized_transcript(self, text):
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _load_model(self):
        if self.model is not None:
            return self.model

        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper is not installed. Install the requirements first."
            )

        rospy.loginfo(
            "Loading faster-whisper model '%s' with compute_type '%s'",
            self.model_name,
            self.compute_type,
        )
        self.model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type=self.compute_type,
        )
        return self.model

    def _record_audio(self, output_path):
        if not self.arecord_path:
            raise RuntimeError("arecord is not available on this machine.")

        duration_seconds = max(1, int(math.ceil(self.record_seconds)))
        command = [
            self.arecord_path,
            "-q",
            "-D",
            self.input_device,
            "-t",
            "wav",
            "-f",
            "S16_LE",
            "-r",
            str(self.sample_rate_hz),
            "-c",
            str(self.channels),
            "-d",
            str(duration_seconds),
            output_path,
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=duration_seconds + 3,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("audio capture timed out") from exc
        except subprocess.CalledProcessError as exc:
            error_output = exc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(error_output or "audio capture failed") from exc

        if not os.path.exists(output_path) or os.path.getsize(output_path) <= 44:
            raise RuntimeError("no audio captured from microphone")

    def _audio_rms_level(self, audio_path):
        try:
            with wave.open(audio_path, "rb") as wav_file:
                sample_width = wav_file.getsampwidth()
                frame_count = wav_file.getnframes()
                raw_frames = wav_file.readframes(frame_count)
        except (wave.Error, OSError) as exc:
            raise RuntimeError("failed to read captured audio: {0}".format(exc))

        if sample_width != 2 or not raw_frames:
            return 0.0

        sample_count = len(raw_frames) // 2
        if sample_count == 0:
            return 0.0

        total_square = 0.0
        for index in range(0, len(raw_frames), 2):
            sample = int.from_bytes(
                raw_frames[index:index + 2],
                byteorder="little",
                signed=True,
            )
            total_square += float(sample * sample)

        return math.sqrt(total_square / sample_count)

    def _ensure_non_silent_audio(self, audio_path):
        rms_level = self._audio_rms_level(audio_path)
        rospy.loginfo("Captured audio RMS level: %.2f", rms_level)
        if rms_level < self.silence_rms_threshold:
            raise RuntimeError(
                "captured audio is too quiet to treat as speech"
            )

    def _transcribe_audio(self, audio_path):
        model = self._load_model()
        language = self.language or None

        segments, _info = model.transcribe(
            audio_path,
            beam_size=max(1, self.beam_size),
            language=language,
        )
        transcript = " ".join(segment.text for segment in segments)
        return self._normalized_transcript(transcript)

    def handle_transcribe(self, _request):
        fd, audio_path = tempfile.mkstemp(
            prefix="chef_robot_assistant_speech_",
            suffix=".wav",
        )
        os.close(fd)

        try:
            self._record_audio(audio_path)
            self._ensure_non_silent_audio(audio_path)
            transcript = self._transcribe_audio(audio_path)
        except RuntimeError as exc:
            rospy.logwarn("Speech capture/transcription failed: %s", exc)
            return TranscribeSpeechResponse(
                success=False,
                transcript="",
                message=str(exc),
            )
        except Exception as exc:
            rospy.logerr("Unexpected speech node failure: %s", exc)
            return TranscribeSpeechResponse(
                success=False,
                transcript="",
                message="speech transcription failed unexpectedly",
            )
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        if not transcript:
            return TranscribeSpeechResponse(
                success=False,
                transcript="",
                message="no speech recognized from captured audio",
            )

        rospy.loginfo("Transcribed speech: %s", transcript)
        return TranscribeSpeechResponse(
            success=True,
            transcript=transcript,
            message="speech transcription succeeded",
        )


def main():
    SpeechNode()
    rospy.spin()


if __name__ == "__main__":
    main()
