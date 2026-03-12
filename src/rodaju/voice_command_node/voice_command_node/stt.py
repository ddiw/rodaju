#!/usr/bin/env python3
"""OpenAI Whisper 기반 STT (Speech-to-Text)."""

import os
import tempfile

import scipy.io.wavfile as wav
import sounddevice as sd
from openai import OpenAI


class STT:
    def __init__(self, openai_api_key: str, duration: int = 5, samplerate: int = 16000):
        self.client     = OpenAI(api_key=openai_api_key)
        self.duration   = duration      # 녹음 시간 (초)
        self.samplerate = samplerate    # Whisper 권장 16kHz

    def speech2text(self) -> str:
        print(f"[STT] 녹음 시작... {self.duration}초 동안 말해주세요.")
        audio = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        print("[STT] 녹음 완료. Whisper API 전송 중...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, self.samplerate, audio)
            tmp_path = f.name

        try:
            with open(tmp_path, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", file=f
                )
            text = transcript.text.strip()
            print(f"[STT] 결과: '{text}'")
            return text
        finally:
            os.unlink(tmp_path)
