#!/usr/bin/env python3
"""openwakeword 기반 웨이크업 단어 감지."""

import os

import numpy as np

try:
    from openwakeword.model import Model as WakeupModel
    from scipy.signal import resample
    WAKEUP_AVAILABLE = True
except ImportError:
    WAKEUP_AVAILABLE = False


class WakeupWord:
    """openwakeword 기반 웨이크업 감지."""

    def __init__(self, buffer_size: int, model_path: str):
        self.model_name  = os.path.basename(model_path).split(".", maxsplit=1)[0]
        self.buffer_size = buffer_size
        self.stream      = None
        self._model      = None
        self._model_path = model_path

    def set_stream(self, stream):
        self._model = WakeupModel(wakeword_models=[self._model_path])
        self.stream = stream

    def is_wakeup(self) -> bool:
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16,
        )
        # 48kHz → 16kHz 리샘플 (MicConfig rate=48000 기준)
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))
        outputs     = self._model.predict(audio_chunk, threshold=0.1)
        confidence  = outputs[self.model_name]
        print(f"[Wakeup] confidence: {confidence:.3f}")
        if confidence > 0.3:
            print("[Wakeup] 웨이크업 감지!")
            return True
        return False
