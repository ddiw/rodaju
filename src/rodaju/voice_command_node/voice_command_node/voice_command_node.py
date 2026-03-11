#!/usr/bin/env python3
"""
voice_command_node.py  ─  음성인식 분석 노드
═══════════════════════════════════════════════════════════════

[처리 흐름]
  마이크 상시 청취
    → WakeupWord("hello rokey") 감지
    → OpenAI Whisper API로 STT
    → GPT-4o LLM으로 명령어 파싱
    → SortCommand 토픽 발행 (/recycle/command)

[인식 명령 예시]
  "분류 시작해"                  → START sorting
  "멈춰 / 정지"                  → STOP
  "잠깐 멈춰"                    → PAUSE
  "다시 시작해"                  → RESUME
  "플라스틱 먼저 분류해"          → SET_POLICY priority=PLASTIC
  "캔이랑 종이만 분류해"          → SET_POLICY exclude=PLASTIC+TRASH
  "다 주워"                      → SET_POLICY exclude=0
"""

import os
import threading

import numpy as np
import pyaudio
import scipy.io.wavfile as wav
import sounddevice as sd
import tempfile

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

try:
    from recycle_interfaces.msg import SortCommand
    INTERFACES_AVAILABLE = True
except ImportError:
    from std_msgs.msg import String
    INTERFACES_AVAILABLE = False

# 대시보드 로그 브릿지용 (항상 import)
from std_msgs.msg import String as _StrMsg

try:
    from openwakeword.model import Model as WakeupModel
    from scipy.signal import resample
    WAKEUP_AVAILABLE = True
except ImportError:
    WAKEUP_AVAILABLE = False

try:
    from voice_command_node.MicController import MicController, MicConfig
    MIC_CONTROLLER_AVAILABLE = True
except ImportError:
    MIC_CONTROLLER_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
#  패키지 경로 & API 키 로드
# ═══════════════════════════════════════════════════════════════

PACKAGE_NAME = "voice_command_node"

try:
    package_path = get_package_share_directory(PACKAGE_NAME)
except Exception:
    package_path = os.path.dirname(__file__)

load_dotenv(dotenv_path=os.path.join(package_path, "resource", ".env"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# WakeupWord 모델 경로
WAKEUP_MODEL_FILENAME = "hello_rokey_8332_32.tflite"
WAKEUP_MODEL_PATH     = os.path.join(package_path, "resource", WAKEUP_MODEL_FILENAME)


# ═══════════════════════════════════════════════════════════════
#  STT
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
#  WakeupWord
# ═══════════════════════════════════════════════════════════════

class WakeupWord:
    """openwakeword 기반 웨이크업 감지."""

    def __init__(self, buffer_size: int, model_path: str = WAKEUP_MODEL_PATH):
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


# ═══════════════════════════════════════════════════════════════
#  LLM 명령어 파서
# ═══════════════════════════════════════════════════════════════

# 프롬프트: 분류로봇 명령어 파싱 전용
_PROMPT = """
당신은 재활용 분류 로봇을 제어하는 명령어 파서입니다.
사용자의 음성 명령을 분석하여 아래 JSON 형식으로만 응답하세요.
다른 설명이나 마크다운 없이 JSON만 출력하세요.

<명령어 종류>
- SWEEP      : 테이블 훑기 (스윕, 청소, 훑어줘, 모아, 모아줘, 쓰레기 모아, 쓰레기 모아줘 등)
- START      : 분류 시작. 우선순위/제외 지정과 함께 분류를 시작하는 경우에도 사용.
               (분류해, 분류 시작, 캔부터 분류해줘, 플라스틱 먼저 분류해줘 등)
- STOP       : 완전 정지
- PAUSE      : 일시정지
- RESUME     : 재개
- STANDBY    : 홈 위치로 복귀 / 대기
- SET_POLICY : 현재 작업 중 정책만 변경 (로봇이 이미 분류 중일 때)

<분류 카테고리>
- PLASTIC : 페트병, 생수병, 플라스틱병
- CAN     : 캔, 알루미늄캔, 금속캔
- PAPER   : 종이컵

<출력 JSON 형식>
{{
  "cmd": "명령어",
  "mode": "sorting | stop | standby | (빈 문자열)",
  "priority_order": ["우선순위 순서대로 나열 (PLASTIC|CAN|PAPER), 없으면 빈 배열"],
  "exclude": ["제외할 카테고리 목록"],
  "raw_text": "원본 텍스트"
}}

<예시>
입력: "스윕해"
출력: {{"cmd":"SWEEP","mode":"","priority_order":[],"exclude":[],"raw_text":"스윕해"}}

입력: "훑어줘"
출력: {{"cmd":"SWEEP","mode":"","priority_order":[],"exclude":[],"raw_text":"훑어줘"}}

입력: "쓰레기 모아"
출력: {{"cmd":"SWEEP","mode":"","priority_order":[],"exclude":[],"raw_text":"쓰레기 모아"}}

입력: "쓰레기 모아줘"
출력: {{"cmd":"SWEEP","mode":"","priority_order":[],"exclude":[],"raw_text":"쓰레기 모아줘"}}

입력: "분류해"
출력: {{"cmd":"START","mode":"sorting","priority_order":[],"exclude":[],"raw_text":"분류해"}}

입력: "분류 시작해"
출력: {{"cmd":"START","mode":"sorting","priority_order":[],"exclude":[],"raw_text":"분류 시작해"}}

입력: "캔부터 분류해줘"
출력: {{"cmd":"START","mode":"sorting","priority_order":["CAN"],"exclude":[],"raw_text":"캔부터 분류해줘"}}

입력: "페트병 먼저 분류해줘"
출력: {{"cmd":"START","mode":"sorting","priority_order":["PLASTIC"],"exclude":[],"raw_text":"페트병 먼저 분류해줘"}}

입력: "플라스틱 캔 순서대로 분류해줘"
출력: {{"cmd":"START","mode":"sorting","priority_order":["PLASTIC","CAN"],"exclude":[],"raw_text":"플라스틱 캔 순서대로 분류해줘"}}

입력: "캔이랑 종이컵만 분류해"
출력: {{"cmd":"START","mode":"sorting","priority_order":[],"exclude":["PLASTIC"],"raw_text":"캔이랑 종이컵만 분류해"}}

입력: "잠깐 멈춰"
출력: {{"cmd":"PAUSE","mode":"","priority_order":[],"exclude":[],"raw_text":"잠깐 멈춰"}}

입력: "다시 시작해"
출력: {{"cmd":"RESUME","mode":"sorting","priority_order":[],"exclude":[],"raw_text":"다시 시작해"}}

입력: "다 주워"
출력: {{"cmd":"SET_POLICY","mode":"","priority_order":[],"exclude":[],"raw_text":"다 주워"}}

<사용자 입력>
"{user_input}"
"""

EXCLUDE_BITS = {"PLASTIC": 1, "CAN": 2, "PAPER": 4}


class LLMCommandParser:
    """GPT-4o 기반 명령어 파서."""

    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,          # 명령어 파싱은 일관성 우선
            openai_api_key=openai_api_key,
        )
        self.prompt = PromptTemplate(
            input_variables=["user_input"],
            template=_PROMPT,
        )
        self.chain = self.prompt | self.llm

    def parse(self, text: str) -> dict | None:
        """
        STT 텍스트 → SortCommand 필드 딕셔너리.
        파싱 실패 시 None 반환.
        """
        import json
        try:
            response = self.chain.invoke({"user_input": text})
            raw      = response.content.strip()
            print(f"[LLM] 응답: {raw}")

            # 마크다운 코드블록 제거
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            # JSON 파싱
            data = json.loads(raw)

            # exclude 리스트 → exclude_mask 비트 변환
            exclude_mask = 0
            for cat in data.get("exclude", []):
                exclude_mask |= EXCLUDE_BITS.get(cat.upper(), 0)

            # priority_order 정규화
            priority_order = [p.upper() for p in data.get("priority_order", [])
                              if p.upper() in EXCLUDE_BITS]

            cmd = data.get("cmd", "NOOP").upper()
            if cmd == "STANDBY":
                cmd  = "START"
                data["mode"] = "standby"

            result = {
                "cmd"          : cmd,
                "mode"         : data.get("mode", ""),
                "priority_order": priority_order,
                "exclude_mask" : exclude_mask,
                "raw_text"     : text,
            }

            print(f"[LLM] 파싱결과: {result}")
            return result if cmd != "NOOP" else None

        except Exception as e:
            print(f"[LLM] 파싱 실패: {e} / 원문: '{text}'")
            return None


# ═══════════════════════════════════════════════════════════════
#  VoiceCommandNode
# ═══════════════════════════════════════════════════════════════

class VoiceCommandNode(Node):

    def __init__(self):
        super().__init__("voice_command_node")

        # ── 파라미터 ────────────────────────────────────────
        self.declare_parameter("stt_duration",     5)       # 녹음 시간 (초)
        self.declare_parameter("stt_samplerate",   16000)
        self.declare_parameter("mic_rate",         48000)   # MicController rate
        self.declare_parameter("mic_chunk",        12000)
        self.declare_parameter("mic_device_index", 10)
        self.declare_parameter("wakeup_threshold", 0.3)
        self.declare_parameter("use_wakeup",       True)    # False면 웨이크업 없이 바로 녹음

        stt_dur      = self.get_parameter("stt_duration").value
        stt_rate     = self.get_parameter("stt_samplerate").value
        mic_rate     = self.get_parameter("mic_rate").value
        mic_chunk    = self.get_parameter("mic_chunk").value
        mic_dev      = self.get_parameter("mic_device_index").value
        self._use_wakeup = self.get_parameter("use_wakeup").value

        if not OPENAI_API_KEY:
            self.get_logger().error("OPENAI_API_KEY not set! Check .env or environment.")

        # ── STT 초기화 ─────────────────────────
        self._stt = STT(
            openai_api_key=OPENAI_API_KEY,
            duration=stt_dur,
            samplerate=stt_rate,
        )

        # ── LLM 파서 초기화 ────────────
        self._parser = LLMCommandParser(openai_api_key=OPENAI_API_KEY)

        # ── MicController + WakeupWord 초기화 ────────────────
        self._mic_controller = None
        self._wakeup         = None

        if self._use_wakeup:
            self._init_mic_and_wakeup(mic_rate, mic_chunk, mic_dev)

        # ── 발행자 ──────────────────────────────────────────
        if INTERFACES_AVAILABLE:
            self._pub = self.create_publisher(SortCommand, "/recycle/command", 10)
        else:
            self._pub = self.create_publisher(String, "/recycle/command", 10)

        # ── 대시보드 로그 브릿지 발행자 ─────────────────────
        self._log_pub = self.create_publisher(_StrMsg, "/recycle/voice/log", 10)

        # ── 청취 스레드 ──────────────────────────────────────
        self._running = True
        self._thread  = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f"VoiceCommandNode ready  "
            f"[wakeup={'ON' if self._use_wakeup else 'OFF'}, "
            f"stt=OpenAI Whisper, llm=GPT-4o]"
        )

    # ═══════════════════════════════════════════════════════
    #  MicController + WakeupWord 초기화
    # ═══════════════════════════════════════════════════════

    def _init_mic_and_wakeup(self, mic_rate, mic_chunk, mic_dev):
        if not MIC_CONTROLLER_AVAILABLE:
            self.get_logger().warn("MicController not available – wakeup disabled.")
            self._use_wakeup = False
            return

        if not WAKEUP_AVAILABLE:
            self.get_logger().warn("openwakeword not available – wakeup disabled.")
            self._use_wakeup = False
            return

        try:
            mic_config = MicConfig(
                chunk=mic_chunk,
                rate=mic_rate,
                channels=1,
                record_seconds=5,
                fmt=pyaudio.paInt16,
                device_index=mic_dev,
                buffer_size=mic_chunk * 2,
            )
            self._mic_controller = MicController(config=mic_config)
            self._wakeup         = WakeupWord(
                buffer_size=mic_config.buffer_size,
                model_path=WAKEUP_MODEL_PATH,
            )
            self.get_logger().info("MicController & WakeupWord initialized.")
        except Exception as e:
            self.get_logger().warn(f"MicController init failed: {e} – wakeup disabled.")
            self._use_wakeup = False

    # ═══════════════════════════════════════════════════════
    #  청취 루프
    # ═══════════════════════════════════════════════════════

    def _listen_loop(self):
        """
          1. 마이크 스트림 열기
          2. WakeupWord 감지 대기
          3. STT → LLM 파싱 → 발행
          4. 반복
        """
        self.get_logger().info("[Voice] 청취 루프 시작.")

        while self._running and rclpy.ok():
            try:
                if self._use_wakeup and self._mic_controller is not None:
                    # ── 웨이크업 모드 ──
                    self.get_logger().info("[Voice] 마이크 스트림 오픈...")
                    self._mic_controller.open_stream()
                    self._wakeup.set_stream(self._mic_controller.stream)

                    self.get_logger().info("[Voice] 웨이크업 단어 대기 중...")
                    while self._running and not self._wakeup.is_wakeup():
                        pass

                    if not self._running:
                        break

                    self.get_logger().info("[Voice] 웨이크업 감지! STT 시작.")
                    self._pub_log("웨이크업 감지")

                else:
                    # ── 웨이크업 없이 바로 STT (keyboard fallback) ──
                    try:
                        input("[Voice] Enter 키를 누르면 녹음 시작...")
                    except EOFError:
                        import time; time.sleep(2.0)

                # ── STT ─────────────────────────
                self._pub_log("STT 시작")
                text = self._stt.speech2text()
                if not text:
                    self.get_logger().debug("[Voice] STT 결과 없음.")
                    continue

                self.get_logger().info(f"[STT] '{text}'")
                self._pub_log(f"[STT] '{text}'")

                # ── LLM 파싱 (get_keyword.py extract_keyword 방식) ──
                parsed = self._parser.parse(text)
                if not parsed:
                    self.get_logger().info(f"[LLM] 명령 인식 실패: '{text}'")
                    continue

                # ── 발행 ─────────────────────────────────────
                self._publish(parsed)

            except OSError as e:
                self.get_logger().error(f"[Voice] 오디오 스트림 오류: {e}")
                import time; time.sleep(1.0)
            except Exception as e:
                self.get_logger().error(f"[Voice] 루프 오류: {e}")
                import time; time.sleep(0.5)

    # ═══════════════════════════════════════════════════════
    #  발행
    # ═══════════════════════════════════════════════════════

    def _publish(self, parsed: dict):
        try:
            msg              = SortCommand()
            msg.stamp          = self.get_clock().now().to_msg()
            msg.cmd            = parsed["cmd"]
            msg.mode           = parsed["mode"]
            msg.priority_order = parsed["priority_order"]
            msg.exclude_mask   = parsed["exclude_mask"]
            msg.raw_text       = parsed["raw_text"]
            self._pub.publish(msg)
            self.get_logger().info(
                f"[CMD 발행] cmd={msg.cmd} mode={msg.mode} "
                f"priority_order={list(msg.priority_order)} exclude={msg.exclude_mask:#04x} "
                f"raw='{msg.raw_text}'"
            )
            self._pub_log(
                f"[CMD 발행] cmd={msg.cmd} mode={msg.mode} raw='{msg.raw_text}'"
            )
            self._pub_log("웨이크업 단어 대기 중")
        except Exception as e:
            self.get_logger().error(f"[Voice] 발행 오류: {e}")

    def destroy_node(self):
        self._running = False
        super().destroy_node()

    def _pub_log(self, text: str):
        """대시보드 모달 트리거용 로그 토픽 발행"""
        try:
            msg = _StrMsg()
            msg.data = text
            self._log_pub.publish(msg)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
