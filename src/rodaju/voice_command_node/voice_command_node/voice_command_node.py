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

import pyaudio

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv

from voice_command_node.stt import STT
from voice_command_node.wakeup import WakeupWord, WAKEUP_AVAILABLE
from voice_command_node.llm_parser import LLMCommandParser

try:
    from recycle_interfaces.msg import SortCommand
    INTERFACES_AVAILABLE = True
except ImportError:
    from std_msgs.msg import String
    INTERFACES_AVAILABLE = False

from std_msgs.msg import String as _StrMsg

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

WAKEUP_MODEL_FILENAME = "hello_rokey_8332_32.tflite"
WAKEUP_MODEL_PATH     = os.path.join(package_path, "resource", WAKEUP_MODEL_FILENAME)


# ═══════════════════════════════════════════════════════════════
#  VoiceCommandNode
# ═══════════════════════════════════════════════════════════════

class VoiceCommandNode(Node):

    def __init__(self):
        super().__init__("voice_command_node")

        # ── 파라미터 ────────────────────────────────────────
        self.declare_parameter("stt_duration",     5)
        self.declare_parameter("stt_samplerate",   16000)
        self.declare_parameter("mic_rate",         48000)
        self.declare_parameter("mic_chunk",        12000)
        self.declare_parameter("mic_device_index", 10)
        self.declare_parameter("wakeup_threshold", 0.3)
        self.declare_parameter("use_wakeup",       True)

        stt_dur      = self.get_parameter("stt_duration").value
        stt_rate     = self.get_parameter("stt_samplerate").value
        mic_rate     = self.get_parameter("mic_rate").value
        mic_chunk    = self.get_parameter("mic_chunk").value
        mic_dev      = self.get_parameter("mic_device_index").value
        self._use_wakeup = self.get_parameter("use_wakeup").value

        if not OPENAI_API_KEY:
            self.get_logger().error("OPENAI_API_KEY not set! Check .env or environment.")

        # ── STT / LLM 초기화 ────────────────────────────────
        self._stt    = STT(openai_api_key=OPENAI_API_KEY, duration=stt_dur, samplerate=stt_rate)
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
        self.get_logger().info("[Voice] 청취 루프 시작.")

        while self._running and rclpy.ok():
            try:
                if self._use_wakeup and self._mic_controller is not None:
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

                # ── LLM 파싱 ──
                parsed = self._parser.parse(text)
                if not parsed:
                    self.get_logger().info(f"[LLM] 명령 인식 실패: '{text}'")
                    continue

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
            msg                = SortCommand()
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
            self._pub_log(f"[CMD 발행] cmd={msg.cmd} mode={msg.mode} raw='{msg.raw_text}'")
            self._pub_log("웨이크업 단어 대기 중")
        except Exception as e:
            self.get_logger().error(f"[Voice] 발행 오류: {e}")

    def destroy_node(self):
        self._running = False
        super().destroy_node()

    def _pub_log(self, text: str):
        """대시보드 모달 트리거용 로그 토픽 발행."""
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
