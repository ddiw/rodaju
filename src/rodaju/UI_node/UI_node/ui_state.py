#!/usr/bin/env python3
"""UI 공유 상태 (ROS 노드와 Flask 웹 서버 간 데이터 공유)."""

import re
import threading
import time


class UIState:
    def __init__(self):
        self.state             = "IDLE"
        self.phase             = "STANDBY"
        self.priority_order: list[str] = []
        self.exclude_mask      = 0
        self.last_message      = "System ready."
        self.progress          = 0.0

        self.current_label       = ""
        self.current_bin         = ""
        self.current_robot_phase = ""

        # 카운터
        self.total   = 0
        self.plastic = 0
        self.can     = 0
        self.paper   = 0
        self.trash   = 0

        # 분류함 용량 (처리된 개수로 역산)
        self.bin_counts: dict[str, int] = {
            "BIN_PLASTIC": 0, "BIN_CAN": 0, "BIN_PAPER": 0, "BIN_TRASH": 0}

        self.log_lines: list[str] = []
        self.lock = threading.Lock()

        # YOLO 프리뷰 최신 JPEG 바이트 (MJPEG 스트림용)
        self.latest_preview: bytes = b""
        self.preview_lock = threading.Lock()

    def update(self, msg):
        with self.lock:
            self.state          = msg.state
            self.phase          = msg.mode       # manager_node는 mode 필드에 phase 값 넣음
            self.priority_order = list(msg.priority_order)
            self.exclude_mask   = msg.exclude_mask
            self.last_message   = msg.last_message
            self.progress       = msg.progress
            self.total          = msg.processed_total
            self.plastic        = msg.processed_plastic
            self.can            = msg.processed_can
            self.paper          = msg.processed_paper
            self.trash          = msg.processed_trash

            # bin_counts 동기화
            self.bin_counts["BIN_PLASTIC"] = self.plastic
            self.bin_counts["BIN_CAN"]     = self.can
            self.bin_counts["BIN_PAPER"]   = self.paper
            self.bin_counts["BIN_TRASH"]   = self.trash

            # last_message에서 현재 아이템 파싱
            m = re.search(r"Sorting:\s*(\w+)\s*→\s*(BIN_\w+)", msg.last_message)
            if m:
                self.current_label = m.group(1)
                self.current_bin   = m.group(2)
            elif "Placed" in msg.last_message or "Failed" in msg.last_message:
                self.current_label = ""
                self.current_bin   = ""
            self.current_robot_phase = self.phase

    def add_log(self, text: str):
        with self.lock:
            if self.log_lines and self.log_lines[-1].split("] ", 1)[-1] == text:
                return
            ts = time.strftime("%H:%M:%S")
            self.log_lines.append(f"[{ts}] {text}")
            if len(self.log_lines) > 300:
                self.log_lines = self.log_lines[-300:]
