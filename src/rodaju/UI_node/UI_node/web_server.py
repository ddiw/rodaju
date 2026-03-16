#!/usr/bin/env python3
"""Flask 기반 웹 대시보드 서버."""

import json as _json
import logging
import os
import time

from flask import Flask, Response, jsonify, request, send_from_directory

try:
    from ament_index_python.packages import get_package_share_directory
    _PKG_SHARE = True
except ImportError:
    _PKG_SHARE = False


def run_web_server(ui, node, host: str = "0.0.0.0", port: int = 5000):
    """Flask 웹 서버 실행.

    Args:
        ui:   UIState 인스턴스
        node: UINode 인스턴스 (send_cmd 메서드 필요)
        host: 바인딩 주소
        port: 포트 번호
    """
    if _PKG_SHARE:
        try:
            html_dir = os.path.join(get_package_share_directory("UI_node"), "resource")
        except Exception:
            html_dir = os.path.join(os.path.dirname(__file__), "..", "resource")
    else:
        html_dir = os.path.join(os.path.dirname(__file__), "..", "resource")

    app = Flask(__name__)

    @app.route("/")
    def index():
        return send_from_directory(html_dir, "dashboard.html")

    @app.route("/stream")
    def stream():
        def event_gen():
            while True:
                with ui.lock:
                    data = {
                        "state"         : ui.state,
                        "phase"         : ui.phase,
                        "priority_order": ui.priority_order,
                        "exclude_mask"  : ui.exclude_mask,
                        "last_message"  : ui.last_message,
                        "progress"      : ui.progress,
                        "total"         : ui.total,
                        "plastic"       : ui.plastic,
                        "can"           : ui.can,
                        "paper"         : ui.paper,
                        "bin_counts"    : ui.bin_counts,
                        "logs"          : ui.log_lines[-30:],
                    }
                yield f"data: {_json.dumps(data)}\n\n"
                time.sleep(0.5)
        return Response(event_gen(), mimetype="text/event-stream")

    @app.route("/cmd", methods=["POST"])
    def cmd():
        body = request.get_json(force=True)
        node.send_cmd(
            cmd           = body.get("cmd", ""),
            mode          = body.get("mode", ""),
            priority_order= body.get("priority_order", []),
            exclude_mask  = body.get("exclude_mask", 0),
            raw           = body.get("raw", ""),
        )
        return jsonify({"ok": True})

    @app.route("/video_feed")
    def video_feed():
        """MJPEG 스트림: YOLO 시각화 프리뷰."""
        _BOUNDARY = b"--frame\r\n"

        def mjpeg_gen():
            while True:
                with ui.preview_lock:
                    frame = ui.latest_preview
                if frame:
                    yield (
                        _BOUNDARY
                        + b"Content-Type: image/jpeg\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                time.sleep(0.05)   # ~20 fps

        return Response(
            mjpeg_gen(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(host=host, port=port, threaded=True)
