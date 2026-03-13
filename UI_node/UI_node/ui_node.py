#!/usr/bin/env python3
"""
ui_node.py  ─  대시보드 / 로그 / 버튼 UI 노드
═══════════════════════════════════════════════════════════════

[표시 정보]
  ① 현재 시나리오 단계  (STANDBY / BAG_PICKUP / SWEEP / SORTING / PAUSED)
  ② 현재 처리 중인 쓰레기 (종류 / 투입 분류함 / 동작 단계 / 진행률)
  ③ 분류함별 남은 용량  (개수 + 게이지 바)
  ④ 처리 통계  (종류별 누적)
  ⑤ 최근 로그 (스크롤)

[키 바인딩]
  Enter / s  → START (봉투 집기부터 시작)
  p          → PAUSE / RESUME 토글
  S          → STOP
  h          → STANDBY (홈 복귀)
  1~4        → 우선순위 변경  (1=Plastic 2=Can 3=Paper 4=Trash)
  q          → 종료
"""

import threading
import time
import curses

import rclpy
from rclpy.node import Node

try:
    from recycle_interfaces.msg import SortCommand, SystemStatus
    INTERFACES_AVAILABLE = True
except ImportError:
    from std_msgs.msg import String
    INTERFACES_AVAILABLE = False

# manager_node 상수 재사용
BIN_CAPACITY: dict[str, int] = {
    "BIN_PLASTIC": 20,
    "BIN_CAN"    : 15,
    "BIN_PAPER"  : 25,
    "BIN_TRASH"  : 30,
}
BIN_LABELS: dict[str, str] = {
    "BIN_PLASTIC": "Plastic",
    "BIN_CAN"    : "Can",
    "BIN_PAPER"  : "Paper",
    "BIN_TRASH"  : "Trash",
}
BIN_COUNTER_KEY: dict[str, str] = {
    "BIN_PLASTIC": "plastic",
    "BIN_CAN"    : "can",
    "BIN_PAPER"  : "paper",
    "BIN_TRASH"  : "trash",
}


# ═══════════════════════════════════════════════════════════════
#  공유 상태
# ═══════════════════════════════════════════════════════════════

class UIState:
    def __init__(self):
        self.state             = "IDLE"
        self.phase             = "STANDBY"
        self.priority_order: list[str] = []
        self.exclude_mask      = 0
        self.last_message      = "System ready."
        self.progress          = 0.0

        # 현재 처리 아이템 (last_message 파싱 보조)
        self.current_label     = ""
        self.current_bin       = ""
        self.current_robot_phase = ""

        # 카운터
        self.total    = 0
        self.plastic  = 0
        self.can      = 0
        self.paper    = 0
        self.trash    = 0

        # 분류함 용량 (처리된 개수로 역산)
        self.bin_counts: dict[str, int] = {k: 0 for k in BIN_CAPACITY}

        self.log_lines: list[str] = []
        self.lock = threading.Lock()
        self._paused = False

    def update(self, msg):
        with self.lock:
            self.state    = msg.state
            self.phase    = msg.mode       # manager_node 는 mode 필드에 phase 값 넣음
            self.priority_order = list(msg.priority_order)
            self.exclude_mask = msg.exclude_mask
            self.last_message = msg.last_message
            self.progress = msg.progress
            self.total   = msg.processed_total
            self.plastic = msg.processed_plastic
            self.can     = msg.processed_can
            self.paper   = msg.processed_paper
            self.trash   = msg.processed_trash

            # bin_counts 동기화
            self.bin_counts["BIN_PLASTIC"] = self.plastic
            self.bin_counts["BIN_CAN"]     = self.can
            self.bin_counts["BIN_PAPER"]   = self.paper
            self.bin_counts["BIN_TRASH"]   = self.trash

            # last_message 에서 현재 아이템 파싱
            # 예: "Sorting: plastic → BIN_PLASTIC"
            import re
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
            ts = time.strftime("%H:%M:%S")
            self.log_lines.append(f"[{ts}] {text}")
            if len(self.log_lines) > 300:
                self.log_lines = self.log_lines[-300:]

    def remaining(self, bin_id: str) -> int:
        return max(0, BIN_CAPACITY.get(bin_id, 0) - self.bin_counts.get(bin_id, 0))

    def pct_full(self, bin_id: str) -> float:
        cap = BIN_CAPACITY.get(bin_id, 1)
        return min(100.0, self.bin_counts.get(bin_id, 0) / cap * 100.0)

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        return self._paused


# ═══════════════════════════════════════════════════════════════
#  ROS 노드
# ═══════════════════════════════════════════════════════════════

class UINode(Node):
    def __init__(self, ui_state: UIState):
        super().__init__("ui_node")
        self._ui = ui_state

        if INTERFACES_AVAILABLE:
            self.create_subscription(SystemStatus, "/recycle/response",
                self._status_cb, 10)
            self._cmd_pub = self.create_publisher(SortCommand, "/recycle/ui/command", 10)
        else:
            self.create_subscription(String, "/recycle/response",
                lambda m: self._ui.add_log(m.data), 10)
            self._cmd_pub = self.create_publisher(String, "/recycle/ui/command", 10)

    def _status_cb(self, msg):
        self._ui.update(msg)
        self._ui.add_log(
            f"[STATUS] {msg.state} | {msg.mode} | "
            f"total={msg.processed_total} | {msg.last_message[:60]}"
        )

    def send_cmd(self, cmd: str, mode: str = "", priority_order: list = None,
                 exclude_mask: int = 0, raw: str = ""):
        try:
            msg = SortCommand()
            msg.stamp          = self.get_clock().now().to_msg()
            msg.cmd            = cmd
            msg.mode           = mode
            msg.priority_order = priority_order or []
            msg.exclude_mask   = exclude_mask
            msg.raw_text       = raw or f"UI:{cmd}"
            self._cmd_pub.publish(msg)
            self._ui.add_log(f"[CMD] {cmd} mode={mode} priority_order={msg.priority_order}")
        except Exception as e:
            self.get_logger().error(f"Publish error: {e}")


# ═══════════════════════════════════════════════════════════════
#  curses TUI
# ═══════════════════════════════════════════════════════════════

def _gauge(value: float, width: int = 20) -> str:
    """0~100 값을 게이지 바 문자열로 변환."""
    filled = int(width * value / 100.0)
    return "█" * filled + "░" * (width - filled)

def _bin_color(pct: float, colors) -> int:
    if pct >= 90:
        return colors["red"]
    elif pct >= 70:
        return colors["yellow"]
    return colors["green"]


def run_tui(ui: UIState, node: UINode):

    def draw(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(250)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN,   -1)
        curses.init_pair(2, curses.COLOR_RED,     -1)
        curses.init_pair(3, curses.COLOR_YELLOW,  -1)
        curses.init_pair(4, curses.COLOR_CYAN,    -1)
        curses.init_pair(5, curses.COLOR_WHITE,   curses.COLOR_BLUE)
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)

        colors = {
            "green":   curses.color_pair(1),
            "red":     curses.color_pair(2),
            "yellow":  curses.color_pair(3),
            "cyan":    curses.color_pair(4),
            "header":  curses.color_pair(5),
            "magenta": curses.color_pair(6),
        }

        while rclpy.ok():
            stdscr.erase()
            H, W = stdscr.getmaxyx()

            # ── 헤더 ─────────────────────────────────────────
            title = " ♻  Recycling Robot Dashboard "
            stdscr.addstr(0, 0, title.center(W), colors["header"])

            # ── 시나리오 단계 / 시스템 상태 ───────────────────
            with ui.lock:
                state    = ui.state
                phase    = ui.phase
                priority_order = ui.priority_order
                last_msg = ui.last_message
                progress = ui.progress
                cur_lbl  = ui.current_label
                cur_bin  = ui.current_bin
                total    = ui.total
                plastic  = ui.plastic
                can      = ui.can
                paper    = ui.paper
                trash    = ui.trash
                bin_counts = dict(ui.bin_counts)
                logs     = ui.log_lines[:]

            # State 색
            sc = (colors["green"]  if state == "RUNNING"
                  else colors["red"]    if state in ("STOPPED", "ERROR")
                  else colors["yellow"] if state == "PAUSED"
                  else colors["cyan"])

            r = 2
            stdscr.addstr(r,   2, "State  : ", curses.A_BOLD)
            stdscr.addstr(state, sc | curses.A_BOLD)
            stdscr.addstr(r,  20, "Phase  : ", curses.A_BOLD)
            stdscr.addstr(phase, colors["cyan"])
            stdscr.addstr(r+1, 2, f"Priority: {' > '.join(priority_order) if priority_order else 'NONE'}")

            # ── 현재 처리 중인 아이템 ─────────────────────────
            r = 5
            stdscr.addstr(r, 2, "─── Current Task ────────────────────────────────",
                          colors["cyan"])
            r += 1
            if cur_lbl:
                stdscr.addstr(r,   2, f"  Item  : ", curses.A_BOLD)
                stdscr.addstr(f"{cur_lbl.upper()}", colors["magenta"] | curses.A_BOLD)
                stdscr.addstr(r,  20, f"  → {cur_bin}")
                r += 1
                bar_w = W - 28
                bar   = _gauge(progress, bar_w)
                stdscr.addstr(r, 2, f"  Phase : {phase:<14}  [{bar}] {progress:.0f}%")
            else:
                phase_color = colors["yellow"] if phase in ("BAG_PICKUP","SWEEP") else colors["cyan"]
                stdscr.addstr(r, 2, f"  {last_msg[:W-4]}", phase_color)
            r += 2

            # ── 분류함 용량 ───────────────────────────────────
            stdscr.addstr(r, 2, "─── Bin Capacity ────────────────────────────────",
                          colors["cyan"])
            r += 1
            gauge_w = 18
            for bin_id, label in BIN_LABELS.items():
                cnt = bin_counts.get(bin_id, 0)
                cap = BIN_CAPACITY[bin_id]
                rem = max(0, cap - cnt)
                pct = cnt / cap * 100.0
                g   = _gauge(pct, gauge_w)
                bc  = _bin_color(pct, colors)
                warn = " ⚠ FULL" if rem == 0 else (" ⚠ LOW" if rem <= 3 else "")

                stdscr.addstr(r, 2, f"  {label:<8}: ")
                stdscr.addstr(f"[{g}]", bc)
                stdscr.addstr(f" {cnt:2d}/{cap:2d}  remain={rem:2d}{warn}",
                              colors["red"] if warn else 0)
                r += 1

            # ── 처리 통계 ─────────────────────────────────────
            r += 1
            stdscr.addstr(r, 2, "─── Statistics ──────────────────────────────────",
                          colors["cyan"])
            r += 1
            stdscr.addstr(r, 2,
                f"  Total={total}  Plastic={plastic}  Can={can}  "
                f"Paper={paper}  Trash={trash}")
            r += 2

            # ── 로그 ─────────────────────────────────────────
            log_area  = H - r - 3
            stdscr.addstr(r, 2, "─── Log ─────────────────────────────────────────",
                          colors["cyan"])
            r += 1
            for line in logs[-(log_area):]:
                if r >= H - 3:
                    break
                try:
                    stdscr.addstr(r, 2, line[:W - 4])
                except curses.error:
                    pass
                r += 1

            # ── 키 가이드 ─────────────────────────────────────
            guide = (" [Enter/s]Start  [p]Pause/Resume  [S]Stop  "
                     "[h]Home  [1-4]Priority  [q]Quit ")
            try:
                stdscr.addstr(H - 2, 0, guide[:W], colors["header"])
            except curses.error:
                pass

            stdscr.refresh()

            # ── 키 처리 ───────────────────────────────────────
            key = stdscr.getch()
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('s'), ord('S') & 0x1f, 10, 13):   # s / Enter
                node.send_cmd("START", mode="sorting")
            elif key == ord('S'):
                node.send_cmd("STOP", mode="stop")
            elif key == ord('p'):
                paused = ui.toggle_pause()
                node.send_cmd("PAUSE" if paused else "RESUME",
                              mode="" if not paused else "")
            elif key == ord('h'):
                node.send_cmd("START", mode="standby")
            elif key == ord('1'):
                node.send_cmd("SET_POLICY", priority_order=["PLASTIC"])
            elif key == ord('2'):
                node.send_cmd("SET_POLICY", priority_order=["CAN"])
            elif key == ord('3'):
                node.send_cmd("SET_POLICY", priority_order=["PAPER"])
            elif key == ord('4'):
                node.send_cmd("SET_POLICY", priority_order=["TRASH"])

    curses.wrapper(draw)


# ═══════════════════════════════════════════════════════════════
#  CLI fallback
# ═══════════════════════════════════════════════════════════════

def run_cli(ui: UIState, node: UINode):
    print("CLI mode. Commands: start | stop | pause | resume | home | 1-4 | q")
    while rclpy.ok():
        try:
            c = input("CMD> ").strip().lower()
        except EOFError:
            break
        if c == "q":
            break
        elif c in ("s", "start", ""):
            node.send_cmd("START", mode="sorting")
        elif c == "stop":
            node.send_cmd("STOP")
        elif c == "pause":
            node.send_cmd("PAUSE")
        elif c == "resume":
            node.send_cmd("RESUME")
        elif c == "home":
            node.send_cmd("START", mode="standby")
        elif c == "1":
            node.send_cmd("SET_POLICY", priority_order=["PLASTIC"])
        elif c == "2":
            node.send_cmd("SET_POLICY", priority_order=["CAN"])
        elif c == "3":
            node.send_cmd("SET_POLICY", priority_order=["PAPER"])
        elif c == "4":
            node.send_cmd("SET_POLICY", priority_order=["TRASH"])
        else:
            with ui.lock:
                print(
                    f"Phase={ui.phase}  State={ui.state}  "
                    f"Total={ui.total}  "
                    f"Plastic={ui.plastic} Can={ui.can} "
                    f"Paper={ui.paper} Trash={ui.trash}"
                )
                for bid, lbl in BIN_LABELS.items():
                    print(f"  {lbl}: {ui.bin_counts.get(bid,0)}/{BIN_CAPACITY[bid]}"
                          f"  remain={ui.remaining(bid)}")


# ═══════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    ui_state = UIState()
    ui_node  = UINode(ui_state)

    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(ui_node), daemon=True)
    spin_thread.start()

    try:
        run_tui(ui_state, ui_node)
    except Exception:
        run_cli(ui_state, ui_node)
    finally:
        ui_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
