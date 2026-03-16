#!/usr/bin/env python3
"""
shake_classifier.py  ─  페트병 내용물 판정 모듈
═══════════════════════════════════════════════════════════════

[역할]
  LIFT 완료 후 페트병을 흔들어 정지 구간 J4 외력 토크 평균값으로
  물이 들어 있는지 판정한다.

[외부 인터페이스]
  do_shake_classify(robot, fb, logger) → str
    "BIN_PLASTIC_FULL"   물 있는 병  (평균 < AVG_THRESHOLD)
    "BIN_PLASTIC_EMPTY"  빈 병       (평균 >= AVG_THRESHOLD)

[샘플링 타이밍 - 핵심]
  movej(radius>0) 내부에서 spin_until_future_complete 를 호출한다.
  백그라운드 샘플링 스레드가 동시에 get_external_torque 를 호출하면
  같은 노드를 두 스레드가 spin → generator already executing 충돌.

  해결: 흔들기 동작 완전 종료(mwait) 후에만 샘플링 시작.
  물이 있으면 출렁임이 남아 평균값이 크고,
  빈 병이면 즉시 안정화되어 평균값이 작다.

[임계값 튜닝]
  shake_torque_test 패키지로 빈 병 / 물 병 각각 측정 후 결정.
  현재: 평균 >= -1.0 → 빈 병 / 평균 < -1.0 → 물 있음
"""

import threading
import time

from execute_node.constants import ACC

# ══════════════════════════════════════════════════════════════
#  흔들기 동작 파라미터
#  ★★★ 로봇을 직접 조그해서 아래 두 위치를 실측값으로 교체하세요 ★★★
# ══════════════════════════════════════════════════════════════

J_SHAKE_A    = [-90.0,  0.0, 90.0,  50.0, 90.0,  120.0]   # ← 실측값으로 교체
J_SHAKE_B    = [-90.0,  0.0, 90.0, 100.0, 90.0, -120.0]   # ← 실측값으로 교체

SHAKE_COUNT  = 4      # 왕복 횟수
SHAKE_VEL    = 300    # 흔들기 속도 (%)
SHAKE_ACC    = 300    # 흔들기 가속도 (%)
SHAKE_RADIUS = 30.0   # blending radius (mm)

HOLD_SECONDS = 1      # 흔들기 완료 후 정지 측정 시간 (초)
J4_INDEX     = 3      # get_external_torque()[3] = J4
SAMPLE_HZ    = 20     # 샘플링 주파수 (Hz)

# ── 판정 임계값 ───────────────────────────────────────────────
# shake_torque_test 실측 결과 기준
# 평균 >= AVG_THRESHOLD → 빈 병
# 평균 <  AVG_THRESHOLD → 물 있음
AVG_THRESHOLD = -1.0   # N·m


# ══════════════════════════════════════════════════════════════
#  TorqueSampler
# ══════════════════════════════════════════════════════════════

class TorqueSampler:
    """정지 구간 J4 외력 토크를 수집하고 평균을 계산한다."""

    def __init__(self, robot):
        self._robot   = robot
        self._samples = []
        self._running = False
        self._lock    = threading.Lock()
        self._thread  = None

    def start(self):
        with self._lock:
            self._samples = []
            self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        """샘플링 종료 후 분석 결과 반환.

        Returns:
            {"avg": float, "count": int}
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        return self._analyze()

    def _loop(self):
        while self._running:
            try:
                val = self._robot.get_external_torque()[J4_INDEX]
                with self._lock:
                    self._samples.append(val)
            except Exception:
                pass
            time.sleep(1.0 / SAMPLE_HZ)

    def _analyze(self) -> dict:
        with self._lock:
            data = list(self._samples)
        if not data:
            return {"avg": 0.0, "count": 0}
        return {"avg": sum(data) / len(data), "count": len(data)}


# ══════════════════════════════════════════════════════════════
#  do_shake_classify
# ══════════════════════════════════════════════════════════════

def do_shake_classify(robot, fb=None, logger=None) -> str:
    """페트병을 흔들어 내용물 유무를 판정하고 bin_id 를 반환한다.

    호출 시점 (motions.py → do_pick_place 내부):
        LIFT(approach 복귀) 완료 직후 / J_HOME 이동 전

    동작 순서:
        1. J_SHAKE_A 로 이동 (흔들기 시작 자세)
        2. A ↔ B 를 SHAKE_COUNT 회 왕복 (radius 블렌딩)
        3. J_SHAKE_B 에서 완전 정지 (mwait)
           ← 여기까지 movej 사용, 샘플링 없음
        4. TorqueSampler 시작 (movej 없는 구간)
        5. HOLD_SECONDS 초 대기
        6. TorqueSampler 종료 → 평균값 계산
        7. 평균 < AVG_THRESHOLD → 물 있음 판정

    Args:
        robot:  RobotAPI 인스턴스
        fb:     feedback 콜백 (phase: str, progress: float) → None
        logger: rclpy logger (optional)

    Returns:
        "BIN_PLASTIC_FULL"   물이 들어 있는 병
        "BIN_PLASTIC_EMPTY"  빈 병
    """
    _fb  = fb     or (lambda p, v: None)
    _log = logger

    # ── 1. 흔들기 시작 자세 이동 ──────────────────────────────
    _fb("SHAKE_READY", 52.0)
    robot.movej(J_SHAKE_A, vel=SHAKE_VEL, acc=SHAKE_ACC)
    robot.mwait()
    if _log:
        _log.info(f"[SHAKE] 준비 완료 A={J_SHAKE_A}")

    # ── 2. 흔들기 동작 ────────────────────────────────────────
    #    radius 블렌딩으로 부드럽게 왕복
    #    이 구간에서는 샘플링 없음 (spin 충돌 방지)
    if _log:
        _log.info(f"[SHAKE] 흔들기 시작 ({SHAKE_COUNT}회 왕복)")

    for i in range(SHAKE_COUNT):
        robot.movej(J_SHAKE_A, vel=SHAKE_VEL, acc=SHAKE_ACC, radius=SHAKE_RADIUS)
        robot.movej(J_SHAKE_B, vel=SHAKE_VEL, acc=SHAKE_ACC, radius=SHAKE_RADIUS)
        _fb("SHAKING", 52.0 + 4.0 * ((i + 1) / SHAKE_COUNT))

    # 마지막 완전 정지
    robot.movej(J_SHAKE_B, vel=SHAKE_VEL, acc=SHAKE_ACC)
    robot.mwait()

    # ── 3. 흔들기 완전 종료 확인 후 샘플링 시작 ──────────────
    #    movej 가 없는 구간 → spin 충돌 없음
    _fb("SHAKE_MEASURE", 58.0)
    if _log:
        _log.info(f"[SHAKE] 흔들기 완료. {HOLD_SECONDS}초 정지 측정 시작")

    sampler = TorqueSampler(robot)
    sampler.start()

    time.sleep(HOLD_SECONDS)

    # ── 4. 측정 종료 및 분석 ──────────────────────────────────
    result = sampler.stop()

    if _log:
        _log.info(
            f"[SHAKE] samples={result['count']}  "
            f"avg={result['avg']:.4f} N·m  "
            f"(기준 {AVG_THRESHOLD})"
        )

    # ── 5. 판정 ───────────────────────────────────────────────
    is_full = result["avg"] < AVG_THRESHOLD
    bin_id  = "BIN_PLASTIC_FULL" if is_full else "BIN_PLASTIC_EMPTY"

    if _log:
        _log.info(
            f"[SHAKE] 판정 → {bin_id}  "
            f"({'물 있음' if is_full else '빈 병'})"
        )

    return bin_id