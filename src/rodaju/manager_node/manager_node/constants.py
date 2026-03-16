#!/usr/bin/env python3
"""매니저 노드 상수 및 열거형."""

from enum import Enum


class Phase(Enum):
    STANDBY = "STANDBY"
    SWEEP   = "SWEEP"
    SORTING = "SORTING"
    PAUSED  = "PAUSED"
    DONE    = "DONE"


class SystemState(Enum):
    IDLE    = "IDLE"
    RUNNING = "RUNNING"
    PAUSED  = "PAUSED"
    STOPPED = "STOPPED"
    ERROR   = "ERROR"


LABEL_TO_BIN: dict[str, str] = {
    "pet"           : "BIN_PLASTIC",   # 500ml 생수 페트병
    "bottle"        : "BIN_PLASTIC",
    "plastic_bottle": "BIN_PLASTIC",
    "plastic"       : "BIN_PLASTIC",
    "water_bottle"  : "BIN_PLASTIC",
    "can"           : "BIN_CAN",       # 캔
    "metal"         : "BIN_CAN",
    "aluminum"      : "BIN_CAN",
    "paper_cup"     : "BIN_PAPER",     # 종이컵
    "paper"         : "BIN_PAPER",
    "cup"           : "BIN_PAPER",
}

BIN_CAPACITY: dict[str, int] = {
    "BIN_PLASTIC": 20,
    "BIN_CAN"    : 15,
    "BIN_PAPER"  : 25,
}

EXCLUDE_BITS: dict[str, int] = {"PLASTIC": 1, "CAN": 2, "PAPER": 4}

PRIORITY_DEFAULT: dict[str, int] = {
    "PLASTIC": 1, "CAN": 2, "PAPER": 3, "NONE": 99
}
