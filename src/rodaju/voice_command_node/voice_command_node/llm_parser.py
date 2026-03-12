#!/usr/bin/env python3
"""GPT-4o 기반 명령어 파서."""

import json

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


EXCLUDE_BITS = {"PLASTIC": 1, "CAN": 2, "PAPER": 4}

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


class LLMCommandParser:
    """GPT-4o 기반 명령어 파서."""

    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            openai_api_key=openai_api_key,
        )
        self.prompt = PromptTemplate(
            input_variables=["user_input"],
            template=_PROMPT,
        )
        self.chain = self.prompt | self.llm

    def parse(self, text: str) -> dict | None:
        """STT 텍스트 → SortCommand 필드 딕셔너리. 파싱 실패 시 None 반환."""
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
                cmd = "START"
                data["mode"] = "standby"

            result = {
                "cmd"           : cmd,
                "mode"          : data.get("mode", ""),
                "priority_order": priority_order,
                "exclude_mask"  : exclude_mask,
                "raw_text"      : text,
            }

            print(f"[LLM] 파싱결과: {result}")
            return result if cmd != "NOOP" else None

        except Exception as e:
            print(f"[LLM] 파싱 실패: {e} / 원문: '{text}'")
            return None
