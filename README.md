# ♻️ RODAJU — 재활용 분류 로봇 시스템

ROS 2 기반의 지능형 재활용 분류 자동화 시스템입니다.  
DSR m0609 로봇암 + RealSense 깊이 카메라 + YOLO 객체 인식 + 음성 명령을 통합하여 페트병·캔·종이컵을 자동으로 분류합니다.

---

## 📐 시스템 아키텍처

```
[카메라]          [음성 마이크]
   │                   │
[vision_node]   [voice_command_node]
   │   (YOLO 인식)      │ (Whisper STT + GPT-4o 파싱)
   └──────────┬─────────┘
              ▼
       [manager_node]  ← /recycle/command (SortCommand)
       (중앙 제어·로직)
              │
       [exec_client]
              │  (PickPlace Action)
              ▼
      [m0609_exec_node]
      (DSR 로봇 + 그리퍼 제어)
              │
      ┌───────┴────────┐
   [RobotAPI]     [GripperAPI]
   (DSR_ROBOT2)   (OnRobot RG2)
```

### 웹 대시보드

```
[ui_node]
   ├── ROS 구독: /recycle/response, /recycle/vision/preview
   ├── ROS 발행: /recycle/ui/command
   └── Flask 웹서버 → http://localhost:5000
```

---

## 🗂 노드 구성

| 노드 | 파일 | 역할 |
|------|------|------|
| `vision_node` | `vision_node.py` | YOLO 추론, 3D 좌표 변환, 감지 결과 발행 |
| `manager_node` | `manager_node.py` | 전체 시나리오 제어, 우선순위·제외 정책 관리 |
| `m0609_exec_node` | `m0609_exec_node.py` | PickPlace 액션 서버, 로봇 동작 수행 |
| `voice_command_node` | `stt.py` + `wakeup.py` + `llm_parser.py` | 웨이크업 감지 → STT → LLM 명령 파싱 |
| `ui_node` | `ui_node.py` + `web_server.py` | 웹 대시보드 제공 |

---

## 📡 토픽 / 액션 인터페이스

| 종류 | 이름 | 타입 | 방향 |
|------|------|------|------|
| Topic | `/recycle/vision/detections` | `Detections2D` | vision → manager |
| Topic | `/recycle/vision/preview` | `CompressedImage` | vision → ui |
| Topic | `/recycle/vision/reset` | `Empty` | manager → vision |
| Topic | `/recycle/command` | `SortCommand` | 외부 → manager |
| Topic | `/recycle/ui/command` | `SortCommand` | ui → manager |
| Topic | `/recycle/response` | `SystemStatus` | manager → ui |
| Action | `/recycle/exec/pick_place` | `PickPlace` | manager → exec |

### 커스텀 메시지

**`Detection2D`**
```
int32 id, string label, float32 confidence
int32 x, y, w, h, cx, cy
bool has_3d, float32 x_m, y_m, z_m
float32 angle_deg
```

**`SortCommand`**
```
string cmd, mode
string[] priority_order
uint8 exclude_mask
string raw_text
```

**`SystemStatus`**
```
string state, mode
string[] priority_order
uint8 exclude_mask
int32 processed_total, processed_plastic, processed_can, processed_paper, processed_trash
string last_message
float32 progress
```

---

## 🔄 동작 시나리오 흐름

```
STANDBY → SWEEP → SORTING ↔ PAUSED → DONE → STANDBY
```

| 페이즈 | 내용 |
|--------|------|
| **STANDBY** | 홈 위치 대기 |
| **SWEEP** | 빗자루로 테이블 위 쓰레기를 모으는 훑기 동작 |
| **SORTING** | YOLO 감지 → Pick & Place 분류 반복 |
| **PAUSED** | 현재 작업 완료 후 일시정지 |
| **DONE** | 테이블 정리(`CLEAN_DESK`) 후 홈 복귀 |

---

## 🎙️ 음성 명령어

| 발화 예시 | 명령(cmd) | 설명 |
|-----------|-----------|------|
| `"스윕해"`, `"훑어줘"`, `"쓰레기 모아"` | `SWEEP` | 테이블 훑기 시작 |
| `"분류해"`, `"분류 시작해"` | `START` | 분류 시작 |
| `"캔부터 분류해줘"` | `START` | 캔 우선 분류 |
| `"캔이랑 종이컵만 분류해"` | `START` | 플라스틱 제외하고 분류 |
| `"잠깐 멈춰"` | `PAUSE` | 일시정지 |
| `"다시 시작해"` | `RESUME` | 재개 |
| `"그만해"` | `STOP` | 작업 종료 후 홈 복귀 |

---

## 🗑️ 분류 카테고리

| 카테고리 | 빈 ID | YOLO 레이블 | 용량 |
|----------|-------|-------------|------|
| 페트병 | `BIN_PLASTIC` | `pet`, `bottle`, `plastic_bottle`, `water_bottle` | 20개 |
| 캔 | `BIN_CAN` | `can`, `metal`, `aluminum` | 15개 |
| 종이컵 | `BIN_PAPER` | `paper_cup`, `paper`, `cup` | 25개 |

---

## 📁 프로젝트 구조

```
rodaju/
├── execute_node/
│   ├── m0609_exec_node.py      # PickPlace 액션 서버
│   ├── motions.py              # 로봇 동작 시퀀스 (do_sweep, do_pick_place 등)
│   ├── robot_api.py            # DSR RobotAPI / GripperAPI 래퍼
│   ├── onrobot.py              # OnRobot RG2 Modbus 드라이버
│   ├── coord_transform.py      # 카메라→로봇 좌표 변환
│   └── constants.py            # 위치 상수, 빈 위치, 그리퍼 파라미터
│
├── manager_node/
│   ├── manager_node.py         # 중앙 제어 노드
│   ├── exec_client.py          # PickPlace 액션 클라이언트 래퍼
│   ├── bin_status.py           # 분류함 용량 관리
│   └── constants.py            # Phase/SystemState Enum, LABEL_TO_BIN 매핑
│
├── vision_node/
│   ├── vision_node.py          # YOLO 추론 및 감지 발행
│   └── depth_utils.py          # pixel_to_3d / obb_to_3d 변환 유틸리티
│
├── voice_command_node/
│   ├── stt.py                  # OpenAI Whisper 기반 STT
│   ├── wakeup.py               # openwakeword 웨이크업 감지
│   ├── MicController.py        # PyAudio 마이크 컨트롤러
│   └── llm_parser.py           # GPT-4o 기반 명령어 파서
│
├── UI_node/
│   ├── ui_node.py              # ROS UI 노드
│   ├── ui_state.py             # 공유 상태 관리
│   └── web_server.py           # Flask 웹 대시보드 서버
│
├── recycle_interfaces/
│   ├── msg/
│   │   ├── Detection2D.msg
│   │   ├── Detections2D.msg
│   │   ├── SortCommand.msg
│   │   └── SystemStatus.msg
│   └── action/
│       └── PickPlace.action
│
└── launch/
    └── rodaju_launch.py        # 전체 시스템 런치 파일
```

---

## ⚙️ 설치 및 실행

### 사전 요구 사항

아래 항목이 미리 설치·설정되어 있어야 합니다.

| 항목 | 버전 / 비고 |
|------|-------------|
| ROS 2 | Humble 또는 Iron |
| Python | 3.10 이상 |
| DSR 로봇 SDK | `DSR_ROBOT2`, `DR_init` (도슨트 로봇 공식 SDK) |
| Intel RealSense SDK | 2.0 이상 + `realsense2_camera` ROS 패키지 |
| OnRobot RG2 그리퍼 | Modbus TCP 연결 (`192.168.1.1:502`) |
| Git | 2.x 이상 |

---

### 1단계 — 저장소 클론

```bash
# ROS 2 워크스페이스로 이동 (없으면 생성)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# main 브랜치 클론
git clone https://github.com/ddiw/rodaju.git
```

---

### 2단계 — Python 의존성 설치

```bash
pip install \
  pymodbus \
  numpy scipy \
  ultralytics \
  openai langchain langchain-openai \
  openwakeword \
  sounddevice \
  pyaudio \
  flask
```

> `cv_bridge`는 ROS 2 패키지로 설치합니다.
> ```bash
> sudo apt install ros-humble-cv-bridge   # Iron이면 humble → iron
> ```

---

### 3단계 — ROS 2 패키지 빌드

```bash
cd ~/ros2_ws

# 의존 패키지 자동 설치
rosdep install --from-paths src --ignore-src -r -y

# 빌드 (recycle_interfaces 먼저 빌드하여 메시지 생성)
colcon build --symlink-install --packages-select recycle_interfaces
colcon build --symlink-install

# 환경 소싱
source install/setup.bash
```

> 이후 새 터미널을 열 때마다 아래 명령을 실행하거나 `~/.bashrc`에 추가하세요.
> ```bash
> source ~/ros2_ws/install/setup.bash
> ```

---

### 4단계 — Hand-Eye 캘리브레이션 파일 배치

카메라-로봇 간 좌표 변환에 필요한 행렬 파일을 지정 경로에 복사합니다.

```bash
cp /path/to/T_gripper2camera.npy \
  $(ros2 pkg prefix execute_node)/share/execute_node/resource/
```

> 파일이 없으면 픽셀 좌표 기반 fallback으로 동작하지만 정밀도가 낮아집니다.

---

### 5단계 — 환경 변수 설정

```bash
# OpenAI API 키 (STT · LLM 파서에 필요)
export OPENAI_API_KEY="sk-..."
```

영구 적용하려면 `~/.bashrc`에 추가합니다.

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

---

### 6단계 — 시스템 실행

#### 전체 실행 (권장)

```bash
ros2 launch rodaju rodaju_launch.py
```

#### 옵션 인자

```bash
# 음성 명령 노드 없이 실행
ros2 launch rodaju rodaju_launch.py use_voice:=false

# 웹 UI 없이 실행
ros2 launch rodaju rodaju_launch.py use_ui:=false

# 음성·UI 모두 비활성화
ros2 launch rodaju rodaju_launch.py use_voice:=false use_ui:=false
```

#### 개별 노드 실행 (디버깅용)

각 노드를 별도 터미널에서 실행합니다.

```bash
# 터미널 1 — 비전 노드
ros2 run vision_node vision

# 터미널 2 — 매니저 노드
ros2 run manager_node manager

# 터미널 3 — 실행 노드 (로봇 제어)
ros2 run execute_node execute

# 터미널 4 — UI 노드 (선택)
ros2 run UI_node ui_node
```

---

### 웹 대시보드 접속

시스템 실행 후 브라우저에서 접속합니다.

```
http://localhost:5000
```

---

## 🛠️ 주요 상수 및 튜닝 파라미터

### 로봇 속도 (`execute_node/constants.py`)

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `VELOCITY` | 100 | 일반 이동 속도 |
| `VELOCITY_SLOW` | 50 | 정밀 동작 속도 |
| `ACC` | 60 | 가속도 |

### Pick & Place (`execute_node/constants.py`)

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `APPROACH_Z_OFFSET` | 80 mm | 접근 높이 오프셋 |
| `PICK_Z_OFFSET` | -40 mm | 파지 높이 오프셋 |
| `MAX_GRASP_RETRIES` | 3 | 파지 재시도 횟수 |

### 비전 (`vision_node.py` 파라미터)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `publish_rate` | 10.0 Hz | 감지 발행 주기 |
| `conf_threshold` | 0.70 | YOLO 신뢰도 임계값 |
| `depth_scale` | 0.001 | depth 단위 변환 계수 (mm→m) |
| `depth_roi_radius` | 3 px | depth 샘플링 ROI 반경 |

---

## 🔧 핵심 모듈 설명

### `coord_transform.py` — 좌표 변환

- `cam_to_robot()`: Hand-Eye 캘리브레이션 행렬을 사용해 카메라 좌표(m)를 로봇 베이스 좌표(mm)로 변환
- `pixel_estimate()`: 캘리브레이션 행렬이 없을 때 픽셀 좌표 기반 fallback 추정

### `depth_utils.py` — 3D 좌표 추출

- `pixel_to_3d()`: 중심 픽셀 + ROI depth 중앙값 → 3D 좌표
- `obb_to_3d()`: OBB(Oriented Bounding Box) 장축 슬라이싱으로 최적 그립 포인트 계산

### `motions.py` — 동작 시퀀스

- `do_sweep()`: 빗자루 픽업 → GATHER_STEPS 순서로 훑기 → 빗자루 반납 → 홈
- `do_clean_desk()`: CLEAN_STEPS로 테이블 끝까지 쓸어내기
- `do_pick_place()`: 접근 → 파지(재시도) → 분류함 이동 → 투입 → 홈 복귀

### `bin_status.py` — 분류함 관리

- 각 분류함의 현재 개수, 잔여 용량, 포화율(%) 추적
- 분류함 만석 시 해당 카테고리 스킵

---

## 📊 시스템 상태 모니터링

`/recycle/response` 토픽을 통해 실시간으로 다음 정보를 확인할 수 있습니다:

```bash
ros2 topic echo /recycle/response
```

- 현재 시스템 상태 (`IDLE` / `RUNNING` / `PAUSED` / `STOPPED` / `ERROR`)
- 현재 페이즈 (`STANDBY` / `SWEEP` / `SORTING` / `PAUSED` / `DONE`)
- 분류 통계 (총계, 플라스틱, 캔, 종이)
- 현재 작업 메시지 및 진행률

---

## 🐛 트러블슈팅

| 증상 | 확인 사항 |
|------|-----------|
| 로봇 연결 실패 | DSR SDK 및 `DR_init` 설정 확인, 로봇 IP 연결 상태 확인 |
| 그리퍼 미동작 | OnRobot IP(`192.168.1.1`), 포트(`502`) 및 `pymodbus` 설치 확인 |
| YOLO 인식 안 됨 | `ultralytics` 설치 및 모델 파일 경로 확인 |
| 3D 좌표 오류 | Hand-Eye 캘리브레이션 파일(`T_gripper2camera.npy`) 경로 확인 |
| 음성 인식 안 됨 | `OPENAI_API_KEY` 환경변수 및 마이크 장치 인덱스 확인 |
| 웹 대시보드 접속 불가 | 포트 5000 방화벽 설정 및 `flask` 설치 확인 |

---

## 📄 라이선스

본 프로젝트는 내부 연구·개발 목적으로 작성되었습니다.
