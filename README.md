# Meeting Rhythm AI (React + FastAPI)

Streamlit을 제거하고 아래 구조로 전환한 프로젝트입니다.

- `frontend/`: React + TypeScript + Vite UI
- `backend/api.py`: FastAPI 서버 (회의 상태, 분석, STT API)
- `llm_client.py`, `schemas.py`, `mock_data.py`: 기존 분석/스키마 로직 재사용

## 1) 설치

### Python
```bash
python -m pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## 2) 환경 변수

루트 `.env`:
```env
GOOGLE_API_KEY=your_key_here
GOOGLE_BASE_URL=https://generativelanguage.googleapis.com/v1beta
MODEL=gemini-2.0-flash
```

- `GOOGLE_API_KEY` 없으면 분석은 mock fallback으로 동작합니다.
- STT는 `openai-whisper` 로컬 모델을 사용합니다.

## 3) 실행

### 한 번에 실행 (권장)
루트에서:
```bash
python run_dev.py
```

PowerShell 스크립트로도 실행 가능:
```powershell
.\run-dev.ps1
```

- `Ctrl+C`를 누르면 backend/frontend 관련 자식 프로세스까지 함께 종료됩니다.

### Backend
```bash
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

### Frontend
```bash
cd frontend
npm run dev
```

기본 주소:
- Frontend: `http://127.0.0.1:5173`
- Backend API: `http://127.0.0.1:8000`

## 4) 기능 요약

- 회의 설정/전사/아젠다/분석/산출물 생성
- `틱 / 업데이트` 단일 분석 호출
- Keyword Engine (K1~K6):
  - Candidates: Top 40 후보 추출
  - Classification: K1 OBJECT / K2 OPTION / K3 CONSTRAINT / K4 CRITERION / K5 EVIDENCE / K6 ACTION
  - Scoring: `DecisionValue + EvidenceBoost`
  - Final Selection: Slot Filling (`K_core` 3~5, `K_facet` 3~8, Diversity Boost)
  - 출력: `k_core`, `k_facet`, `items(타입/점수/출현시점)`, `pipeline`
- Agenda Tracker:
  - 목적: 토픽이 아닌 "Closing 가능한 아젠다 단위" 추적
  - Candidate Generation: 2-of-3 규칙(Topic shift sustained / Collective intent / Decision slots)
  - Sub-issue Promotion: pricing/schedule/policy/owner 이슈 분리 승격
  - 출력: `agenda_candidates(PROPOSED pool)`, `agenda_vectors(agenda별 top terms)`
- Agenda FSM:
  - 상태: `PROPOSED -> ACTIVE -> CLOSING -> CLOSED`
  - 전이:
    - `PROPOSED -> ACTIVE`: 2-of-3 규칙 충족 후보 승격
    - `ACTIVE -> CLOSING`: Decision Lock triggered
    - `CLOSING -> CLOSED`: vote/agreement/final confirmation 감지
  - 출력: `agenda_state_map`, `active_agenda_id`, `agenda_events(active_agenda_id 변경 포함)`
- Drift Dampener:
  - 지표: `S45 = mean(sim(u_t, A_j))` over last 45s
  - 임계값:
    - Green `>= 0.72`: no action
    - Yellow `0.62~0.72` sustained >30s: K_core subtle glow
    - Red `< 0.62` sustained >30s: K_core focus 고정 + facet 축소
    - Re-orient `< 0.62` sustained >120s: 3초 재정렬 배너
  - 예외: 회의 시작 180초 이내 Red/Re-orient 억제(safe zone)
  - 출력: `drift_state`, `drift_ui_cues`, `drift_debug`
- DPS (Decision Progress Score, 0~1.0):
  - 가중치 공식:
    - Option coverage `0.25`
    - Constraint coverage `0.20`
    - Evidence coverage `0.20`
    - Trade-off coverage `0.20`
    - Closing readiness `0.15`
  - 입력: Keyword Engine(K2/K3/K4/K5), 현재 agenda/FSM 상태
  - 출력: `DPS_t`(매 tick 갱신), `dps_breakdown`, Decision Cockpit 퍼센트 바
- Flow Pulse (Loop/Stagnation Detection):
  - Circular stagnation 조건(AND):
    - A) Surface repetition(3min): `NoveltyRate < 0.15`
    - B) Content repetition: `ArgNovelty < 0.20`
    - C) No progress: `ΔDPS < 0.05`
  - Anchoring 예외: `K_core(OBJECT/CONSTRAINT/CRITERION)` 반복은 유효 앵커링으로 처리
  - 출력: `stagnation_flag`, `loop_state(Normal/Watching/Looping/Anchoring)`, `flow_pulse_debug`
  - 연동: `stagnation_flag=true` 시 Decision Lock trigger #2를 활성화
- Decision Lock (Closing Detection):
  - OR 트리거:
    - Stance convergence `>= 0.70`
    - Circular repetition(Flow Pulse stagnation condition)
    - Time-box breach `>= 15분`
  - 시스템 반응:
    - 3초 shared banner
    - Vote/Summary/Closing UI 활성화(no gating)
    - ACTIVE -> CLOSING 전이 이벤트 생성, closing action으로 CLOSED 유도
  - 출력: `intervention.decision_lock`, `decision_lock_debug`
- Live STT:
  - 마이크 입력 (`getUserMedia`)
  - 시스템 오디오 (`getDisplayMedia` 오디오 공유)
  - 프론트에서 2초 청크 업로드 -> 백엔드 STT -> 전사 피드 반영

## 5) 주요 API

- `GET /api/state`
- `POST /api/config`
- `POST /api/transcript/manual`
- `POST /api/analysis/tick`
- `POST /api/artifacts/{kind}`
- `POST /api/reset`
- `POST /api/stt/chunk`
