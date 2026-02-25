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
