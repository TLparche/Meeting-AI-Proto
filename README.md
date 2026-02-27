# Meeting STT + Agenda MVP

실시간 회의 STT 전사 + Gemini 기반 안건/요약/의사결정/액션아이템 추출 MVP입니다.  
UI는 `front-ref/parche` 레이아웃을 기준으로 유지하고, 기능만 연결했습니다.

## 핵심 기능
- 시스템 오디오(화면 공유) 기반 실시간 STT
- 로컬 Whisper(`openai-whisper`) 전사
- Gemini LLM 연결/해제/핑
- 4개 발화마다 자동 요약 갱신(LLM 연결 시)
- 현재 소주제(안건) 추적 및 주제 전환 시 이전 안건 `완료(CLOSED)` 처리
- 안건별 요약 클릭 시 전사 위치 점프(타임스탬프 기반)
- 의사결정 결과(확정된 항목만) 정리
- 액션아이템(무엇/누가/기한/근거) 정리
- JSON 사전 업로드(테스트용) + `metadata.topic` 자동 회의 목표 반영

## 프로젝트 구조
- `backend/api.py`: FastAPI 서버 (상태/분석/STT/업로드 API)
- `frontend/`: Next.js UI
- `llm_client.py`: Gemini REST 클라이언트
- `run_dev.py`: 백엔드+프론트 동시 실행 스크립트

## 요구사항
- Python 3.10+
- Node.js 20+
- `ffmpeg` 설치 및 PATH 등록 (Whisper 오디오 디코딩 필수)
- (선택) CUDA 환경: Whisper 추론 가속

## 설치
### Python
```bash
python -m pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## 환경 변수
루트 `.env` 예시:
```env
# Gemini
GEMINI_API_KEY=your_key_here
# 또는 GOOGLE_API_KEY 사용 가능
GEMINI_MODEL=gemini-2.0-flash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta

# Whisper
WHISPER_MODEL=large

# Optional ports
BACKEND_PORT=8000
FRONTEND_PORT=5173
```

참고:
- LLM 분석은 `LLM 연결` 버튼을 눌러야 활성화됩니다.
- API 키가 없으면 LLM 연결 실패가 정상입니다(STT/업로드는 사용 가능).

## 실행
루트에서:
```bash
python run_dev.py
```

PowerShell:
```powershell
.\run-dev.ps1
```

기본 접속 주소:
- Frontend: `http://127.0.0.1:5173` (포트 충돌 시 자동 변경)
- Backend: `http://127.0.0.1:8000` (포트 충돌 시 자동 변경)

`Ctrl+C`로 종료하면 백엔드/프론트 자식 프로세스를 함께 정리합니다.

## 사용 순서 (권장)
1. 앱 실행 후 `LLM 연결` 클릭
2. 회의 목표 입력 후 `설정 저장` (선택)
3. 실시간 STT: `Start STT` 클릭 -> 브라우저 화면 공유에서 **탭 오디오 공유 허용**
4. 테스트 데이터: JSON 파일 업로드 또는 JSON 폴더 로드
5. `분석 실행` 또는 4개 발화 단위 자동 분석 결과 확인

## JSON 업로드 포맷
다음 포맷을 지원합니다(예: `dataset/DGBBA21000001.json`):
- `metadata.topic`: 회의 목표로 자동 반영
- `speaker[]`: `id`, `age`, `occupation`, `role`
- `utterance[]`: `speaker_id`, `start`, `original_form`

전사 매핑 규칙:
- 화자 라벨: `age + occupation + role` (예: `40대 기자 토론자`)
- 전사 텍스트: `original_form` 우선
- 타임스탬프: `start` 초 -> `HH:MM:SS`

## 안건/요약 동작 규칙
- LLM은 최근 발화를 받아 현재 소주제를 추론합니다.
- 진행 중 안건이 있을 때는, 기존 안건과 충분히 벗어난 경우에만 안건 전환합니다.
- 안건 전환 시 기존 안건은 `CLOSED` 처리됩니다.
- 요약은 4개 발화마다 갱신됩니다.
- 의사결정은 **확정된 내용만** 수집합니다.
- 액션아이템은 `task / owner / due / evidence`를 수집합니다.

## 주요 API
- `GET /api/health`
- `GET /api/state`
- `GET /api/llm/status`
- `POST /api/llm/connect`
- `POST /api/llm/disconnect`
- `POST /api/llm/ping`
- `POST /api/config`
- `POST /api/transcript/manual`
- `POST /api/transcript/import-json-dir`
- `POST /api/transcript/import-json-files`
- `POST /api/analysis/tick`
- `POST /api/reset`
- `POST /api/stt/chunk`

## 트러블슈팅
- `오디오 트랙이 없습니다`
  - 화면 공유 시작 시 반드시 탭 오디오 공유를 켜야 합니다.
- `Failed to load audio` / ffmpeg 관련 에러
  - `ffmpeg` 설치/환경변수(PATH) 확인
- `LLM이 연결되지 않았습니다`
  - `GEMINI_API_KEY` 설정 후 `LLM 연결` 버튼 클릭
- `Next.js dev lock` 에러
  - 남아있는 `next dev` 프로세스를 종료 후 재실행
