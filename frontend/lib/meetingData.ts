export type AgendaStatus = "Not started" | "In progress" | "Done";

export interface Agenda {
  id: string;
  label: string;
  title: string;
  edited?: boolean;
  status: AgendaStatus;
  confidence: number;
  progress: number;
  nextUp: string;
  keyPoints: string[];
  risks: string[];
  decisionSoFar: string[];
  nextQuestions: string[];
  keywords?: string[];
  summaryPointIds?: string[];
  summaryPointRanges?: string[];
  summaryBullets: string[];
  recommendation: string;
  lastUpdated: string;
}

export interface TranscriptUtterance {
  id: string;
  timestamp: string;
  speaker: string;
  text: string;
  agendaId: string;
}

export interface DecisionItem {
  id: string;
  agendaId: string;
  issue: string;
  options: string[];
  finalStatus: "Approved" | "Pending" | "Rejected";
  confidence: number;
  evidence: string[];
}

export interface ActionItem {
  id: string;
  agendaId: string;
  action: string;
  owner: string;
  due: string;
  status: "Open" | "In progress" | "Done";
  evidence: string[];
}

export interface EvidenceItem {
  id: string;
  agendaId: string;
  supports: "Action" | "Decision" | "Summary";
  targetId: string;
  targetLabel?: string;
  agendaTitle?: string;
  quote: string;
  timestamp: string;
  speaker: string;
}

export interface Participant {
  name: string;
  role: string;
  status: "Speaking" | "Active" | "Listening";
}

export const meeting = {
  title: "서비스 전략 미팅",
  date: "2026년 2월 26일",
  duration: "01:30",
  participants: "참여자 6명",
  elapsed: "42분 지남",
  lastUpdated: "오전 10:41",
};

export const participantRoster: Participant[] = [
  { name: "지민", role: "진행", status: "Speaking" },
  { name: "민수", role: "프로덕트 리드", status: "Active" },
  { name: "수아", role: "디자인 매니저", status: "Active" },
  { name: "현우", role: "사용자 리서처", status: "Listening" },
  { name: "지연", role: "운영 파트너", status: "Listening" },
  { name: "태호", role: "채용 담당", status: "Listening" },
];

export const agendas: Agenda[] = [
  {
    id: "agenda-1",
    label: "안건 1",
    title: "리서치 리뷰",
    status: "Done",
    confidence: 93,
    progress: 100,
    nextUp: "안건 2: 채용 계획",
    keyPoints: [
      "온보딩에서 사람들이 프로필 단계에서 많이 멈춘다고 해요.",
      "모바일에서는 입력 폼이 길어서 더 답답하다는 의견이 많았어요.",
      "체험 사용자들은 진행 상황이 더 잘 보였으면 좋겠다고 했어요.",
    ],
    risks: [
      "기업 고객 인터뷰 수가 아직 적어서 해석이 과할 수 있어요.",
      "현지화 이후 기준 지표가 없어서 비교가 조금 어려워요.",
    ],
    decisionSoFar: [
      "온보딩 전체 개편보다는 프로필 흐름부터 먼저 손보기로 했어요.",
      "우선 완성률 변화를 확인한 뒤에 다음 단계로 넓히기로 했어요.",
    ],
    nextQuestions: [
      "다음 주에 기업 고객 인터뷰 한 건 더 넣을 수 있을까요?",
      "전체 배포 전에 도움말 문구부터 먼저 정리해 볼까요?",
    ],
    summaryBullets: [
      "첫 진입 구간에서 막히는 지점 두 개를 확실히 확인했어요.",
      "전체 재설계 대신 프로필 완성 경험을 먼저 개선하기로 했어요.",
      "효과 확인을 위해 2주 안에 이벤트 추적도 같이 붙이기로 했어요.",
    ],
    recommendation:
      "디자인 수정이 들어갈 때 측정 이벤트도 같이 붙이면 판단이 훨씬 쉬워져요.",
    lastUpdated: "오전 10:05",
  },
  {
    id: "agenda-2",
    label: "안건 2",
    title: "채용 계획",
    status: "In progress",
    confidence: 88,
    progress: 62,
    nextUp: "안건 3: 리서치 운영 도구",
    keyPoints: [
      "2분기 일정 맞추려면 프로덕트 디자이너 한 명은 바로 필요해요.",
      "면접에는 디자인 크리틱이랑 우선순위 케이스를 같이 넣기로 했어요.",
      "패널 피드백은 24시간 안에 주는 걸 기본으로 맞추면 좋겠어요.",
      "재무 승인만 나면 바로 공고 올릴 수 있게 준비하자는 분위기예요.",
    ],
    risks: [
      "예산 승인 늦어지면 채용 일정이 바로 밀릴 수 있어요.",
      "패널 가용 시간이 부족해서 오퍼까지 오래 걸릴 수도 있어요.",
      "보상 밴드 조정이 늦어지면 공고 시점이 흔들릴 수 있어요.",
    ],
    decisionSoFar: [
      "이번에는 한 명 먼저 채용하고 두 번째 포지션은 3분기에 다시 보기로 했어요.",
      "시작일이 밀리면 계약 인력으로 잠깐 메우는 안도 열어 두기로 했어요.",
    ],
    nextQuestions: [
      "정규 채용 전까지 계약 인력으로 공백 메우는 게 괜찮을까요?",
      "보상 밴드 업데이트를 먼저 끝내고 공고를 올릴까요?",
      "리퍼럴 캠페인으로 이번 주 안에 후보 풀을 확보할 수 있을까요?",
    ],
    summaryBullets: [
      "디자이너 한 명은 지금 바로 필요하다는 데 다들 의견이 모였어요.",
      "재무 확인되면 다음 주 월요일부터 소싱 시작하기로 했어요.",
      "면접 패널 구성과 평가 기준은 지금 방향으로 가기로 정리됐어요.",
      "두 번째 채용은 3분기 계획 점검 시점에서 다시 판단하기로 했어요.",
    ],
    recommendation:
      "패널 슬롯부터 먼저 잡아 두면 채용 속도 떨어지는 걸 꽤 막을 수 있어요.",
    lastUpdated: "오전 10:39",
  },
  {
    id: "agenda-3",
    label: "안건 3",
    title: "리서치 운영 도구",
    status: "Not started",
    confidence: 71,
    progress: 0,
    nextUp: "안건 4: 출시 체크리스트",
    keyPoints: [
      "도구 후보는 노코드 하나, 연동 중심 하나로 압축해 둔 상태예요.",
      "기존 동의 처리 방식은 법무 확인이 먼저 필요해요.",
      "마이그레이션 인력과 일정은 아직 계산이 덜 된 상태예요.",
    ],
    risks: [
      "파일럿 데이터 매핑 전에는 실제 이전 공수를 정확히 잡기 어려워요.",
      "구매 절차가 길어지면 2분기 일정과 충돌할 수 있어요.",
    ],
    decisionSoFar: ["아직 이 안건에서 확정된 결정은 없어요."],
    nextQuestions: [
      "구매 일정이 2분기 롤아웃 목표 안에 들어올 수 있을까요?",
      "도구 파일럿이랑 법무 검토를 동시에 돌릴 수 있을까요?",
    ],
    summaryBullets: [
      "회의 후반에 따로 시간을 잡아서 이야기할 예정이에요.",
      "지금은 법무 확인이 가장 큰 선행 과제예요.",
    ],
    recommendation:
      "다음 회의 전까지 비용이랑 구축 난이도 비교표를 한 장으로 정리해 오면 좋아요.",
    lastUpdated: "오전 10:41",
  },
  {
    id: "agenda-4",
    label: "안건 4",
    title: "출시 체크리스트",
    status: "Not started",
    confidence: 64,
    progress: 0,
    nextUp: "마무리 및 담당자 확인",
    keyPoints: [],
    risks: [],
    decisionSoFar: [],
    nextQuestions: [],
    summaryBullets: [],
    recommendation:
      "아직 정리된 내용이 없어요. 논의 시작하면 이 영역이 채워질 거예요.",
    lastUpdated: "오전 10:41",
  },
];

export const transcript: TranscriptUtterance[] = [
  {
    id: "utt-1",
    timestamp: "09:58",
    speaker: "지민",
    text: "최근 인터뷰 보니까 다들 프로필 작성에서 한번씩 멈추더라고요.",
    agendaId: "agenda-1",
  },
  {
    id: "utt-2",
    timestamp: "10:01",
    speaker: "민수",
    text: "필수 항목만 좀 줄여도 두 스프린트 안에는 개선 볼 수 있을 것 같아요.",
    agendaId: "agenda-1",
  },
  {
    id: "utt-3",
    timestamp: "10:04",
    speaker: "수아",
    text: "저도 전체를 갈아엎기보다 필드부터 줄이는 게 맞아 보여요.",
    agendaId: "agenda-1",
  },
  {
    id: "utt-4",
    timestamp: "10:12",
    speaker: "현우",
    text: "지금 디자이너 한 명이 온보딩이랑 리텐션을 같이 보고 있어서 너무 타이트해요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-5",
    timestamp: "10:14",
    speaker: "지민",
    text: "이번 분기에 한 명만 더 합류해도 병목은 꽤 풀릴 거예요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-6",
    timestamp: "10:17",
    speaker: "민수",
    text: "재무 승인 월요일쯤 날 것 같아서 공고 준비는 지금 시작해도 되겠어요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-7",
    timestamp: "10:21",
    speaker: "수아",
    text: "면접은 크리틱이랑 우선순위 케이스 하나는 꼭 넣었으면 해요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-8",
    timestamp: "10:23",
    speaker: "현우",
    text: "3월에는 패널 일정이 빡빡해서 이번 주 안에 슬롯은 잡아야 해요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-9",
    timestamp: "10:27",
    speaker: "지연",
    text: "리서치 도구 쪽은 법무 검토가 끝나야 다음으로 넘어갈 수 있어요.",
    agendaId: "agenda-3",
  },
  {
    id: "utt-10",
    timestamp: "10:30",
    speaker: "지민",
    text: "비용 비교표 나오기 전엔 도구 결정은 잠깐 보류하는 게 낫겠어요.",
    agendaId: "agenda-3",
  },
  {
    id: "utt-11",
    timestamp: "10:33",
    speaker: "민수",
    text: "채용 진행되는 동안엔 계약 인력으로 버틸 수 있는지도 같이 보죠.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-12",
    timestamp: "10:36",
    speaker: "수아",
    text: "오늘 안으로 역할 설명서 초안은 제가 만들어 볼게요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-13",
    timestamp: "10:39",
    speaker: "현우",
    text: "두 번째 포지션은 3분기 계획 점검 때 다시 보죠.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-14",
    timestamp: "10:40",
    speaker: "지민",
    text: "그럼 면접관 담당자는 지금 여기서 바로 정하면 어떨까요?",
    agendaId: "agenda-2",
  },
  {
    id: "utt-15",
    timestamp: "10:42",
    speaker: "민수",
    text: "재무랑 채용 요청 등록은 제가 맡아서 진행할게요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-16",
    timestamp: "10:43",
    speaker: "수아",
    text: "채용 채널에 역할 설명서 초안이랑 평가표 템플릿 올려둘게요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-17",
    timestamp: "10:45",
    speaker: "지연",
    text: "정규 채용 전까진 리서치 운영 쪽은 계약 인력으로 커버 가능해요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-18",
    timestamp: "10:47",
    speaker: "현우",
    text: "보상 밴드는 피플옵스 확인이 남아서 공고 전에 마무리해야 해요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-19",
    timestamp: "10:49",
    speaker: "지민",
    text: "다음 주 화요일에 중간 점검하고 그전까지는 매일 진행 상황 공유해요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-20",
    timestamp: "10:52",
    speaker: "수아",
    text: "초반 소싱은 리퍼럴이랑 포트폴리오 커뮤니티부터 여는 게 좋겠어요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-21",
    timestamp: "10:54",
    speaker: "민수",
    text: "역할 설명서 초안만 있으면 내부 리퍼럴 캠페인은 내일부터 시작할 수 있어요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-22",
    timestamp: "10:56",
    speaker: "현우",
    text: "후보자 이탈 줄이려면 패널 피드백은 24시간 안에 꼭 끝내야 해요.",
    agendaId: "agenda-2",
  },
  {
    id: "utt-23",
    timestamp: "10:58",
    speaker: "지민",
    text: "일정 지키려면 최종 면접 패널은 금요일까지 확정해 두죠.",
    agendaId: "agenda-2",
  },
];

export const decisions: DecisionItem[] = [
  {
    id: "dec-1",
    agendaId: "agenda-1",
    issue: "온보딩 개선 범위",
    options: [
      "온보딩 전체를 한 번에 개편해요",
      "프로필 완성 흐름부터 먼저 개선해요",
      "리서치 더 모은 뒤에 바꿔요",
    ],
    finalStatus: "Approved",
    confidence: 91,
    evidence: ["09:58", "10:04"],
  },
  {
    id: "dec-2",
    agendaId: "agenda-2",
    issue: "디자인 채용 진행 방식",
    options: [
      "디자이너 두 명을 바로 채용해요",
      "한 명 먼저 채용하고 두 번째는 미뤄요",
      "당분간 계약 인력만 활용해요",
    ],
    finalStatus: "Approved",
    confidence: 86,
    evidence: ["10:14", "10:33", "10:39"],
  },
  {
    id: "dec-3",
    agendaId: "agenda-2",
    issue: "공고 전 보상 밴드 처리",
    options: [
      "지금 밴드로 바로 공고해요",
      "피플옵스 확인 끝나고 공고해요",
      "임시 밴드로 올리고 나중에 조정해요",
    ],
    finalStatus: "Pending",
    confidence: 78,
    evidence: ["10:47", "10:49"],
  },
];

export const actionItems: ActionItem[] = [
  {
    id: "act-1",
    agendaId: "agenda-2",
    action: "프로덕트 디자이너 역할 설명서 초안 작성",
    owner: "수아",
    due: "3월 1일",
    status: "In progress",
    evidence: ["10:36", "10:43"],
  },
  {
    id: "act-2",
    agendaId: "agenda-2",
    action: "재무 승인 확인 후 채용 요청 열기",
    owner: "민수",
    due: "3월 2일",
    status: "Open",
    evidence: ["10:17", "10:42"],
  },
  {
    id: "act-3",
    agendaId: "agenda-1",
    action: "온보딩 프로필 단계 측정 이벤트 추가",
    owner: "현우",
    due: "3월 4일",
    status: "Open",
    evidence: ["10:01"],
  },
  {
    id: "act-4",
    agendaId: "agenda-2",
    action: "면접 패널 슬롯 확정",
    owner: "지민",
    due: "2월 28일",
    status: "Open",
    evidence: ["10:40", "10:58"],
  },
  {
    id: "act-5",
    agendaId: "agenda-2",
    action: "리퍼럴 소싱 캠페인 시작",
    owner: "민수",
    due: "2월 27일",
    status: "In progress",
    evidence: ["10:52", "10:54"],
  },
  {
    id: "act-6",
    agendaId: "agenda-2",
    action: "계약 인력 대체 범위 정리",
    owner: "지연",
    due: "3월 3일",
    status: "Open",
    evidence: ["10:45"],
  },
];

export const evidenceLog: EvidenceItem[] = [
  {
    id: "ev-1",
    agendaId: "agenda-1",
    supports: "Decision",
    targetId: "dec-1",
    quote: "다들 프로필 작성에서 한번씩 멈추더라고요.",
    timestamp: "09:58",
    speaker: "지민",
  },
  {
    id: "ev-2",
    agendaId: "agenda-1",
    supports: "Decision",
    targetId: "dec-1",
    quote: "전체 개편보다 필드부터 줄이는 게 맞아 보여요.",
    timestamp: "10:04",
    speaker: "수아",
  },
  {
    id: "ev-3",
    agendaId: "agenda-2",
    supports: "Decision",
    targetId: "dec-2",
    quote: "한 명만 더 합류해도 병목은 꽤 풀릴 거예요.",
    timestamp: "10:14",
    speaker: "지민",
  },
  {
    id: "ev-4",
    agendaId: "agenda-2",
    supports: "Action",
    targetId: "act-1",
    quote: "오늘 안으로 역할 설명서 초안은 제가 만들어 볼게요.",
    timestamp: "10:36",
    speaker: "수아",
  },
  {
    id: "ev-5",
    agendaId: "agenda-2",
    supports: "Decision",
    targetId: "dec-2",
    quote: "두 번째 포지션은 3분기에 다시 보죠.",
    timestamp: "10:39",
    speaker: "현우",
  },
  {
    id: "ev-6",
    agendaId: "agenda-2",
    supports: "Action",
    targetId: "act-4",
    quote: "면접관 담당자는 지금 여기서 바로 정하면 어떨까요?",
    timestamp: "10:40",
    speaker: "지민",
  },
  {
    id: "ev-7",
    agendaId: "agenda-2",
    supports: "Action",
    targetId: "act-2",
    quote: "재무랑 채용 요청 등록은 제가 맡아서 진행할게요.",
    timestamp: "10:42",
    speaker: "민수",
  },
  {
    id: "ev-8",
    agendaId: "agenda-2",
    supports: "Decision",
    targetId: "dec-3",
    quote: "보상 밴드는 공고 전에 확인을 마무리해야 해요.",
    timestamp: "10:47",
    speaker: "현우",
  },
  {
    id: "ev-9",
    agendaId: "agenda-2",
    supports: "Action",
    targetId: "act-5",
    quote: "리퍼럴 캠페인은 내일부터 시작할 수 있어요.",
    timestamp: "10:54",
    speaker: "민수",
  },
  {
    id: "ev-10",
    agendaId: "agenda-2",
    supports: "Action",
    targetId: "act-4",
    quote: "최종 면접 패널은 금요일까지 확정해 두죠.",
    timestamp: "10:58",
    speaker: "지민",
  },
];
