"use client";

import { useMemo, useState } from "react";
import {
  actionItems,
  agendas,
  decisions,
  evidenceLog,
  meeting,
  participantRoster,
  transcript,
  type Agenda,
  type AgendaStatus,
  type ActionItem,
  type DecisionItem,
  type EvidenceItem,
  type Participant,
} from "@/lib/meetingData";

type SummaryScope = "current" | "all";

const agendaStatusClass: Record<AgendaStatus, string> = {
  "Not started": "statusChip statusChipNeutral",
  "In progress": "statusChip statusChipProgress",
  Done: "statusChip statusChipDone",
};

const actionStatusClass: Record<ActionItem["status"], string> = {
  Open: "statusChip statusChipNeutral",
  "In progress": "statusChip statusChipProgress",
  Done: "statusChip statusChipDone",
};

const agendaStatusLabel: Record<AgendaStatus, string> = {
  "Not started": "시작 전",
  "In progress": "진행 중",
  Done: "완료",
};

const actionStatusLabel: Record<ActionItem["status"], string> = {
  Open: "대기",
  "In progress": "진행 중",
  Done: "완료",
};

const decisionStatusLabel: Record<DecisionItem["finalStatus"], string> = {
  Approved: "확정",
  Pending: "보류",
  Rejected: "반려",
};

const participantStatusLabel: Record<Participant["status"], string> = {
  Speaking: "발언 중",
  Active: "참여 중",
  Listening: "청취 중",
};

const evidenceSupportLabel: Record<EvidenceItem["supports"], string> = {
  Action: "액션",
  Decision: "의사결정",
};

function agendaLabel(agenda: Agenda): string {
  return `${agenda.label}: ${agenda.title}`;
}

function decisionStatusClass(status: DecisionItem["finalStatus"]): string {
  if (status === "Approved") return "statusChip statusChipDone";
  if (status === "Pending") return "statusChip statusChipProgress";
  return "statusChip statusChipNeutral";
}

function participantStatusClass(status: Participant["status"]): string {
  if (status === "Speaking") return "statusChip statusChipProgress";
  if (status === "Active") return "statusChip statusChipDone";
  return "statusChip statusChipNeutral";
}

export default function Home() {
  const initialAgenda = agendas.find((agenda) => agenda.status === "In progress") ?? agendas[0];
  const [selectedAgendaId, setSelectedAgendaId] = useState(initialAgenda?.id ?? "");
  const [query, setQuery] = useState("");
  const [speakerFilter, setSpeakerFilter] = useState("전체");
  const [highlightRelated, setHighlightRelated] = useState(true);
  const [summaryScope, setSummaryScope] = useState<SummaryScope>("current");

  const selectedAgenda =
    agendas.find((agenda) => agenda.id === selectedAgendaId) ?? agendas[0] ?? null;

  const speakerOptions = useMemo(
    () => ["전체", ...new Set(transcript.map((utterance) => utterance.speaker))],
    [],
  );

  const agendaOverview = useMemo(() => {
    const done = agendas.filter((agenda) => agenda.status === "Done").length;
    const inProgress = agendas.filter((agenda) => agenda.status === "In progress").length;
    const notStarted = agendas.filter((agenda) => agenda.status === "Not started").length;
    const averageConfidence =
      agendas.length === 0
        ? 0
        : Math.round(
            agendas.reduce((total, agenda) => total + agenda.confidence, 0) / agendas.length,
          );
    return { done, inProgress, notStarted, averageConfidence };
  }, []);

  const selectedContext = useMemo(() => {
    if (!selectedAgenda) {
      return {
        transcriptCount: 0,
        evidenceCount: 0,
        decisionCount: 0,
        actionCount: 0,
        openActionCount: 0,
      };
    }
    const transcriptCount = transcript.filter((utterance) => utterance.agendaId === selectedAgenda.id).length;
    const evidenceCount = evidenceLog.filter((evidence) => evidence.agendaId === selectedAgenda.id).length;
    const scopedActions = actionItems.filter((action) => action.agendaId === selectedAgenda.id);
    return {
      transcriptCount,
      evidenceCount,
      decisionCount: decisions.filter((decision) => decision.agendaId === selectedAgenda.id).length,
      actionCount: scopedActions.length,
      openActionCount: scopedActions.filter((action) => action.status !== "Done").length,
    };
  }, [selectedAgenda]);

  const filteredTranscript = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return transcript.filter((utterance) => {
      const speakerMatch = speakerFilter === "전체" || utterance.speaker === speakerFilter;
      const queryMatch =
        normalizedQuery.length === 0 ||
        utterance.text.toLowerCase().includes(normalizedQuery) ||
        utterance.speaker.toLowerCase().includes(normalizedQuery) ||
        utterance.timestamp.includes(normalizedQuery);
      return speakerMatch && queryMatch;
    });
  }, [query, speakerFilter]);

  const summaryAgendas = useMemo(() => {
    if (!selectedAgenda) return [];
    if (summaryScope === "all") return agendas;
    return agendas.filter((agenda) => agenda.id === selectedAgenda.id);
  }, [selectedAgenda, summaryScope]);

  const summaryEvidence = useMemo(() => {
    if (summaryScope === "all") return [...evidenceLog].slice(-8).reverse();
    if (!selectedAgenda) return [];
    return evidenceLog.filter((evidence) => evidence.agendaId === selectedAgenda.id).slice(-8).reverse();
  }, [selectedAgenda, summaryScope]);

  const bottomAgendas = selectedAgenda ? agendas.filter((agenda) => agenda.id === selectedAgenda.id) : [];
  const bottomDecisions = decisions.filter((decision) => selectedAgenda && decision.agendaId === selectedAgenda.id);
  const bottomActions = actionItems.filter((action) => selectedAgenda && action.agendaId === selectedAgenda.id);
  const bottomEvidence = evidenceLog.filter((evidence) => selectedAgenda && evidence.agendaId === selectedAgenda.id);

  const onSelectAgenda = (agendaId: string) => {
    setSelectedAgendaId(agendaId);
    setSummaryScope("current");
  };

  const jumpToTranscript = (agendaId: string, timestamp: string) => {
    setSelectedAgendaId(agendaId);
    setQuery(timestamp);
  };

  const copySnippet = async (item: EvidenceItem) => {
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(`[${item.timestamp}] ${item.speaker}: ${item.quote}`);
    } catch {
      // Visual-only control.
    }
  };

  const renderSummaryCard = () => (
    <article className="card panelCard">
      <div className="panelHeader">
        <h3>회의 요약 (안건별)</h3>
        <span className="chip chipSoft">{bottomAgendas.length}개 안건 보기</span>
      </div>
      {bottomAgendas.length === 0 ? (
        <p className="emptyState">선택한 범위에 대한 안건 요약이 없어요.</p>
      ) : (
        <div className="accordionList">
          {bottomAgendas.map((agenda) => (
            <details key={agenda.id} open>
              <summary>
                <span>{agendaLabel(agenda)}</span>
                <span className={agendaStatusClass[agenda.status]}>{agendaStatusLabel[agenda.status]}</span>
              </summary>
              {agenda.summaryBullets.length === 0 ? (
                <p className="emptyState compact">이 안건 논의가 시작되면 요약이 보여요.</p>
              ) : (
                <ul className="bulletList">
                  {agenda.summaryBullets.map((point) => (
                    <li key={point}>{point}</li>
                  ))}
                </ul>
              )}
              <div className="callout">
                <p className="calloutLabel">권장 사항</p>
                <p>{agenda.recommendation}</p>
              </div>
            </details>
          ))}
        </div>
      )}
    </article>
  );

  const renderDecisionCard = () => (
    <article className="card panelCard">
      <div className="panelHeader">
        <h3>의사결정 결과</h3>
        <span className="chip chipSoft">{bottomDecisions.length}건 기록됨</span>
      </div>
      {bottomDecisions.length === 0 ? (
        <p className="emptyState">이 안건에는 아직 기록된 의사결정이 없어요.</p>
      ) : (
        <div className="decisionGroups">
          {bottomAgendas.map((agenda) => {
            const scopedDecisions = bottomDecisions.filter((decision) => decision.agendaId === agenda.id);
            if (scopedDecisions.length === 0) return null;
            return (
              <section key={agenda.id} className="decisionGroup">
                <h4>{agendaLabel(agenda)}</h4>
                {scopedDecisions.map((decision) => (
                  <article key={decision.id} className="decisionItem">
                    <div className="decisionRow">
                      <p className="decisionIssue">{decision.issue}</p>
                      <span className={decisionStatusClass(decision.finalStatus)}>{decisionStatusLabel[decision.finalStatus]}</span>
                    </div>
                    <p className="mutedLabel">옵션 / 의견</p>
                    <ul className="bulletList">
                      {decision.options.map((option) => (
                        <li key={option}>{option}</li>
                      ))}
                    </ul>
                    <div className="inlineMeta">
                      <span>신뢰도 {decision.confidence}%</span>
                      <div className="chipRow">
                        {decision.evidence.map((timestamp) => (
                          <button
                            key={timestamp}
                            className="chip chipInteractive"
                            type="button"
                            onClick={() => jumpToTranscript(decision.agendaId, timestamp)}
                          >
                            근거 {timestamp}
                          </button>
                        ))}
                      </div>
                    </div>
                  </article>
                ))}
              </section>
            );
          })}
        </div>
      )}
    </article>
  );

  const renderActionCard = () => (
    <article className="card panelCard">
      <div className="panelHeader">
        <h3>액션 아이템</h3>
        <span className="chip chipSoft">{bottomActions.length}건</span>
      </div>
      {bottomActions.length === 0 ? (
        <p className="emptyState">이 안건에 연결된 액션 아이템이 아직 없어요.</p>
      ) : (
        <div className="tableWrap">
          <table>
            <thead>
              <tr>
                <th>액션</th>
                <th>담당자</th>
                <th>기한</th>
                <th>상태</th>
                <th>근거</th>
              </tr>
            </thead>
            <tbody>
              {bottomActions.map((item) => (
                <tr key={item.id}>
                  <td>{item.action}</td>
                  <td>{item.owner}</td>
                  <td>{item.due}</td>
                  <td><span className={actionStatusClass[item.status]}>{actionStatusLabel[item.status]}</span></td>
                  <td>
                    <div className="chipRow">
                      {item.evidence.map((timestamp) => (
                        <button
                          key={timestamp}
                          className="chip chipInteractive"
                          type="button"
                          onClick={() => jumpToTranscript(item.agendaId, timestamp)}
                        >
                          {timestamp}
                        </button>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </article>
  );

  const renderEvidenceCard = () => (
    <article className="card panelCard">
      <div className="panelHeader">
        <h3>근거 로그</h3>
        <span className="chip chipSoft">{bottomEvidence.length}개 스니펫</span>
      </div>
      {bottomEvidence.length === 0 ? (
        <p className="emptyState">연결된 액션 또는 의사결정이 생기면 근거 스니펫이 표시됩니다.</p>
      ) : (
        <div className="evidenceList">
          {bottomEvidence.map((item) => (
            <article key={item.id} className="evidenceItem">
              <div className="evidenceMeta">
                <span className="chip chipSoft">{evidenceSupportLabel[item.supports]}</span>
                <span className="timestamp">{item.timestamp}</span>
                <span className="chip chipSpeaker">{item.speaker}</span>
              </div>
              <p className="quote">&quot;{item.quote}&quot;</p>
              <div className="evidenceActions">
                <button className="ghostButton" type="button" onClick={() => jumpToTranscript(item.agendaId, item.timestamp)}>
                  전사문으로 이동
                </button>
                <button className="ghostButton" type="button" onClick={() => copySnippet(item)}>
                  복사
                </button>
              </div>
            </article>
          ))}
        </div>
      )}
    </article>
  );

  return (
    <div className="workspaceShell">
      <aside className="sidebar">
        <div className="sidebarInner">
          <div>
            <p className="brand">파르체</p>
            <p className="brandSub">회의 인텔리전스</p>
          </div>
          <nav className="sidebarNav">
            <button className="navItem" type="button">대시보드</button>
            <button className="navItem navItemActive" type="button">회의 워크스페이스</button>
            <button className="navItem" type="button">리포트</button>
            <button className="navItem" type="button">팀 노트</button>
          </nav>

        </div>
      </aside>

      <main className="mainArea">
        <div className="mainInner">
          <section className="leftSection">
          <header className="pageHeader awsHeader glassStickyHeader">
            <div className="headerMain">
              <div>
                <h1>{meeting.title}</h1>
                <div className="metaRow">
                  <span>{meeting.date}</span>
                  <span>{meeting.duration}</span>
                  <span>{meeting.participants}</span>
                </div>
              </div>
              <div className="headerActions" aria-label="회의 메트릭">
                <div className="sidebarMetricList">
                  <div className="sidebarMetricRow">
                    <span>커버리지</span>
                    <strong>{agendaOverview.done}/{agendas.length}</strong>
                  </div>
                  <div className="sidebarMetricRow">
                    <span>대상</span>
                    <strong>{selectedContext.transcriptCount}</strong>
                  </div>
                  <div className="sidebarMetricRow">
                    <span>액션</span>
                    <strong>{selectedContext.openActionCount}</strong>
                  </div>
                  <div className="sidebarMetricRow">
                    <span>자신감</span>
                    <strong>{agendaOverview.averageConfidence}%</strong>
                  </div>
                </div>
              </div>
            </div>
            <div className="contextBar">
              <span className="chip chipInteractive">{selectedAgenda ? agendaLabel(selectedAgenda) : "선택된 안건 없음"}</span>
              <span>{selectedAgenda ? `${selectedAgenda.progress}% 완료` : "0% 완료"} . {meeting.elapsed}</span>
              <span className="mutedLabel">마지막 업데이트 {meeting.lastUpdated}</span>
            </div>
          </header>

          <nav className="awsTabs awsTabsSeparate" aria-label="회의 워크스페이스 탭">
            <button className="awsTab awsTabActive" type="button">개요</button>
            <button className="awsTab" type="button">전사문 검토</button>
            <button className="awsTab" type="button">안건 인사이트</button>
            <button className="awsTab" type="button">결과</button>
          </nav>
          
          <article className="card panelCard transcriptCard transcriptCardLeft">
            <div className="panelHeader">
              <h2>전사문 (전체)</h2>
              <span className="chip chipSoft">{filteredTranscript.length}개 표시</span>
            </div>
            <div className="transcriptControls transcriptControlsCompact">
              <input
                aria-label="전사문 검색"
                placeholder="전사문 검색"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
              <select aria-label="화자 필터" value={speakerFilter} onChange={(event) => setSpeakerFilter(event.target.value)}>
                {speakerOptions.map((speaker) => (
                  <option key={speaker} value={speaker}>{speaker}</option>
                ))}
              </select>
              <label className="toggleLabel">
                <input checked={highlightRelated} type="checkbox" onChange={(event) => setHighlightRelated(event.target.checked)} />
                관련 발화 강조
              </label>
            </div>
            <div className="transcriptMetaBar">
              <span className="chip chipSoft">문맥 발화: {selectedContext.transcriptCount}</span>
              <span className="chip chipSoft">연결된 의사결정: {selectedContext.decisionCount}</span>
              <span className="chip chipSoft">연결된 액션: {selectedContext.actionCount}</span>
            </div>
            <div className="transcriptList">
              {filteredTranscript.length === 0 ? (
                <p className="emptyState">현재 필터와 일치하는 발화가 없습니다. 검색어나 화자 필터를 조정해 주세요.</p>
              ) : (
                filteredTranscript.map((utterance) => {
                  const isRelated = selectedAgenda ? utterance.agendaId === selectedAgenda.id : false;
                  const shouldDim = highlightRelated && !isRelated;
                  const shouldHighlight = highlightRelated && isRelated;
                  return (
                    <article
                      key={utterance.id}
                      className={`utterance ${shouldHighlight ? "utteranceHighlight" : ""} ${shouldDim ? "utteranceDim" : ""}`}
                    >
                      <div className="utteranceMeta">
                        <span className="timestamp">{utterance.timestamp}</span>
                        <span className="chip chipSpeaker">{utterance.speaker}</span>
                      </div>
                      <p>{utterance.text}</p>
                      <div className="utteranceActions">
                        <button type="button">+ 액션</button>
                        <button type="button">+ 의사결정</button>
                        <button type="button">+ 근거</button>
                      </div>
                    </article>
                  );
                })
              )}
            </div>
          </article>
          </section>

          <section className="rightSection">
          <section className="contentSignalGrid">
            <article className="card panelCard sidebarSection">
              <div className="panelHeader tight">
                <h3>실시간 참여자</h3>
                <span className="chip chipSoft">{participantRoster.length}명 참여 중</span>
              </div>
              <div className="participantList">
                {participantRoster.map((member) => (
                  <div key={member.name} className="participantItem">
                    <div className="participantAvatar">{member.name.slice(0, 2)}</div>
                    <div>
                      <p className="participantName">{member.name}</p>
                      <p className="participantRole">{member.role}</p>
                    </div>
                    <span className={participantStatusClass(member.status)}>{participantStatusLabel[member.status]}</span>
                  </div>
                ))}
              </div>
            </article>

            <article className="card panelCard sidebarSection">
              <div className="panelHeader tight">
                <h3>실시간 시그널</h3>
                <span className="chip chipSoft">최신 3개</span>
              </div>
              <div className="signalTimeline">
                <div>
                  <span className="timelineTime">10:58</span>
                  <p>면접 패널 확정 결정이 액션 항목에 연결됨.</p>
                </div>
                <div>
                  <span className="timelineTime">10:47</span>
                  <p>보상 밴드가 보류 이슈로 표시됨.</p>
                </div>
                <div>
                  <span className="timelineTime">10:36</span>
                  <p>디자인 매니저가 직무기술서 담당을 확정함.</p>
                </div>
              </div>
            </article>
          </section>

          <section className="topGrid">
            <article className="card panelCard">
              <div className="panelHeader"><h2>안건</h2></div>
              {selectedAgenda ? (
                <section className="currentAgenda">
                  <p className="mutedLabel">현재 안건</p>
                  <h3>{agendaLabel(selectedAgenda)}</h3>
                  <div className="progressTrack"><span style={{ width: `${selectedAgenda.progress}%` }} /></div>
                  <div className="inlineMeta">
                    <span>{selectedAgenda.progress}% 완료</span>
                    <span>다음: {selectedAgenda.nextUp}</span>
                  </div>
                </section>
              ) : (
                <p className="emptyState compact">진행 중인 안건이 없어요.</p>
              )}

              <div className="agendaHealthGrid">
                <article><p className="mutedLabel">완료</p><strong>{agendaOverview.done}</strong></article>
                <article><p className="mutedLabel">진행 중</p><strong>{agendaOverview.inProgress}</strong></article>
                <article><p className="mutedLabel">시작 전</p><strong>{agendaOverview.notStarted}</strong></article>
              </div>

              <div className="agendaList">
                {agendas.map((agenda) => (
                  <button
                    key={agenda.id}
                    className={`agendaItem ${agenda.id === selectedAgendaId ? "agendaItemSelected" : ""}`}
                    type="button"
                    onClick={() => onSelectAgenda(agenda.id)}
                  >
                    <div>
                      <p className="agendaTitle">{agendaLabel(agenda)}</p>
                      <p className="mutedLabel">신뢰도 {agenda.confidence}%</p>
                    </div>
                    <span className={agendaStatusClass[agenda.status]}>{agendaStatusLabel[agenda.status]}</span>
                  </button>
                ))}
              </div>

              <div className="panelActions">
                <button type="button">추출 다시 실행</button>
                <button type="button">안건 편집</button>
                <button type="button">전사문으로 이동</button>
              </div>
            </article>

            <article className="card panelCard summaryCard">
              <div className="panelHeader">
                <h2>안건 요약</h2>
                <div className="segmented">
                  <button className={summaryScope === "current" ? "active" : ""} type="button" onClick={() => setSummaryScope("current")}>현재 안건</button>
                  <button className={summaryScope === "all" ? "active" : ""} type="button" onClick={() => setSummaryScope("all")}>전체</button>
                </div>
              </div>

              <div className="summarySignals">
                <article><p className="mutedLabel">신뢰도</p><strong>{selectedAgenda?.confidence ?? 0}%</strong></article>
                <article><p className="mutedLabel">의사결정</p><strong>{selectedContext.decisionCount}</strong></article>
                <article><p className="mutedLabel">근거</p><strong>{selectedContext.evidenceCount}</strong></article>
              </div>

              {summaryAgendas.length === 0 ? (
                <p className="emptyState">안건이 정리되면 요약이 보여요.</p>
              ) : (
                <div className="summarySections">
                  {summaryAgendas.map((agenda) => (
                    <section key={agenda.id} className="summaryBlock">
                      <h3>{agendaLabel(agenda)}</h3>
                      <div className="summaryGrid">
                        <div>
                          <p className="mutedLabel">핵심 포인트</p>
                          {agenda.keyPoints.length === 0 ? <p className="emptyState compact">아직 핵심 포인트가 없습니다.</p> : <ul className="bulletList">{agenda.keyPoints.map((point) => <li key={point}>{point}</li>)}</ul>}
                        </div>
                        <div>
                          <p className="mutedLabel">리스크</p>
                          {agenda.risks.length === 0 ? <p className="emptyState compact">기록된 리스크가 없습니다.</p> : <ul className="bulletList">{agenda.risks.map((risk) => <li key={risk}>{risk}</li>)}</ul>}
                        </div>
                        <div>
                          <p className="mutedLabel">현재까지의 의사결정</p>
                          {agenda.decisionSoFar.length === 0 ? <p className="emptyState compact">아직 의사결정이 없습니다.</p> : <ul className="bulletList">{agenda.decisionSoFar.map((decisionPoint) => <li key={decisionPoint}>{decisionPoint}</li>)}</ul>}
                        </div>
                        <div>
                          <p className="mutedLabel">다음 질문</p>
                          {agenda.nextQuestions.length === 0 ? <p className="emptyState compact">열린 질문이 없습니다.</p> : <ul className="bulletList">{agenda.nextQuestions.map((question) => <li key={question}>{question}</li>)}</ul>}
                        </div>
                      </div>
                      <div className="inlineMeta">
                        <span>신뢰도 {agenda.confidence}%</span>
                        <span>업데이트 {agenda.lastUpdated}</span>
                      </div>
                    </section>
                  ))}
                </div>
              )}

              <section className="summaryEvidence">
                <div className="panelHeader tight">
                  <h3>관련 근거</h3>
                  <span className="chip chipSoft">{summaryEvidence.length}개 링크</span>
                </div>
                {summaryEvidence.length === 0 ? (
                  <p className="emptyState compact">이 안건의 근거 스니펫이 아직 없어요.</p>
                ) : (
                  <div className="miniEvidenceList">
                    {summaryEvidence.slice(0, 5).map((item) => (
                      <button key={item.id} className="miniEvidence" type="button" onClick={() => jumpToTranscript(item.agendaId, item.timestamp)}>
                        <span className="timestamp">{item.timestamp}</span>
                        <span className="chip chipSpeaker">{item.speaker}</span>
                        <p>{item.quote}</p>
                      </button>
                    ))}
                  </div>
                )}
              </section>
            </article>
          </section>

          <div className="bottomFilter">
            <span className="chip chipInteractive">필터 기준: {selectedAgenda ? agendaLabel(selectedAgenda) : "없음"}</span>
            <span className="mutedLabel">하단 섹션은 선택된 안건과 동기화돼요.</span>
          </div>

          <section className="bottomDesktop">
            <div className="stackColumn">{renderSummaryCard()}{renderDecisionCard()}</div>
            <div className="stackColumn">{renderActionCard()}{renderEvidenceCard()}</div>
          </section>
          </section>
        </div>
      </main>
    </div>
  );
}
