Create a “Meeting Workspace” desktop page using the same visual style as my current dashboard (white background, light borders, rounded cards, pill chips, minimal shadows).

FRAME & GRID
- Desktop 1440 width
- Left fixed sidebar 280px (reuse my existing sidebar style)
- Main content max width ~1080px with 32px padding
- 8px spacing system, cards radius 16px, border 1px #ECECEC, subtle shadow optional

PAGE HEADER (top of main)
- Meeting title: “UX Strategy Meet up”
- Meta row: date, duration, participants
- Right actions: Export, Share
- Under meta: a global context bar showing:
  - Current agenda chip (e.g., “Agenda 2: Hiring Plan”)
  - progress percentage + elapsed time
  - last updated time

TOP SECTION (3-column)
Left panel (Agenda extraction & progress):
- Card title: “Agendas”
- Current agenda highlight box with progress bar and “Next up”
- Agenda list items with status (Not started / In progress / Done) and confidence
- Controls: Re-run extraction, Edit agendas, Jump to transcript

Center panel (Transcript):
- Card title: “Transcript”
- Sticky controls: search, filter (All vs Current agenda), speaker filter, highlight toggle
- Scroll list of utterances: timestamp + speaker chip + text
- Hover quick actions: “+ Action”, “+ Decision”, “+ Evidence”
- When an agenda is selected, highlight related utterances

Right panel (Agenda-based summary):
- Card title: “Agenda Summary”
- Toggle: Current agenda / All
- Sections: Key points, Risks, Decision so far, Next questions
- Show confidence and last updated time

BOTTOM SECTION (2 columns, stacked cards)
Left column:
1) “Meeting Summary (by agenda)”:
- Accordion list, each agenda shows bullet summary
- Add a “Recommendations” callout per agenda (contextual advice)
2) “Decision Results”:
- List of decisions grouped by agenda
- Each decision: issue, options/opinions, final decision status, confidence
- Add evidence chips linking to transcript timestamps

Right column:
1) “Action Items”:
- Checklist/table: Action, Owner, Due, Status, evidence chips
2) “Evidence Log”:
- Chronological list: supports (Action/Decision), transcript quote snippet, timestamp, speaker chip
- Buttons: Jump to transcript, Copy

INTERACTION STATES (visual only)
- Selecting an agenda filters the transcript, summary, and bottom sections
- Evidence chips look clickable (pill chips)
- Provide empty states for panels when data is missing

RESPONSIVE
- Tablet: stack the 3 top panels into 1 column; bottom becomes tabs (Summary / Decisions / Actions / Evidence) to reduce clutter.