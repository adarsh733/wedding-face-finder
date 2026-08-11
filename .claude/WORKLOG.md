# WORK LOG — what actually got done

Organised by **session agenda**. Adarsh gives an agenda at the start of each
chat; the agent writes that agenda here immediately, then ticks items off as
they complete. Purpose: at month end, one scroll tells you how you got here.

Terse by design. Detail belongs in `docs/`, not here.

---

## How the agent maintains this

**At chat start**, as soon as the agenda is given — before any work — append a
session block under the current month:

```
### DD · <session title, 3–6 words>
**Agenda:** 1) <item> 2) <item> 3) <item>
```

**As each agenda item finishes**, append one line under that block:

```
- ✅ <item> — <what changed, plain language> · `file`, `file` · <state>
```

- `<state>`: `local` · `pushed` · `wip` · `verified` · `reverted` · `blocked: <why>`
- Status marks: `✅` done · `⏸` blocked/deferred · `❌` dropped (say why) ·
  `➕` unplanned work that came up mid-session
- Prefix `⚠` if something was found broken and left unfixed.
- **Never rewrite past entries.** Corrections are new lines.
- New month → new `##` heading, newest month at the top.
- If a session ends with agenda items untouched, mark them `⏸ carried to next
  session`.

---

## 2026-08

### 10 · Architecture kickoff
**Agenda:** 1) Explain the whole system end-to-end in plain language (backend, architecture, functioning, data flow) 2) Cover both perspectives — us building it, and photographers/guests using it 3) Lay out alternatives at each layer 4) Cost model for the zero-revenue starting phase 5) Agree the build order and the first decisions

- ✅ Repo scaffolding for the concurrency protocol — created `.claude/ACTIVE-WORK.md` (fresh, empty claim tables) and this `WORKLOG.md`. Repo is otherwise **empty: git init'd on `main`, zero commits, zero files.** · `.claude/*` · `wip`
- ✅ 1–4 Full architecture walkthrough delivered in chat — five-box model (Ingest → Brain → Store → Serve → Money), per-layer tech choice + alternative + price, per-wedding cost model in ₹, and a trace of one photo and one guest search through the system. Nothing built yet. · `—` · `verified`
- ⚠ Two commercial blockers surfaced, unresolved, **must be checked before any paid customer**: (a) InsightFace **pretrained model weights are non-commercial-research licensed** — the code is fine, the weights are the problem; (b) Google Drive `drive.readonly` is a **restricted scope** requiring a paid annual CASA security assessment — the dodge is Google Picker + `drive.file` scope, or browser upload for v1. Also noted: Vercel Hobby tier forbids commercial use.
- ⏸ 5 Build order proposed; awaiting Adarsh's answers on v1 ingestion route, who the paying customer is, and where the worker runs. No code written this session by design — the ask was discussion.
- ✅ 5 Decisions locked: **ingestion = browser folder upload** (Drive deferred, which also defers the restricted-scope problem); **worker = own desktop**, written as a queue puller so a rented GPU is a config change later; **no GPU and no real wedding photo sets yet** → CPU-only, and sourcing test data is now the top non-code task. · `—` · `verified`
- ➕ Review document built — `docs/architecture.html`, single self-contained file (no JS, no external assets, light/dark aware) covering all of the above in 12 numbered sections, ending with 5 numbered open questions for Devesh. Verified in-browser: all anchors resolve, no horizontal overflow at 375px or desktop. **Not committed — Adarsh reviews and pushes.** · `docs/architecture.html` · `verified`
- ➕ Consequence of no-GPU/no-data: pipeline must be **split into a cached expensive stage (detect+embed, written to disk once) and a cheap tunable stage (clustering, re-runs in seconds)**. Without this, CPU-only iteration on clustering thresholds is unworkable. Locked as an architectural requirement before any code.

### 10 · Answers to open questions + test dataset script
**Agenda:** 1) Record Devesh's answers to Q1–Q5 in `docs/architecture.html` 2) Build a script to assemble a test photo dataset 3) Commit the changes

- ✅ Q1 Hardware — answered: keep it API-based, nobody runs anything on their own machine; stays on local hardware for now per §9, reach for GPU-on-demand (rented, per-job) rather than owning one if/when local isn't enough. · `docs/architecture.html` · `local`
- ✅ Q2 Test photos — answered: bootstrap with cricket players (checked for an existing open dataset that fit better first, found none) — public figures, many photos each across different events/angles/lighting, closer to the wedding shape than studio headshots. Wrote `scripts/build_test_dataset.py`, stdlib-only, pulls freely-licensed images + a license/provenance manifest from Wikimedia Commons for a configurable player list. Test run (2 players, 3 images each) verified real JPEGs download with correct metadata; hit a 429 mid-run so added retry-with-backoff. Explicitly **not** a substitute for real wedding photos when tuning clustering thresholds (§11) — only proves detect→embed→cluster runs end to end. · `docs/architecture.html`, `scripts/build_test_dataset.py` · `verified`
- ✅ Q3 Who pays — answered: no single answer, depends on channel — photographer pays if we sell direct to them, customer (couple/family) pays if we sell to them. Both stay open. · `docs/architecture.html` · `local`
- ✅ Q4 Anything different — answered: agreed with §4 as written (R2/Postgres), no changes; confirmed §10's pipeline-before-UI order — validate clustering works before spending any time on UI. · `docs/architecture.html` · `local`
- ⏸ Q5 Time — not answered this session, carried forward.
- ➕ Added `.gitignore` (`test-data/`, `__pycache__/`, `*.pyc`) so generated/downloaded dataset images never get committed as binary blobs. · `.gitignore` · `local`
