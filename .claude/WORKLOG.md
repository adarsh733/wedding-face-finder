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

### 11 · Resolution & storage architecture review
**Agenda:** 1) Stress-test Adarsh's suspicion that computing on compressed previews instead of originals costs accuracy 2) Evaluate a ~100 GB Google Drive (or equivalent) staging arrangement — all alternatives and outcomes 3) Work out how compute runs against that storage while it stays on local hardware, and what changes when it moves to rented GPU 4) Freeze the architecture

- ✅ 1 Adarsh's suspicion **confirmed and was a real bug** — Draft 1 shrank to 1000px *before* detecting, which drops a 90px back-row face to 15px. Worse than missed photos: undersized crops get upsampled into ArcFace, embeddings collapse toward a "generic face" region, and the system returns **confident wrong matches** — the exact failure §4 said would kill trust. Diagnosis corrected though: it's a **pipeline-ordering** problem, not a storage one. Detect+embed at native res in memory, shrink only for delivery, discard original. Cost ₹0. · `—` · `verified`
- ✅ 1 Added three things Draft 1 was missing without which "process the original" achieves nothing: **tiled detection** (SCRFD shrinks its input to 640 internally, so a full-res photo alone doesn't help), **face alignment** via 5-point landmarks, and **EXIF rotation** applied first. · `—` · `verified`
- ✅ 2 100 GB staging arrangement **rejected as unnecessary** — the accuracy fix needs zero persistent storage (photos pass through, they don't park). Also: Google Drive is the wrong tool at the same price as R2 (~₹130–210/mo) with worse API + rate limits; the quality-loss worry was about Google *Photos*, not Drive, which stores bit-exact. For the tuning phase the answer is an external drive, ₹0/month. · `—` · `verified`
- ✅ 3 Compute plan: desktop + Cloudflare Tunnel for the pilot (free public HTTPS to a home machine, no cold starts, sidesteps every free tier that sleeps or bans commercial use); rented GPU at ~₹20–30/wedding once earning. Hard rule added — **worker never references local filesystem paths, object URIs only** — or the GPU move becomes a rewrite. · `—` · `verified`
- ✅ 4 Architecture frozen and written up as **`docs/ARCHITECTURE.md`** — full 8-step flow (Ingest → Manifest → Process → Cluster → Publish → Review → Gallery → Search), each step with tech choices, edge cases and a free-now/paid-later column; plain-language pixel primer; complete SQL data model; landmines; build order. Supersedes `docs/architecture.html` (kept for history, not deleted). · `docs/ARCHITECTURE.md` · `local`
- ➕ **v1 ingestion changed from browser upload to link-first** (Adarsh confirmed 100%) — reverses a 10 Aug locked decision. Photographers already deliver by link; re-uploading 50 GB through a tab held open for 3h is a new, worse ask. Adapter pattern over Drive/Dropbox/direct-zip/OneDrive/browser-upload so the pipeline never knows the source. · `docs/ARCHITECTURE.md` · `local`
- ➕ **Google Drive restricted-scope landmine defused** — the CASA audit only applies to `drive.readonly` on a user's *account*. A public "anyone with the link" folder is readable anonymously with our own API key: no OAuth, no restricted scope, no annual audit. **⚠ Unverified — needs proving against a real public folder before we build on it.** Biggest single finding of the session. · `docs/ARCHITECTURE.md` · `blocked: needs a real Drive link to verify`
- ➕ **Download policy locked: A+B** — 2048px preview always, plus "download full size" that proxies the original live from the photographer's source via the saved `source_file_id`, falling back to the preview if the source is gone. True originals at **₹0 storage**. Only possible because of link-first ingest; also closes a hole nobody had caught — Draft 1 promised Drive-linked original downloads while locking browser upload, under which no Drive pointer exists at all. · `docs/ARCHITECTURE.md` · `local`
- ➕ **Supabase free tier would have died at wedding #2**, not 4–5 as Draft 1 claimed (~200 MB/wedding of `vector(512)` rows + index against a 500 MB cap). Fixed by splitting storage: ~147 person-centroids in Postgres (~300 KB), 32k individual fingerprints in `faces.parquet` on R2 (33 MB float16) → ~5 MB/wedding, 40+ weddings free. Search becomes two-tier: centroids for the 90% case (~5 ms), brute-force numpy over the parquet as fallback (~10 ms for 32k vectors). **A dedicated vector DB is solving a problem we don't have at this scale.** · `docs/ARCHITECTURE.md` · `local`
- ➕ Clustering designed properly: **agglomerative average-link** (one tunable knob, deterministic — repeatability matters while tuning) over k-means (needs the answer up front) and DBSCAN (assumes uniform density; bride has 2,104 faces, a guest has 4). **Two-pass** — tight first into ~380 pure fragments, then merge with evidence, including the free **same-photo rule: two faces in one photo can never be the same person**. Safety rule: splitting is a mild failure, merging is a catastrophe, always err tight. Noise faces kept with `person_id = NULL`, not discarded — Tier-2 search is what finds the cousin who's in 3 photos. · `docs/ARCHITECTURE.md` · `local`
- ➕ **`embedding_model_version` stamp added to `events`/`faces`/`persons`** — without it, a future model swap makes old and new fingerprints mutually unreadable with no error: guests silently stop finding photos forever. Free now, means reprocessing every wedding if retrofitted. · `docs/ARCHITECTURE.md` · `local`
- ➕ Materialised **`person_photos`** join table so display is a single indexed read (~5 ms) with zero vector maths at request time — all the thinking happens once, overnight. Sub-album captured from source folder names (`Sangeet/`), giving "Priya's Sangeet photos" as a free feature. · `docs/ARCHITECTURE.md` · `local`
- ⚠ **The cricket-player test set cannot validate any of the resolution work** — cropped press photos, big faces filling the frame. Proves detect→embed→cluster runs; cannot test the back-row-of-a-group-photo case, which is now the main technical risk. Two real files needed: one public Drive wedding folder link, one 24 MP group shot. Both open, Adarsh sourcing.
