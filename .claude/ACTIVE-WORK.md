# ACTIVE WORK — concurrency registry

Single source of truth for **who is editing what, right now**, across every
concurrently open chat window. Not in git (this folder sits outside the repo's
tracked content), so it never causes commit noise.

**Reads are free. You only ever claim files you intend to WRITE.**

---

## Claim protocol (every agent, every task)

**1 — Plan before touching anything.**
Produce the plan first: what the task is, and the explicit list of files/globs
you will write to. No claim without a plan; no edit without a claim.

**2 — Read this file.** Check your list against every row in *Active claims*.

**3 — Overlap?**
- **No overlap** → go to step 4.
- **Overlap** → do NOT edit the contested files. Then, in this order:
  a. Do every part of the task that does *not* touch them, claiming only those files.
  b. Tell the user plainly: *"`worker/embed.py` is claimed by C-… since 20:15 — I've
     done X and Y; the rest is waiting on that."*
  c. Offer to re-check. Re-check on request or after finishing the rest — never
     spin in a polling loop, and never edit a claimed file "quickly".

**4 — Write your claim row**, then **immediately re-read this file**. If a
conflicting row for the same file appeared with an *earlier* start time, delete
your row and treat it as an overlap (step 3). Earlier timestamp always wins;
tie → shorter Claim ID wins.

**5 — Work.** If the task runs long, refresh `Heartbeat` whenever you re-read
this file. If scope grows to new files, claim them the same way *before* editing.

**6 — On completion (or abandonment):** move the row to *Recently released* with
a one-line outcome, and append the matching entry to
[`WORKLOG.md`](WORKLOG.md). Releasing is part of "done" — a task is not
finished while its claim is still open.

**Stale claims:** a row whose Heartbeat is >2h old is *probably* a chat that was
closed mid-task. Do not silently take it. Ask the user.

**Claim ID format:** `C-YYYYMMDD-HHMM-<slug>` — e.g. `C-20260810-1830-worker-pipeline`.
Get the time with `Get-Date -Format "yyyy-MM-dd HH:mm"`.

**Claiming a directory** is allowed with a glob (`worker/**`) when a task
genuinely rewrites a subsystem — but prefer listing files.

---

## Active claims

| Claim ID | Started | Task (one line) | Files / globs claimed | Status | Heartbeat |
|---|---|---|---|---|---|

---

## Recently released (keep last 10, newest first)

| Claim ID | Released | Task | Outcome |
|---|---|---|---|
| C-20260811-1850-arch-md | 2026-08-11 19:04 | Write the full 8-step architecture as a markdown doc | Created `docs/ARCHITECTURE.md` (Draft 2), superseding `docs/architecture.html` (kept for history, not deleted). Covers: what changed from Draft 1 and why; the three locked changes (**link-first ingest**, **native-resolution processing**, **download policy A+B**); a plain-language pixel primer explaining why shrinking-before-detecting produces confident *wrong* matches; the full 8-step pipeline (Ingest → Manifest → Process → Cluster → Publish → Review → Gallery → Search) with tech, edge cases and a free-now/paid-later column at every step; the ingest adapter pattern across Drive/Dropbox/zip/OneDrive/upload; tiled detection, alignment and EXIF rotation; two-pass clustering with the same-photo rule; the complete SQL data model including the `person_photos` materialised join and the Postgres/parquet storage split; two-tier selfie search; the ₹0 free stack incl. the Cloudflare Tunnel trick; landmines (Drive one **defused**, InsightFace licence + DPDP still open); build order; 6 open items. **Nothing committed — Adarsh reviews and pushes.** No product code written; still zero implementation. · `docs/ARCHITECTURE.md`, `.claude/WORKLOG.md` |
| C-20260810-2033-answers-dataset | 2026-08-10 20:55 | Record Devesh's answers to the 5 open questions; build a cricket-player test dataset script | Added a "Devesh's answer" callout under Q1–Q4 in `docs/architecture.html` (Q5 left open, not yet answered). Wrote `scripts/build_test_dataset.py` (stdlib-only) — resolves each player to a Wikimedia Commons category, filters out logos/flags/signatures, downloads freely-licensed photos plus a `manifest.csv` of source URL + license per image, with 429 retry-backoff. Test run on 2 players confirmed real JPEGs + correct license metadata download end to end. Added `.gitignore` so `test-data/` output never gets committed. · `docs/architecture.html`, `scripts/build_test_dataset.py`, `.gitignore` |
| C-20260810-1830-arch-doc | 2026-08-10 18:36 | Single-page architecture review doc for Adarsh + Devesh | Built `docs/architecture.html` — self-contained single file, zero JS, zero external assets, light/dark aware, GitHub-Pages-ready. 12 numbered sections: product + both user journeys, the never-store-originals rule with the 50 GB→1.1 GB table, five-box pipeline diagram, box-by-box tech choices each with alternatives + verdict, 6-table data model, step-by-step trace of one photo and one guest search, cost table in ₹ with the 90-day retention policy, the three landmines (InsightFace non-commercial weights / Drive restricted scope / DPDP) + the Vercel one, decisions locked 10 Aug incl. the stage-1-cached / stage-2-cheap split, build order, who-does-what, and 5 numbered open questions for Devesh to reply to. Verified in-browser: 12/12 TOC anchors resolve, 0 broken links, 0 horizontal page overflow at 375px and desktop, wide tables scroll inside their own containers, pipeline diagram stacks on mobile. **Nothing committed — Adarsh reviews and pushes.** No product code written; repo still has zero commits. |
