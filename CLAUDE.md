# Wedding Face Finder — instructions for Claude

## Talk to Adarsh in plain language. This is rule #1.

**Adarsh is building software for the first time.** He is not a programmer. Jargon,
long technical explanations, and questions phrased in engineering terms go straight
over his head — and when that happens we both get confused, he can't give a useful
answer, and we burn tokens going in circles.

So:

- **Explain things the way you'd explain them to a smart friend who doesn't code.**
  Use everyday words. If a technical term is genuinely unavoidable, define it once,
  in half a sentence, the first time it appears.
- **When you ask him a question, ask about the product, not the plumbing.**
  Bad: *"Should we use HNSW or brute-force cosine similarity for the search index?"*
  Good: *"Search can be instant but cost money, or take 2 seconds and be free.
  Which matters more right now?"*
  If a decision has no product consequence he could feel, **don't ask — decide it
  yourself and mention it in one line.**
- **Always say what he will actually see or do.** Not "the CLI exposes a `run`
  subcommand" — but "you type one line in the terminal, wait, and a web page opens
  showing the faces it found."
- **Lead with the answer, then the reason.** Short paragraphs. No walls of text.
- He asked for this explicitly. Do not drift back into engineer-speak after a few
  messages.

He *is* sharp about product, cost, and whether something is actually useful — talk
to him at full depth there. It's only the technical vocabulary that gets in the way.

---

## What this product is, in one paragraph

A photographer finishes a wedding with 4,000 photos. Instead of dumping them in a
Drive folder for 200 relatives to scroll through, they give us one link. We give
back one link. A guest opens it on their phone, takes a selfie, and sees only the
photos they're in. Neither the photographer nor the guest should ever hear the words
*embedding*, *cluster*, or *vector*. Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What exists today (12 Aug 2026)

**A command-line tool, not an app.** No website, no login, no cloud — that's Week 3–4
on purpose. Today you run one command in a terminal and it produces an HTML page you
open in a browser. That page is the accuracy meter: piles of face thumbnails, one pile
per person, so mistakes can be counted by eye.

Run it on any folder of photos on this machine:

```bash
python -m wff run ev_test01 "D:/path/to/your/photos"
```

It prints progress, then prints the path to `report.html`. Open that file. Everything
it writes lands in `.wff-store/wff/events/<event-id>/` and nothing is uploaded anywhere.

Other commands: `manifest` (just count what's there), `process` (the slow face-finding
step, resumable), `cluster` (the fast grouping step — re-run it as often as you like).

## Ground rules for the code

- **Stage 1 is expensive and cached; Stage 2 is cheap and re-runnable.** Never make
  Stage 2 depend on re-reading the original photos.
- **Nothing downstream of ingest may touch a filesystem path.** Sources are handles,
  so Drive/Dropbox/zip stay one-file swaps. There's a test enforcing this.
- **Never commit or push.** Adarsh reviews and pushes, always.
- Face-recognition data is biometric data under India's DPDP Act. Don't design
  anything that quietly retains it. See Landmine #3 in the architecture doc.

## The real bottleneck

Clustering quality *is* the product, and it can't be tuned on imaginary photos. The
cricket-player test set proves the machinery runs but can't test the hard case
(small faces in the back row of a group shot). **One real wedding folder unblocks
more than any amount of code.**
