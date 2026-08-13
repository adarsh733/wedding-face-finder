# Wedding Face Finder

A photographer finishes a wedding with 4,000 photos. Instead of dumping them in a
Drive folder for 200 relatives to scroll through, they give us one link. We give
back one link. A guest opens it on their phone, takes a selfie, and sees only the
photos they're in.

Full design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## What exists today (13 Aug 2026)

**A tool that runs on your own laptop. No website, no login, no cloud, nothing
uploaded anywhere.** That's deliberate — the hosted version is week 3–4.

Two ways to use it:

- **The console** — `python -m wff ui` opens a page in your browser. Paste a
  folder, press Start, watch it work, then look at what it found.
- **The terminal** — one command, prints progress, writes an HTML page you open.

Both do the same work. The console is the one to use.

The point of all of it is the **review page**: piles of face thumbnails, one pile
per person it thinks it found. That page is the accuracy meter — mistakes get
counted by eye. Clustering quality *is* the product, so this is where the real
work happens.

---

## Setup (about 5 minutes, plus one download)

You need **Python 3.12**. Check with `python --version`.

**1 — Get the code**

```bash
git clone https://github.com/adarsh733/wedding-face-finder.git
```

**2 — Make a private Python environment for it and install what it needs**

On Windows (PowerShell):

```powershell
cd wedding-face-finder; python -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

On macOS / Linux:

```bash
cd wedding-face-finder && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

That's it. **There is no step 3.** The two face models (~275 MB) download
themselves the first time you run anything, into `~/.wff/models/`. It happens
once, ever, and it prints its progress.

> If the download fails (corporate network, GitHub blocked), it tells you the URL
> and exactly where to unzip the two `.onnx` files by hand.

---

## Running it

### The console — start here

```bash
python -m wff ui
```

A browser tab opens at `http://127.0.0.1:8765/`. On that page:

1. **Paste a folder path** into the box — e.g. `D:/Photos/Manali` — and press Start.
2. Watch it go. Faces appear on screen as they're found, with a time estimate.
   You can stop it and restart later; it picks up where it left off.
3. When it finishes, click the run to open the **review page**: every person it
   found, as a pile of face thumbnails. Click any face to see the original photo
   with boxes drawn on it.

The first thing to do on the review page is press **"Show the risky ones"** and
then use the **review window** — it shows you one pile at a time and asks one
question (is this all the same person? **Y** / **N** / **→** to skip). Those
answers are saved and used to score accuracy properly.

### The terminal

```bash
python -m wff run ev_test01 "D:/path/to/your/photos"
```

It prints progress, then prints the path to a `report.html`. Open that file in a
browser.

The other commands, if you want the steps separately:

| Command | What it does |
|---|---|
| `python -m wff manifest <id> <folder>` | Just count what's in there. Fast, touches nothing. |
| `python -m wff process <id> <folder>` | The slow part — find and measure every face. Resumable. |
| `python -m wff cluster <id>` | The fast part — group faces into people, write the report. Re-run this as often as you like. |
| `python -m wff run <id> <folder>` | All three in order. |
| `python -m wff ui` | The console. |

`<id>` is any short name you pick for that folder, e.g. `ev_sharma01`. Reuse the
same id to keep adding to the same job.

**Why the split matters:** finding faces is expensive and cached; grouping them
is cheap and re-runnable. Once `process` has run, you can re-`cluster` the same
photos a hundred times in seconds while tuning. Never make grouping depend on
re-reading the original photos.

### Google Drive links

You can paste a Drive folder link instead of a folder on disk — but Google won't
hand over even a fully public folder to an anonymous program without an API key.
You need one, once:

The console asks for it in place when you paste a Drive link without one, and
shows you the four clicks that produce it. Or from a terminal:

```bash
python -m wff set-key AIza...your-key... --check "https://drive.google.com/drive/folders/..."
```

It's saved to `~/.wff/settings.env`, outside the repo, and checked against Google
before it's kept. The folder must be shared as **"Anyone with the link"**.

---

## What to expect

**Speed.** Roughly **2.3 seconds per photo** on a normal 4-core laptop with no
graphics card — measured on 851 real iPhone photos. So ~850 photos is about
30 minutes, and a 4,000-photo wedding is about 2.5 hours. It's resumable, so
leave it running and close the lid if you want.

Most of that time is looking at each photo five times at high resolution, which
is the whole reason it finds small faces in the back row of a group shot. You can
turn that off with `WFF_TILING=0` for a ~2.3× speedup and worse accuracy.

**Photo formats.** iPhone `.HEIC` works. JPG, PNG work. Videos, RAW files and
Google Takeout's `.json` sidecars are recognised and skipped.

**Where things land.** Everything goes in `.wff-store/wff/events/<your-id>/` in
the project folder. Nothing is uploaded anywhere, ever. To throw a job away,
delete its folder.

**Nothing is perfect yet.** The grouping thresholds were tuned on trip photos and
one real wedding folder. Expect one person to occasionally appear as two piles.
That's the known state of things, not a bug to report.

---

## Running the tests

```bash
python -m pytest tests/ -q
```

They're fast and they don't need photos or a network. If you change anything in
`wff/`, run these before you push.

---

## Tuning, if you're poking at accuracy

Every threshold in the pipeline is an environment variable — nothing is
hard-coded. They all live in [`wff/config.py`](wff/config.py) with a comment
explaining what each one costs. The ones that matter most:

| Variable | Default | What it does |
|---|---|---|
| `WFF_TILING` | `1` | Look at each photo in overlapping tiles. Off = ~2.3× faster, misses small faces. |
| `WFF_PASS1_THRESHOLD` | `0.42` | How similar two faces must be to start a group. Tight on purpose. |
| `WFF_PASS2_THRESHOLD` | `0.55` | How similar two groups must be to merge into one person. |
| `WFF_MIN_FACE_PX` | `50` | Smallest face height (in the original photo) worth keeping. |
| `WFF_MIN_FACES_PER_PERSON` | `3` | Below this it's a "leftover", not a person. Leftovers are kept, never deleted. |

Set them for one run rather than editing the file:

```bash
WFF_PASS2_THRESHOLD=0.60 python -m wff cluster ev_test01
```

The console also **measures the right threshold by itself** from evidence each
folder carries for free (two faces in one photo are never the same person), so
you usually shouldn't need to touch these.

---

## How the code is laid out

```
wff/
  ingest/     where photos come from — local folder, Google Drive
  process/    stage 1: find faces, check quality, make fingerprints
  cluster/    stage 2: group fingerprints into people
  storage/    reads and writes objects — swappable for cloud later
  web/        the console you see in the browser
  cli.py      the terminal commands
  config.py   every tunable number, in one place
tests/        216 tests
docs/         the architecture document
scripts/      one-off helpers
```

---

## Rules if you're changing things

- **Stage 1 is expensive and cached; Stage 2 is cheap and re-runnable.** Never
  make Stage 2 depend on re-reading the original photos.
- **Nothing downstream of ingest may touch a filesystem path.** Photo sources are
  handles, not paths, so swapping in Drive / Dropbox / a zip file stays a
  one-file change. There's a test that enforces this.
- **Never commit face crops or photos.** See the privacy note below.
- Adarsh reviews and pushes.

---

## Two things you must know before this touches a real client

**1 — Face data is biometric data under India's DPDP Act.** Face crops,
fingerprints and the photos themselves are regulated personal data. This repo is
public, so **no face data of any kind goes in it** — `.gitignore` keeps the
object store out. Don't work around that. Before a paying client's wedding lands
here, retention and deletion need designing properly; see Landmine #3 in the
architecture doc.

**2 — The face models are licensed for non-commercial research use.** Fine for
building, testing and demos. **Not fine the day someone pays us.** That's why
nothing outside [`wff/process/models.py`](wff/process/models.py) names them —
the detector and the recogniser sit behind our own interfaces, so swapping in a
permissively-licensed model is a one-file change. See Landmine #1.

---

## The real bottleneck

Clustering quality is the product, and it can't be tuned on imaginary photos.
**One real wedding folder unblocks more than any amount of code** — specifically
one with big group shots, where the hard case lives: a small face in the back
row, half-turned, slightly out of focus.
