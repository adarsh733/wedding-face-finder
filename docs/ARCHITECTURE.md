# Wedding Face Finder — Architecture

**Draft 2 · 11 August 2026 · Adarsh & Devesh**
Status: **nothing built yet.** This document exists so we disagree before writing code rather than after.

> Supersedes `docs/architecture.html` (Draft 1, 10 Aug). Three things in Draft 1 turned out
> to be wrong — see [What changed from Draft 1](#what-changed-from-draft-1). Keep the HTML for
> history; this file is the live one.

---

## Contents

- [What we're building](#what-were-building)
- [What changed from Draft 1](#what-changed-from-draft-1)
- [Decisions locked](#decisions-locked)
- [Background: pixels, and the one number that matters](#background-pixels-and-the-one-number-that-matters)
- [The core insight: pass-through, not park](#the-core-insight-pass-through-not-park)
- [The pipeline: 8 steps](#the-pipeline-8-steps)
  - [Step 1 — Ingest](#step-1--ingest)
  - [Step 2 — Manifest](#step-2--manifest)
  - [Step 3 — Process](#step-3--process-per-photo)
  - [Step 4 — Cluster](#step-4--cluster)
  - [Steps 5 & 6 — Publish and Review](#steps-5--6--publish-and-the-photographers-review)
  - [Steps 7 & 8 — The guest, and the selfie search](#steps-7--8--the-guest-and-the-selfie-search)
- [The data model](#the-data-model)
- [The stack: free now, paid later](#the-stack-free-now-paid-later)
- [Landmines](#landmines)
- [Build order](#build-order)
- [Who does what](#who-does-what)
- [Open items](#open-items)

---

## What we're building

A photographer finishes a wedding. Instead of dumping 4,000 photos into a Drive folder and letting
200 relatives scroll for an hour, they hand us **one link**. We give them back **one link**. Any
guest opens it, takes a selfie, and instantly sees only the photos they're in.

**What the photographer experiences.** Logs in, pastes the link to the folder they were going to
send anyway, goes to sleep. Wakes up to a share link and a page saying *"4,102 photos · 147 people
found"*. Sends the link on WhatsApp. Done.

**What the aunt experiences.** Taps the WhatsApp link on her phone. Sees the full gallery. A button
says *"Find my photos"*. She allows camera, takes one selfie. Three seconds later: 37 photos, all of
her. Taps download.

> **The bar.** Neither of them ever hears the words *embedding*, *cluster* or *vector*. If either
> one has to understand anything about how it works, we've failed.

---

## What changed from Draft 1

Three corrections, in descending order of importance.

### 1. Draft 1 would have computed on shrunk photos. That was a serious accuracy bug.

Draft 1's per-photo trace shrank each photo to 1000 px **first**, then looked for faces in the shrunk
copy. That destroys the majority of the product's value and, worse, produces confident wrong matches.
Full explanation in [Background: pixels](#background-pixels-and-the-one-number-that-matters).

**Fix:** reorder four steps. Detect and fingerprint at native resolution, in memory, then make the
small copy, then discard the original. **Costs ₹0.** No extra storage, no extra service.

### 2. Version 1 ingestion is now a **link**, not a browser upload.

Draft 1 locked "photographer drags a folder into a web page." Wrong. Photographers already deliver
by link; asking them to re-upload 50 GB through a tab that must stay open for three hours is asking
them to do a new, worse thing.

Link-first also **defuses the Google Drive landmine** and **unlocks free full-size downloads**.
See [Step 1](#step-1--ingest).

### 3. The database plan would have died at wedding #2.

Draft 1 put all ~32,000 face fingerprints per wedding into Postgres — about 200 MB per wedding
against a 500 MB free tier. Fix: **person-averages in Postgres, individual fingerprints in a
parquet file on R2.** Takes it from 2 weddings to 40+ on the free tier.
See [The data model](#the-data-model).

Also added, none of which were in Draft 1: **tiled detection**, **face alignment**,
**EXIF rotation**, **two-pass clustering with the same-photo rule**, **the embedding model version
stamp**, and the **two-tier selfie search**.

---

## Decisions locked

| Decision | Chosen | Date | Consequence |
|---|---|---|---|
| **v1 ingestion** | **Link-first** (Google Drive public folder link primary; Dropbox, direct-zip, browser upload as fallbacks) | 11 Aug | Photographer does nothing they weren't already doing. Kills the OAuth restricted-scope audit. Enables free original downloads. |
| **Resolution** | **Detect + fingerprint at native resolution**, in memory. Shrink only for delivery. | 11 Aug | ~2× CPU time. ₹0 extra cost. Non-negotiable — the product doesn't work otherwise. |
| **Download policy** | **A + B** — 2048 px preview always; "download full size" proxies the original live from the photographer's source link, falling back to the preview if the source is gone | 11 Aug | Guests get true originals at **₹0 storage cost**. Only works because of link-first ingest. |
| **Where processing runs** | Our own desktop, CPU-only, as a queue puller | 10 Aug | ₹0. Moving to a rented GPU later is a config change, not a rewrite — **provided the no-local-paths rule below is respected.** |
| **Vector storage** | Person-averages in Postgres; individual fingerprints in parquet on R2 | 11 Aug | 40+ weddings on the Supabase free tier instead of 2. |
| **Object storage** | Cloudflare R2 | 10 Aug | Zero egress fees. We're an image product; this matters more than any code we'll write. |
| **Database** | Supabase (Postgres + `pgvector`) | 10 Aug | One service for users, events, payments and vectors. |
| **Web** | Next.js on Cloudflare Pages | 10 Aug | Vercel's free tier forbids commercial use; Cloudflare's permits it. |
| **Payments** | Razorpay, per event (not monthly) | 10 Aug | Photographers work event to event. A subscription asks them to pay in months they shoot nothing. |
| **Test data** | Cricket-player set (`scripts/build_test_dataset.py`) for mechanics only | 10 Aug | **Cannot test any of the resolution work** — those are cropped press photos. Real wedding photos still the top blocker. |

### The one rule that keeps the GPU migration cheap

**The worker must never reference a local filesystem path.** Every input and output is an object
URI (`r2://events/{id}/...`), even while everything runs locally. Point it at R2 (or a local MinIO
container) from day one. Break this in week 1 and moving to a rented GPU becomes a rewrite instead
of a settings change.

---

## Background: pixels, and the one number that matters

*Written for a non-specialist. Skip if you already know this.*

A digital photo is a grid of coloured dots. Each dot is a **pixel** (px). A photo from a
professional wedding camera is about **6000 dots wide × 4000 tall** — 24 million dots
("24 megapixel").

The number that decides whether this product works is not the photo's size. It is:

> **How many dots tall is a person's face inside the photo?**

Same camera, same photo: a face in a close-up might be 800 dots tall, while someone in the back row
of a 30-person group shot is 60 dots tall.

### A scale to calibrate against

| Face is this tall | What that looks like | Can a computer identify them? |
|---|---|---|
| 1500 px | Bride's close-up. You can count eyelashes. | Trivially |
| 400 px | A normal selfie | Easily |
| **112 px** | Roughly a thumbnail on a phone | **The exact minimum our model needs** |
| 50 px | You can tell it's a human. Not *who*. | Unreliable — dangerous |
| 15 px | About as tall as one line of body text | No |

**112 px is the magic number.** ArcFace, our recognition model, takes exactly a 112 × 112 square.
That's its eye. Fewer dots than that and it is guessing.

### What shrinking destroys

Shrinking 6000 → 1000 px wide throws away **5 of every 6 dots, permanently.** A photocopy at 17%,
after which we burn the original.

| Where the person is standing | Face natively | After shrinking to 1000 px | Verdict |
|---|---|---|---|
| Couple's portrait | 800 px | 133 px | survives |
| Small group of six | 300 px | 50 px | struggling |
| **Back row of a 30-person group** | 90 px | **15 px** | **gone** |
| **Dancing across the room** | 60 px | **10 px** | **gone** |

### Why this produces *wrong* matches, not just missing ones

You'd expect the computer to say *"too blurry, I can't tell."* **It never says that. It always
produces an answer.**

Squint at a crowd from across a hall — every face is a beige oval. Asked "is that Priya?", you'd
guess, and guess *confidently*, because at that size everyone genuinely does look identical. The
model does exactly this: blurry faces all get described as roughly "generic human face," so they
all land near each other.

> **The chain:** upsampled crop → the network sees smooth, detail-free pixels → the fingerprint
> drifts toward the "generic face" region → **everyone's blurry face lands near everyone else's** →
> Priya is shown a photo of a stranger.

And note *which* photos are lost: the big group shots and the candids — the photos with the most
people in them. Your aunt is in 3 portraits and 20 group shots. Shrink first and we find her 3.

> **One wrong tag destroys more trust than ten missed photos.** This is precisely how that happens.

---

## The core insight: pass-through, not park

The natural conclusion from the above is *"so we have to store the originals — let's rent 100 GB
somewhere."* **That's wrong, and it's an expensive wrong turn.**

> **Water passing through a filter is not the same as water sitting in a reservoir.**

Every photo has to **pass through** our computer so we can look at it at full resolution. Not one of
them has to be **parked** anywhere afterwards.

```
WRONG (Draft 1)                        RIGHT
1. shrink to 1000px  ← detail dies     1. open at FULL size, in memory
2. find faces (in the shrunk copy)     2. find faces at full size
3. fingerprint (from the shrunk copy)  3. crop from FULL-SIZE pixels, fingerprint
4. delete original                     4. NOW make the small copy, for the website
                                       5. delete original
```

Same storage at the end (~1.3 GB per wedding). Every face measured while the detail was still there.

**Cost:** ~500 MB of working memory, held for half a second per photo. Overnight run goes from
~2 hours to ~4–6 hours. That's the entire price.

---

## The pipeline: 8 steps

```
1 INGEST      photographer pastes a link
2 MANIFEST    we list every file, estimate the job
3 PROCESS     per photo: find faces, fingerprint them, make small copies
4 CLUSTER     group ~32,000 fingerprints into ~147 people
5 PUBLISH     build the lookup tables, generate the share link
6 REVIEW      photographer eyeballs the people grid (optional, 5 min)
7 GALLERY     guests browse
8 SEARCH      guest takes a selfie, gets their photos
```

Steps 1–6 happen **once, overnight**. Steps 7–8 happen thousands of times and must be instant.
**Every design decision below is about pushing work out of 7–8 and into 1–6.**

---

### Step 1 — Ingest

#### What the photographer does

Pastes one link into a box. That's it. We detect what kind of link it is and handle the rest.
**We assume they will do nothing more than this.**

#### Every link type, honestly assessed

| Source | How common | Automatable? | How |
|---|---|---|---|
| **Google Drive folder** ("anyone with link") | **~60%** | **Yes** | Drive API v3 with **our own API key**. Because the folder is *public*, the photographer never logs in and never grants us permission. |
| **Dropbox shared folder** | ~10% | **Yes** | Append `?dl=1` for a zip, or list via the shared-link API |
| **Pixieset / Pic-Time / ShootProof** | ~15% | **Partly** | No public API, but every one has a "download all" button producing a zip link. We accept that zip link. |
| **WeTransfer** | ~10% | **Fragile** | No official API, links die in 7 days. Accept the direct download URL, warn it may fail. |
| **OneDrive** | ~3% | Yes | Graph API; public share links work |
| **Pen drive on a desk** | ~2% | No | Fallback: browser upload |

#### The key unlock — this kills a landmine

Draft 1 listed Google Drive as a landmine: reading a user's Drive needs the `drive.readonly`
permission, a **"restricted scope"** requiring a **paid annual third-party security assessment
costing thousands of dollars a year.**

**That only applies if the photographer logs in with Google and grants us access to their account.**

If instead they paste a link to a folder already set to *"anyone with the link can view"*, we read it
as an anonymous member of the public using our own API key. **No login, no OAuth, no restricted
scope, no audit.**

That landmine is defused, for free, by asking for a link instead of a login. **This is the single
most valuable finding in Draft 2.**

> ⚠ **Unverified.** This needs proving end to end against a real public Drive folder before we build
> on it. See [Open items](#open-items).

#### How it's built — the Adapter pattern

One interface, many implementations, so the rest of the pipeline never knows or cares where photos
came from:

```python
class IngestAdapter:
    def validate(link)  -> ok | error      # is this link readable?
    def list_files()    -> [FileRef]       # walk it, including subfolders
    def open(FileRef)   -> byte stream     # stream one file, never save it

# implementations
GoogleDriveFolderAdapter    DropboxLinkAdapter
DirectZipAdapter            BrowserUploadAdapter
OneDriveAdapter             LocalFolderAdapter (dev only)
```

Adding a new source later is one new file. Nothing else changes.

#### Edge cases that will actually happen

| Situation | What we do |
|---|---|
| **Subfolders** (`Haldi/`, `Sangeet/`, `Reception/`) | Recurse into all of them — **and keep the folder name.** Free feature: *"Priya's photos → Sangeet (12)"*. |
| **Duplicate folders** (`Edited/` and `Raw/` of the same shots) | Perceptual hash (`imagehash.phash`, Hamming distance < 6). Near-identical images collapse into one, keeping the higher-resolution copy. |
| **RAW files** (`.CR2`, `.NEF`, `.ARW`) | Skip in v1. 40 MB each, and rarely delivered to clients. |
| **iPhone `.HEIC`** | Handle it — `pillow-heif`. Common enough to matter. |
| **Videos** | Skip, but log the count so the totals aren't confusing. |
| **Link goes private or expires mid-job** | Job pauses; photographer emailed *"we lost access, please re-share."* Resumes from where it stopped. |
| **Google throttles us** | Exponential backoff. A 4,000-file job needs a polite, slow, retrying downloader — not a tight loop. |
| **Zip is 40 GB** | Stream-extract entry by entry. Never unpack the whole thing to disk. |

#### Free vs. paid

| Now (free) | Later |
|---|---|
| Drive API free tier is generous. **Streaming means zero staging storage** — download one photo, process it, discard it, move on. | If jobs get slow or flaky, add a ~200 GB R2 staging bucket (~₹130/mo) so we download once and retry cheaply. |

---

### Step 2 — Manifest

Before any real work, walk the whole link and build a list.

```
Found:   4,102 images  ·  38.4 GB
         Haldi 412 · Mehendi 690 · Sangeet 1,205 · Wedding 1,340 · Reception 455
Skipped: 18 videos, 6 unreadable files
Estimated time: about 4 hours
```

One row per photo in the database, each with status `pending`. Then the job is queued.

**Why this is its own step:** if the machine crashes at photo 2,800, we restart and skip the 2,800
already marked `done`. Without a manifest, a crash means starting over. On a 4-hour job that's the
difference between a product and a toy.

**Tech:** Python; Supabase (Postgres) for the manifest table; a simple `jobs` table as the queue.

---

### Step 3 — Process (per photo)

The expensive step. Runs once per photo, ever.

```
stream photo from source  ──▶  in memory, never saved to disk
       │
       ├─ 1. fix rotation (EXIF)
       ├─ 2. find faces — whole image + 3×3 tiles
       ├─ 3. align each face — eyes to fixed positions
       ├─ 4. score quality — size, sharpness, angle
       ├─ 5. fingerprint — 112×112 crop → 512 numbers
       ├─ 6. make 2048px preview + 400px thumbnail
       └─ 7. upload the small stuff, DISCARD the original
```

**1 · Rotation.** Photos carry a hidden "which way is up" tag. Ignore it and half the wedding is
sideways and the face-finder sees nothing. The most common silent bug in image pipelines. Apply it
first, always.

**2 · Finding faces — two passes.** The face-finder shrinks its input to ~640 px internally, so a
back-row face at 90 px becomes 9 px and is invisible. So we look twice:

```
 One 6000px photo             Looked at as 9 overlapping squares
 ┌──────────────────┐         ┌─────┬─────┬─────┐
 │  30 people, all  │   ──▶   ├─────┼─────┼─────┤   each square gets
 │  faces ~90px     │         ├─────┼─────┼─────┤   full attention
 └──────────────────┘         └─────┴─────┴─────┘
   finder sees 9px               finder sees ~90px  ✓
```

Once at the whole photo (catches big faces, cheap), once at a 3×3 grid of overlapping tiles at
native resolution (catches small faces). Merge the two lists, remove duplicates where both passes
found the same face (NMS, IoU ~0.4).

**Without this, "process the original" achieves nothing** — we'd miss the same faces anyway.

*Tech: SCRFD via ONNX Runtime on CPU.*

**3 · Alignment.** The model doesn't want a rectangle cut from the photo — it wants the face
*straightened*, eyes at fixed positions, like a passport photo. The finder gives 5 landmarks
(2 eyes, nose, 2 mouth corners); we use them to rotate and scale the crop into a standard 112 × 112
square. Skipping this costs real accuracy and is easy to forget.

**4 · Quality scoring.** Four numbers per face:

| Measure | Reject if |
|---|---|
| Face height **in the original photo** | under 50 px |
| Detector confidence | under 0.6 |
| Sharpness (variance of Laplacian) | very low |
| Head turn (yaw) | more than ~50° from front |

Faces between **50–80 px are kept but flagged second-class** — they may join an existing person but
may never *start* one. See [Step 4](#step-4--cluster).

**5 · Fingerprint.** ArcFace turns each aligned 112 × 112 crop into 512 numbers, then we scale them
so every fingerprint sits on a sphere of the same size — making comparison a single fast operation
(cosine distance).

**6 · Small copies.** Two, both cut from the full-resolution original:

- **Preview, 2048 px** — what a guest sees when they tap a photo. ~400 KB.
- **Thumbnail, 400 px** — what fills the grid. ~40 KB.

*Tech: `pyvips` — much faster and far lighter on memory than the obvious alternatives.*

**7 · Discard.** The 12 MB original is released from memory. Never written to our disk, never
uploaded to our storage.

#### Output per photo

```
R2:      ev_ab12/prev/000481.jpg     400 KB
         ev_ab12/thumb/000481.jpg     40 KB
DB:      1 row in photos, 4 rows in faces
Parquet: 4 rows of 512 numbers
```

#### Timing, honestly

| | 4,102 photos |
|---|---|
| Your desktop, CPU only | **4–6 hours** |
| Rented GPU at ~₹25/hr | **25–40 minutes → ~₹20/wedding** |

Overnight is fine now. The GPU option becomes worth it the moment a paying customer wants results
the same day.

---

### Step 4 — Cluster

#### "How do you know there are 150 people?"

**You don't. Nobody tells us. The number is the *answer*, not the input.**

Every face is now 512 numbers. Think of those as **coordinates**. Two numbers = a point on paper;
three = a point in a room. With 512 it's a space we can't picture, but the maths is identical, and
the key property holds:

> The fingerprinting model was trained so that **two photos of the same person land close together**,
> and two different people land far apart — regardless of lighting, angle or makeup.

So after Step 3 we have **32,136 dots floating in this space**. If you could see it, you'd see clumps.
Each clump is one person.

```
        · ·                                    ·
      ·  A  ·        ·· ·                    · D
       · ··          ·  B ·                  ·
                      · ·         · ·
                                ·  C  ·
   clump A = 2,104 dots           ·· ·        clump D = 4 dots
   (the bride)                                (someone's uncle,
                                clump C        in 4 photos)
                                = 340 dots
```

The algorithm's whole job is: **find the clumps.** It comes back and says "there are 147." That's
where the number comes from.

#### Which algorithm, and why not the obvious one

| Algorithm | Verdict |
|---|---|
| **K-means** | **No.** You must state the number of groups in advance. We don't know it. Disqualified. |
| **DBSCAN** | Finds clumps by density. Assumes clumps are equally dense — but the bride has 2,104 faces and a guest has 4. Poor fit. |
| **HDBSCAN** | DBSCAN's smarter cousin; handles wildly different clump sizes. Good fit. |
| **Agglomerative, average-link ✓** | Every face starts as its own group; repeatedly merge the two closest; stop at a distance threshold you set. **One knob, and completely repeatable.** **Start here.** |

Repeatability matters enormously while tuning: change one number, re-run, and any difference in the
result is caused by *your change* and nothing else.

*Tech: `sklearn.cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=T,
metric='cosine', linkage='average')`. Compare against HDBSCAN once there's real data.*

#### The two-pass approach (what actually gets built)

A single pass forces an impossible trade-off. Loose threshold → two lookalike cousins merge into one
person → **guests see strangers' photos.** Tight threshold → the bride splits into 6 fragments →
guests find only part of their photos.

So: **be tight first, then merge deliberately.**

```
PASS 1 — deliberately tight
  32,136 faces → 380 small, very pure groups
  (the bride is split across 6 of them — fine and expected)

PASS 2 — merge, with evidence
  compare the 380 group-averages to each other
  merge two groups when they are close AND the evidence agrees:
    · are they ever in the SAME PHOTO? if yes they are DIFFERENT PEOPLE
      → never merge (two faces in one photo cannot be one person)
    · do they share many photos in common?
  380 → 147 people
```

That **same-photo rule** is free, obvious once stated, and eliminates a whole class of the worst
errors. It was not in Draft 1.

#### What comes out — a realistic result

```
4,102 photos processed
   41,238 faces found
    9,102 rejected by the quality gate  (blurry, tiny, side-on)
   32,136 good faces  →  clustering  →

   147 people  +  1,847 leftover faces

   Person 1   (bride)     2,104 faces  in 1,890 photos
   Person 2   (groom)     1,976 faces  in 1,801 photos
   Person 3–15  (family)    200–800 photos each
   Person 16–90 (guests)     20–200 photos each
   Person 91–147              3–20 photos each
```

#### The 1,847 "leftover" faces — do not throw these away

Some faces belong to no clump: someone in 2 photos, or a face at an awkward angle. The algorithm
labels them "noise."

**Keep every one, with no person assigned (`person_id = NULL`).** A distant cousin who's in 3 photos
is *exactly* the person most delighted to find them, and [Step 8](#steps-7--8--the-guest-and-the-selfie-search)
has a fallback that searches these individually. Delete them and you silently lose that guest entirely.

#### The safety rule

**Splitting a person is a mild failure. Merging two people is a catastrophe.**

- Splitting → Priya finds 60% of her photos and is mildly disappointed.
- Merging → Priya sees photos of a stranger, and stops trusting the product.

**Always err tight.** Two automatic sanity checks:

- Under ~40 people for a 4,000-photo wedding → threshold too loose, people are being merged. **Alarm.**
- Over ~500 → too tight, everyone's fragmented. Annoying, not dangerous.

---

### Steps 5 & 6 — Publish, and the photographer's review

**Publish** builds `person_photos`, picks each person's best face as their cover tile, generates the
share slug, and marks the event live.

**Review** is optional, takes 5 minutes, and is worth a surprising amount:

```
┌────────────────────────────────────────────────┐
│  147 people found                              │
│                                                │
│  [face] Person 1     [face] Person 2   ...     │
│  1,890 photos        1,801 photos              │
│                                                │
│  ⌄ merge   ⌄ rename   ⌄ hide                   │
└────────────────────────────────────────────────┘
```

Sorted by photo count, so the couple and family are the first tiles. The photographer can:

- **Merge** two tiles ("these are both the bride") → fixes any splitting in one click
- **Name** people → guests can then browse by name without a selfie at all
- **Hide** junk clusters (a face on a poster, a reflection)

**The value:** because we deliberately erred tight in Step 4, some people are split. A human fixes
that in seconds — something no amount of threshold tuning will ever fully achieve. Cheap to build,
large quality gain.

---

### Steps 7 & 8 — The guest, and the selfie search

#### The three image sizes, and exactly where each is used

| | Size | Weight | Used for |
|---|---|---|---|
| **Thumbnail** | 400 px | 40 KB | the results grid |
| **Preview** | 2048 px | 400 KB | tapping a photo, full screen |
| **Original** | 6000 px | 12 MB | the download button only |

#### The guest journey, step by step

**1.** Aunt taps the WhatsApp link → gallery opens. Full wedding, thumbnails, scrollable. Big button:
**"Find my photos"**.

**2.** Taps it → plain-language consent screen:

> *We'll convert your selfie into a set of numbers to match against the photos.
> **The selfie itself is deleted immediately and never stored.***

**3.** Camera opens. One selfie. **The browser shrinks it to 640 px before sending** — an 80 KB
upload, not 4 MB. Fast on patchy wedding-venue mobile data.

**4.** Our server:

```
receive selfie (in memory)
  find faces  → must be exactly one
                0 faces   → "we couldn't see a face, try better light"
                2+ faces  → "just you in the frame, please"
                too small → "hold the phone a bit closer"
  fingerprint it → 512 numbers
  DELETE the selfie bytes        ← never touches a disk
  match (below)
  return a list of photo IDs
```

**5. The match — two tiers.**

> **Tier 1 (90% of cases, ~5 ms).** Compare the selfie's 512 numbers to the **147 person-averages**
> in Postgres. If one is clearly closest and comfortably inside the threshold → that's her. Return
> her photos from `person_photos`. Done.
>
> Comparing against a *group average* is more accurate than comparing against any single face —
> averaging 80 photos of Priya cancels the noise from any one bad angle.

> **Tier 2 (the rest, ~15 ms).** If nothing matched confidently, or two people tie, load that
> event's `faces.parquet` (33 MB) into memory and compare against **all 32,000 individual
> fingerprints**, including the 1,847 leftovers.
>
> 32,000 comparisons sounds like a lot. It's 16 million multiplications — **about 10 milliseconds**
> in numpy. **At this scale a dedicated vector database is solving a problem we don't have.**
>
> Tier 2 is what rescues the cousin who's in 3 photos and never formed a group of her own.

Cache the parquet in RAM keyed by event; a 1 GB VPS holds ~20 events' worth.

**6.** She sees a grid of **thumbnails** — 37 photos, loaded in under a second.

**7.** Taps one → the **2048 px preview**, full screen. Sharp on any phone.

**8.** Taps **Download** → see below.

#### Does she get the original? — the download policy

| | What she gets | What it costs us | Catch |
|---|---|---|---|
| **A — Preview only** | 2048 px, ~400 KB | **₹0** | Fine for WhatsApp, Instagram, a 6×8 print. Not enough for A3. |
| **B — Proxy from source** | **The true original, 12 MB** | **₹0 storage.** Bandwidth only. | Our server fetches it live from the photographer's source using the saved `source_file_id`, and streams it through. **Breaks if the photographer deletes the folder.** |
| **C — We store originals** | The true original | ~₹200 per wedding | Reliable, but you're paying storage from customer one |

**✅ LOCKED: A + B.** Preview is instant and always works; a "Download full size" button quietly
proxies the original when the source link is alive, and falls back gracefully to the preview when
it isn't.

**This costs ₹0, and it only works because we ingest from a link.** Browser upload would have
destroyed the pointer and forced us into option C.

#### Things worth deciding here

| | |
|---|---|
| **Can guests see everyone's photos?** | Default yes — that's how wedding galleries work. But give the photographer a **"private mode"** toggle: guests see *only* their own results, no browsable gallery. Some families will want this. |
| **Abuse** | Rate-limit by IP — say 10 searches/hour. Someone could search using a photo of a person who isn't them. Can't be fully prevented; state it plainly in the privacy text. |
| **"Not my photos"** | A report button on every result. Feeds straight back into tuning — a free accuracy meter from real users. |

---

## The data model

```sql
events
  id, photographer_id, name, event_date,
  share_slug,              -- 'sharma-wedding-x7k2'  → the public URL
  source_type,             -- 'gdrive' | 'dropbox' | 'zip' | 'upload'
  source_folder_id,        -- so we can fetch originals later  ← enables download policy B
  status, photo_count, person_count,
  embedding_model_version, -- CRITICAL, see below
  expires_at               -- 90 days out

photos
  id, event_id,
  original_filename,       -- 'Sangeet/IMG_4821.jpg'
  sub_album,               -- 'Sangeet'   ← free feature from folder names
  source_file_id,          -- Drive's ID for THIS file → the download link
  preview_key,             -- 'ev_ab12/prev/000481.jpg'
  thumb_key,               -- 'ev_ab12/thumb/000481.jpg'
  width, height, taken_at, phash, face_count

faces                      -- ~32,000 rows per wedding
  id, event_id, photo_id,
  bbox_x, bbox_y, bbox_w, bbox_h,   -- position in the ORIGINAL photo
  quality_score, det_score, blur, yaw,
  person_id,               -- NULL for the 1,847 leftovers
  embedding_ref            -- row number in the parquet file (NOT the 512 numbers)

persons                    -- ~147 rows per wedding
  id, event_id,
  label,                   -- 'Person 14', or 'Priya' if named
  centroid vector(512),    -- the AVERAGE of all their faces
  face_count, photo_count,
  cover_face_id            -- their best face → the tile thumbnail

person_photos              -- ~12,000 rows. THIS is the one that matters.
  person_id, photo_id, event_id,
  best_face_id, quality, taken_at, sub_album
  PRIMARY KEY (person_id, photo_id)

searches                   -- usage proof + debugging when someone complains
  id, event_id, matched_person_id, result_count, tier, created_at, ip_hash
```

### Why `person_photos` exists — how display actually works

You *could* work out "which photos is Person 14 in?" at display time by joining and filtering. But we
know the answer at 4 a.m. when the job finishes, so **we write the answer down** — one row per
(person, photo) pair.

Showing a guest their photos is then this, and nothing more:

```sql
SELECT p.thumb_key, p.preview_key, p.sub_album, p.taken_at
FROM person_photos pp
JOIN photos p ON p.id = pp.photo_id
WHERE pp.person_id = $1
ORDER BY p.taken_at
LIMIT 60 OFFSET $2;
```

One indexed lookup. **About 5 milliseconds.** No maths, no AI, no vectors — a boring database read.

> **The whole philosophy: all the thinking happens once, overnight. Displaying is dumb and instant.**

Sorting by `person_photos.quality` gives "best photos first." Filtering on `sub_album` gives
"Priya's Sangeet photos." Both free, because we wrote the columns down.

### The storage split — and why Draft 1's plan died at wedding #2

Draft 1 put all face fingerprints into the main database:

```
32,000 faces × 512 numbers × 4 bytes  =  66 MB
+ the search index Postgres builds     ≈ 200 MB per wedding
Supabase free tier                     =  500 MB total
```

**Two weddings before the free tier dies.** Draft 1 claimed 4–5.

**The fix — split by size:**

| What | Where | Size per wedding |
|---|---|---|
| The **147 person-averages** | Postgres (`pgvector`) | **300 KB** |
| The **32,000 individual fingerprints** | one `faces.parquet` on R2 | 33 MB (float16) |
| Everything else (positions, scores, mappings) | Postgres | ~5 MB |

**Postgres per wedding: ~5 MB instead of 200 MB → 40+ weddings on the free tier instead of 2.**
The parquet file costs about ₹0.05/month.

This works because of the two-tier search in Step 8.

### The model version stamp — add it today

Every fingerprint must be stamped with **which version of the model made it.**

Fingerprints from two model versions are mutually unreadable — like storing some measurements in
centimetres and some in inches with no label. Nothing crashes. **Guests just silently stop finding
their photos, forever.**

`embedding_model_version` goes on `events`, `faces` and `persons`, and every search filters on it.
Costs nothing now. Retrofitting means reprocessing every wedding ever done.

---

## The stack: free now, paid later

| Piece | Free (first 5 weddings) | Cost | Upgrade when |
|---|---|---|---|
| **Ingest + worker** | Your desktop, Python | ₹0 | Jobs too slow → rented GPU, ~₹20–30/wedding |
| **Small copies + parquet** | Cloudflare R2, 10 GB free | ₹0 | ~7 weddings live → ~₹1.50/wedding/month |
| **Database** | Supabase free, 500 MB | ₹0 | ~40 weddings (with the split above) → $25/mo covers hundreds |
| **Website** | Cloudflare Pages | ₹0 | Free for a very long time |
| **Selfie-search API** | **Your desktop + Cloudflare Tunnel** | **₹0** | See below |
| **Payments** | Razorpay | ~2% per transaction | — |

### The Cloudflare Tunnel trick

The selfie-search API must be reachable from the public internet 24/7. Every "free" hosting tier
either falls asleep (50-second wake-up — unusable when a guest is standing there waiting) or forbids
commercial use.

**Cloudflare Tunnel gives your own desktop a real public HTTPS address** — free, no port-forwarding,
no fixed IP. Your machine is running anyway during the pilot.

Move to a ~₹400/month VPS the day you can't afford your desktop being the single point of failure.
Not before.

> **Honest total for the first 5 weddings: ₹0.**

### What pushes us off the free tiers

Only **old weddings we're still storing**. Which hands us a natural policy: **galleries stay live for
90 days**, then previews are deleted — the originals were never ours anyway. That caps storage
permanently and doubles as our data-retention story for the DPDP Act.

### What we charge

Photographers work event to event, not monthly. A subscription asks them to pay in the months they
shoot nothing — the fastest route to a cancellation. So: **price per event, tiered by photo count** —
something like ₹1,500 / ₹3,000 / ₹5,000. **Treat those as a hypothesis to test on the first five
photographers, not a decision.**

> **The real risk in this business.** The technology is not the risk. The risk is that **the
> photographer feels only mild pain here, while the guest feels the real pain — and guests don't buy
> software.** So which is it: the photographer buys it as a differentiator to win bookings, or the
> couple buys it as an add-on? Five conversations answers this. Have them before we've built three
> months of product.
>
> *Devesh's position (10 Aug): depends on the channel, not a single answer — photographer pays if we
> sell direct to them, the couple pays if we sell to them. Both channels stay open.*

---

## Landmines

### 1 — Model licensing ⚠ UNRESOLVED

The InsightFace *code* is open, but the **pretrained model weights — the actually valuable part — are
licensed for non-commercial research use only**, as far as we currently understand it. Fine for
building, testing and demoing. **Not fine the day someone pays us.**

**Options:** read the current licence text ourselves (it may have changed); switch to a
permissively-licensed model with somewhat lower accuracy; or price a commercial API into the unit
economics. **Does not block the next two months** — but must be settled before invoice number one.

### 2 — Google Drive permissions ✅ DEFUSED

**Was:** `drive.readonly` is a restricted scope requiring a paid annual security assessment.

**Now:** we never ask the photographer to log in with Google. They paste a link to an
already-public folder, and we read it with our own API key as a member of the public. **No OAuth, no
restricted scope, no audit.**

> ⚠ Still needs proving end to end against a real public folder before we build on it.

### 3 — Biometric data law ⚠ UNRESOLVED

Face fingerprints are **biometric personal data** under India's DPDP Act. Practically: a consent
screen for guests, the 90-day deletion policy, a "delete me" button, and clarity in the photographer
contract that **they** are the data fiduciary and **we** are the processor.

An hour with a lawyer before the first paid event. The rules have been shifting — don't trust
anything written a year ago, **including this document**.

### 4 — Minor, but real

Vercel's free tier prohibits commercial use. Hence Cloudflare Pages. Cheap to get right now,
annoying to discover later.

---

## Build order

| When | What | "Done" means |
|---|---|---|
| **Week 1–2** | Local scripts only. Link in → face clusters out. No website at all. | On a real wedding, ≥90% of one person's faces land in a single pile, with near-zero wrong faces mixed in. |
| **Week 3–4** | Shareable gallery link + selfie search. Ugly is fine. | We can WhatsApp a link to Adarsh's mother and she finds her photos unaided. |
| **Month 2** | Photographer login, link-paste flow, downloads, consent + deletion. | A photographer can use it without either of us in the room. |
| **Month 3** | Five photographers, free, in exchange for their data and brutal feedback. Payments last. | We know who pays, and how much. |

If the clustering doesn't work, a beautiful gallery is worthless. And if it does work, everything
after it is ordinary web development. **So we find out first.**

### The first three things that get written

1. **The ingest adapter + processing script, no UI.** Point it at a Drive link (or a local folder in
   dev). It walks every image, detects faces at native resolution with tiling, discards the bad ones,
   writes fingerprints to a parquet cache, and saves face crops so we can *see* what it found.
2. **The clustering script.** Reads the cache, groups the faces, produces an HTML page —
   "Person 1 — 84 photos" with a thumbnail grid under each. We scroll it and count the mistakes by
   eye. **That page is our accuracy meter, and we'll be staring at it for two weeks.**
3. **Tune** against real photos until the piles are clean.

No database, no cloud, no login until week 3.

### The stage split that makes CPU-only workable

```
STAGE 1 — expensive, runs ONCE per wedding, cached to disk
  photos → detect (tiled, native res) → quality gate → 512-number fingerprints → faces.parquet

STAGE 2 — cheap, re-runs in seconds, unlimited times
  faces.parquet → cluster → evaluate → report
```

Stage 1 runs overnight. After that every clustering experiment reads the cached file and finishes in
about five seconds — a hundred tuning iterations a day on a laptop. **This split is what makes "no
GPU" a non-issue**, and it's the right design anyway: a GPU later just makes Stage 1 faster without
changing anything else.

---

## Who does what

| Who | Owns |
|---|---|
| **Adarsh** | Product calls, the five photographer conversations, pricing — and the one job that genuinely can't be outsourced: **getting real wedding photo sets to test on.** |
| **Devesh** | Runs the processing machine, and does the human accuracy testing: take 200 photos, check by eye how many the system got right and wrong. Tedious, essential, and the only honest measure of whether this works. |
| **Claude** | All code, all infrastructure, all tuning. |

> **The current bottleneck.** Clustering quality *is* the product, and it cannot be tuned against
> imaginary photos. We need roughly **one real wedding, or ~500 photos of a group of 20-ish recurring
> people**, before any tuning work means anything.

Three sources, in order of value:

1. **Ask a photographer friend for one wedding folder now.** Offer free processing forever in
   exchange. This is simultaneously our test data *and* our first sales conversation — we'll learn
   whether they even feel the pain, which is the real question.
2. **Our own phone galleries.** Family functions, group shots, the same people across years and
   lighting conditions. Same problem shape, smaller.
3. Public face datasets — only useful for confirming the fingerprinting works at all.

> ⚠ **The cricket-player test set cannot test the resolution work.** Those are cropped press photos
> — big faces filling the frame. It'll prove the machinery runs end to end, but it cannot tell us
> whether we've solved the back-row-of-a-group-photo problem, which is now the main technical risk.

---

## Open items

| # | Item | Owner | Blocks |
|---|---|---|---|
| **1** | **One real public Google Drive wedding folder link** — even a small one, even a family function. Proves the Drive-API-with-public-link path works end to end before we build on it. **Highest-value thing available right now.** | Adarsh (trying to source; will create one from a personal account if needed) | Step 1 implementation |
| **2** | One real 24 MP wedding photo, ideally a large group shot. Every number in [Background: pixels](#background-pixels-and-the-one-number-that-matters) is arithmetic on an assumed camera. One real file turns it from *estimated* into *measured*. | Adarsh | Tiling parameters |
| **3** | InsightFace weights licence — read the current text | — | First invoice, not the build |
| **4** | DPDP Act — one hour with a lawyer | Adarsh | First paid event |
| **5** | Q5 from Draft 1: realistically, how many hours a week? | Devesh | Whether the build order is 4 weeks or 10 |
| **6** | "Private mode" toggle — do we build it in v1 or defer? | Adarsh | Month 2 |

---

*Draft 2 · 11 August 2026. Numbers marked "roughly" are estimates and should be re-checked against
current provider pricing before anyone relies on them. The landmines above are unverified and need
confirming independently.*
