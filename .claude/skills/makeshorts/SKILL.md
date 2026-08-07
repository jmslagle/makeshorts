---
name: makeshorts
description: Use when turning a long-form recording (webinar, talk, podcast, Zoom meeting) into short vertical clips for social with the makeshorts pipeline. Covers running ms prepare, choosing sources, writing clips.json as the editorial step, and verifying framing before rendering.
---

# Turning a recording into vertical clips

You are the editorial step. The tool does the mechanical work; **you choose
which moments are worth posting and why**, and you write that decision into
`clips.json` where a human can read it, disagree, and re-render without you.

```
ms prepare <input>   mechanical: transcript, silence, regions, PROMPT.md
   ── YOU read PROMPT.md and write clips.json ──
ms lint <slug>       the gate; nothing renders until it passes
ms render <slug>     encodes each clip
```

## 1. Pick the source before anything else

A recording often ships several views. **Count the pixels on the subject** —
this decides clip quality more than any later setting, and it cannot be
recovered afterwards.

```bash
ffprobe -v error -show_entries stream=width,height -of csv=p=0 <file>
```

For a Zoom export: `_avo_` is active-speaker-only (usually the most pixels on a
face), `_gvo_` is cameras-only gallery, `_as_` is the shared screen, `_gallery_`
is slides with a small camera tile. A face that occupies a 322×181 tile needs a
**10× upscale** to fill a 1080-wide frame and will look like mush; the same face
full-frame at 1280×720 needs 2.67×. Check before committing.

If views are frame-aligned (same duration, same audio), you can draw from
several files in one clip — sharp slides from one, a usable face from another.
Verify alignment rather than assuming it:

```bash
# identical loudness envelopes at the same timestamp == same clock
ffmpeg -v error -ss 1200 -t 60 -i A.mp4 -map 0:a -ac 1 -ar 8000 -f f32le - | ...
```

## 2. Read PROMPT.md, then read the transcript properly

`PROMPT.md` is generated from `config/criteria.yaml` and tells you the rubric,
the gates and the exact JSON shape. Read it — do not guess the schema.

Then actually read `transcript.txt`. Skimming for keywords produces clips that
score well and land flat. You are looking for moments that survive being cut
out of everything around them.

## 3. Write clips.json

Rules that matter, learned the hard way:

- **Never invent a timestamp.** Every `start`/`end` must be a real word
  boundary from `words.json`, and `source_text` must be that span verbatim.
  Lint checks both and prints a word-level diff. Build the file by slicing
  `words.json` programmatically rather than typing numbers.
- **Score honestly.** `why.scores` needs every rubric criterion with a quoted
  piece of evidence. A 4 you can defend is worth more than a 5 you cannot — a
  human reads this to decide whether to trust the clip.
- **Diversity beats volume.** `max_per_theme` exists because eight clips about
  pricing is one clip about pricing. Label `why.theme` truthfully; do not
  invent a distinct theme to slip a duplicate past the cap.
- **`visual_dependency: true`** when the words lean on what is on screen. Lint
  then *requires* a slide region in the layout. Do not set it false to dodge
  the check — that hides a `standalone` problem rather than fixing it.

## 4. Run `ms lint --fix` — always

This is the step most likely to be skipped and most likely to matter. Without
it, clips cut exactly on the first phoneme with no lead-in, which sounds like a
stray noise or a clipped breath. `--fix` snaps edges to real boundaries and
applies `pad_in`/`pad_out`.

If padding shifts a clip onto another word, `source_text` will no longer match
— regenerate it from the new span. `--fix` deliberately will not do this for
you: rewriting quotes automatically would hide a fabricated one.

Then `ms lint` until clean. Read the findings; they name the fix.

## 5. Verify framing empirically, before rendering 16 clips

**Region detection classifies content, not subjects.** It will tell you "this
whole frame is a speaker" — true and useless for framing. A centred crop of an
off-centre subject cuts their face in half.

Measure where the face actually is, and measure it **in a rendered clip**, not
in the source: source backgrounds contain warm-coloured clutter that drags a
skin-tone centroid sideways, while a rendered clip has that cropped away.

Do not assume the subject holds still. Measure several clips across the whole
recording — a presenter commonly drifts across a long talk (0.69 → 0.76 of
frame width in one real case), so a single centre that suits the opening will
be wrong by the end.

Two levers, both in `clips.json`:

- **centre**: the region rect's midpoint.
- **zoom out + `fit: contain_blur`**: a wider region shows *more* real pixels
  (`cover` always crops to 9:16, so widening it only pans) and makes centring
  error proportionally smaller, at the cost of blurred bars. ~1.2–1.3× absorbs
  a typical drift.

## 6. Render and check

```bash
ms render <slug> --only 01      # one clip first
ms render <slug>                # then the batch
```

Pull frames and look at them. Check the subject is in frame, captions are
legible, and any watermark reads against the actual background — a dark logo
vanishes on a dark camera feed, a white one vanishes on a white slide.

## Things that are config, not code

Almost every quality problem here is fixed by editing data, and reaching for
code is usually a sign of a misdiagnosis:

| Symptom | Fix |
|---|---|
| Subject off-centre / cut off | region rect in `clips.json` |
| Too soft, or too much upscaling | wider region + `contain_blur` |
| Clip starts abruptly | `ms lint --fix` (padding) |
| Clip opens on "So"/"And"/"Um" | pick a different span; `gates.forbid_opening_fillers` catches it |
| Good clips rejected for "dead air" | measure real pause lengths, then set `max_internal_silence` from the data |
| Too many similar clips | `diversity.max_per_theme` |
| Watermark invisible | different logo variant in `config/render.yaml` |

When a gate rejects something you believe is good, work out whether the rule or
the clip is wrong — and settle it with a measurement, not a preference. On one
real run a 1.2s `max_internal_silence` rejected half the clips; the median pause
in that talk was 0.76s and the 90th percentile 1.55s, so the rule was wrong and
the evidence said where to put it.
