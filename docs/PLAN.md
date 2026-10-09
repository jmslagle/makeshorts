# makeshorts — design record

> **Status: built.** This was the plan agreed before implementation, kept
> because the *reasoning* behind each fork is still the best explanation of why
> the code looks like it does. It has been corrected where the build diverged,
> and [What changed during implementation](#what-changed-during-implementation)
> at the end records those divergences and what forced them. For how to use the
> tool see the [README](../README.md); for how to change it see
> [CLAUDE.md](../CLAUDE.md).

## Context

Long-form webinar recordings contain a handful of moments worth clipping, and finding them is judgment work while everything around it is not. The point of this tool is the seam: mechanical work stays deterministic and re-runnable, editorial work produces one reviewable file, and rendering consumes that file without re-invoking any of the thinking.

The deliverable of the thinking step is `clips.json`. A human must be able to read it, disagree with it, edit it, and re-render without touching the AI. `clips.json` is engine-agnostic — ffmpeg is today's renderer, not a concept the edit list knows about.

Greenfield at the time of writing: an empty directory.

### Environment findings that shape the design

- **`ffmpeg` 8.1.2 (homebrew/core) is built without `libass` and without `libfreetype`.** `-buildconf` shows only `--enable-libx264 --enable-libx265 --enable-videotoolbox --enable-audiotoolbox`. **The `subtitles` and `drawtext` filters do not exist on this machine.** Verified by `ffmpeg -filters`. Burned-in captions cannot use the conventional path here.
- Present and usable: `overlay` (timeline-enabled, so `enable='between(t,a,b)'` works), `crop`, `scale`, `pad`, `vstack`, `hstack`, `boxblur`, `silencedetect`, `cropdetect`, `libx264`, `h264_videotoolbox`, `aac`.
- Pillow 10.4.0 with its own freetype 2.13.2 and raqm — full text shaping available in Python, independent of ffmpeg's build.
- Fonts on disk: `/System/Library/Fonts/Supplemental/{Arial Bold,Impact}.ttf`, `/System/Library/Fonts/SFNS.ttf`.
- `uv` and Python 3.12.0 available. `faster-whisper` not installed.

### Decisions taken

| Fork | Decision |
|---|---|
| Captions | Two backends behind one interface, style-driven; auto-select by probing the installed ffmpeg. Resolves to Pillow on this machine. |
| Source geometry | Named, typed **regions** — not "left half / right half". Handles 1 speaker, 2 speakers, and speaker+slides identically. |
| Region detection | Mechanical proposal (`regions.json`, with confidence) + editorial confirmation into `clips.json`. |
| Layouts | `focus`, `hero_inset`, `stack`. No `full` mode — it is `{mode:"focus", region:"frame"}` with `fit:"contain_blur"`. |
| Mid-clip layout changes | In schema **and** implemented in v1, via segment-and-concat. |
| Speaker attribution | Editorial. Declared per clip in `clips.json`, overridable by hand. |
| AI invocation | Claude Code, interactive. No API key, no per-run cost. `prepare` emits `PROMPT.md`; the human/agent writes `clips.json`. |

---

## Architecture

Three stages, three trust levels, one artifact between each.

```
  MECHANICAL (deterministic, no AI)        EDITORIAL (AI, reviewable)      RENDER (deterministic)
  ────────────────────────────────         ──────────────────────────      ─────────────────────
  ms prepare <input>                       (Claude Code reads PROMPT.md)   ms render <slug>
    ├─ media.json                            └─ writes clips.json  ────────►  out/<slug>--01-....mp4
    ├─ words.json      ┐                            ▲                        out/<slug>--01-....json
    ├─ silence.json    ├── inputs to lint ──────────┘                          (render receipt)
    ├─ regions.json    ┘         ▲
    ├─ transcript.txt            │
    ├─ raw.srt                   ms lint <slug>  ← the gate. Nothing renders until this passes.
    └─ PROMPT.md
```

`ms lint` is what makes the seam real. It mechanically verifies the AI's output against `words.json`, `silence.json`, and `config/criteria.yaml` — see **Lint rules** below. It is the reason a bad edit list fails loudly instead of rendering something wrong.

---

## `clips.json` — the edit list

Engine-agnostic by construction. Times are absolute seconds in the source. Geometry is normalized `[x, y, w, h]` in 0..1. Styles are *names* that resolve at render time. **No codec, bitrate, preset, filter string, pixel crop, font path, or output directory appears anywhere in this file** — those live in `config/render.yaml`.

```jsonc
{
  "schema_version": "1.0",
  "job": "acme-q3-webinar",
  "source": {
    "path": "jobs/acme-q3-webinar/source.mp4",
    "duration": 3612.44,
    "resolution": [1920, 1080],
    // Confirmed from regions.json by the editorial step. `frame` is implicit, always present.
    "regions": [
      {"id": "cam_a",  "kind": "speaker", "label": "Dana Reyes", "rect": [0.0, 0.0, 0.5, 1.0]},
      {"id": "slides", "kind": "slide",                          "rect": [0.5, 0.0, 0.5, 1.0]}
    ]
  },
  "output": {"width": 1080, "height": 1920, "fps": 30},
  "criteria_ref": {"file": "config/criteria.yaml", "version": "2026-08-06", "sha256": "…"},

  "clips": [
    {
      "id": "01-cac-payback-math",          // ^\d{2}-[a-z0-9][a-z0-9-]*$ — drives the filename
      "title": "The CAC payback number nobody checks",
      "start": 1284.10,
      "end":   1327.88,
      "source_text": "Eighteen months. That's the number that kills companies…",

      "why": {
        "one_line": "Contrarian claim on a hard number, resolves inside 40s, opens cold.",
        "theme": "unit-economics",
        "scores": {                          // one entry per rubric id in criteria.yaml — lint enforces
          "hook_strength": {"score": 5, "evidence": "First 2.1s is a bare number plus a stake."},
          "standalone":    {"score": 5, "evidence": "No back-references; defines CAC payback inline."},
          "single_idea":   {"score": 4, "evidence": "One claim, one worked example."},
          "payoff":        {"score": 5, "evidence": "Delivers the corrective rule at 0:38."},
          "specificity":   {"score": 5, "evidence": "18 months, 4.2x, two named categories."},
          "quotability":   {"score": 4, "evidence": "'Eighteen months is where companies die.'"},
          "housekeeping":  {"score": 5, "evidence": "No logistics or screen-share chatter."}
        },
        "weighted_score": 4.71,              // lint recomputes this and compares
        "rejected_alternatives": [
          {"start": 1190.0, "end": 1240.0, "reason": "standalone=2 — opens with 'like I said earlier'."}
        ]
      },

      "visual_dependency": true,             // clip references on-screen content → lint requires a slide region
      "speaker": "cam_a",

      "layout": [                            // object OR list of timed spans; `at` is relative to clip start
        {"at": 0.0,  "mode": "focus",      "region": "cam_a", "fit": "cover"},
        {"at": 11.7, "mode": "hero_inset", "hero": "slides", "inset": "cam_a",
                     "inset_corner": "bottom_right", "inset_scale": 0.28}
      ],

      "captions": {"style": "pill-karaoke", "position": "lower_third"},
      "audio": {"normalize": true}
    }
  ]
}
```

**Layout modes** (all compose over regions, so speaker-count is irrelevant to the code):

- `{"mode": "focus", "region": R, "fit": "cover"|"contain_blur"}` — R fills the frame. Active-speaker crop, single-speaker source, and slides-only are all this.
- `{"mode": "hero_inset", "hero": R1, "inset": R2, "inset_corner": …, "inset_scale": …}` — slide + talking head, or speaker + reaction.
- `{"mode": "stack", "regions": [R1, R2, …]}` — tiled vertically, each fit to its band.

---

## Selection — the part that isn't mechanical

The rubric is **data**, and `PROMPT.md` is *generated from it*. Editing `config/criteria.yaml` changes the prompt, changes the score keys the linter demands, and changes the thresholds enforced. There is no rubric text inside a Python string.

`config/criteria.yaml`:

```yaml
version: "2026-08-06"

# ── Gates: mechanically enforced by `ms lint`. The model never adjudicates these. ──
gates:
  duration: {min: 15, target: 35, max: 60}   # was 20/45/60 when planned
  snap_to_word_boundaries: true       # start/end must land on a real word edge in words.json
  start_on_sentence_start: true
  end_on_sentence_end: true
  max_internal_silence: 2.0           # was 1.2 when planned; see 'what changed'
  pad_in: 0.12
  pad_out: 0.25
  hook_window: 3.0                    # the "first three seconds" the rubric judges
  min_separation: 45                  # seconds between clip midpoints — keeps clips spread out
  max_clips: 40                       # was 8 when planned
  require_slide_region_when_visually_dependent: true

# ── Rubric: what "compelling" means. This is the tunable surface. ──
rubric:
  - id: hook_strength
    weight: 3
    question: >
      Within the first 3.0 seconds, is there a claim, number, tension, or question
      that makes a scrolling stranger stop? Not a preamble. Not a greeting.
    anchors:
      1: "Starts mid-thought or in filler: 'So, um, yeah, and then…'"
      3: "Interesting, but the point arrives only after 5+ seconds."
      5: "The first sentence IS the hook. A stranger knows the stakes immediately."

  - id: standalone
    weight: 3
    question: >
      Does this make full sense with zero prior context? Penalize unresolved
      references: 'as I mentioned', 'this slide', 'he just said', 'the third one'.
    anchors:
      1: "Incomprehensible without the preceding 10 minutes."
      3: "Mostly self-contained; one soft back-reference."
      5: "Fully self-contained. Defines its own terms."

  # single_idea (2) · payoff (2) · specificity (2) · quotability (1) · housekeeping (1)
  # …each with the same question/anchors shape.

scoring:
  method: weighted_mean
  min_weighted_score: 3.8
  vetoes:                             # any single failure kills the clip regardless of the mean
    hook_strength: 4
    standalone: 4

diversity:
  max_per_theme: 2                    # model labels `why.theme`; lint enforces the cap
```

Reading *why* a clip was picked means reading `why.scores` — per-criterion score plus a one-line evidence quote. Changing the rule means editing this YAML and re-running `ms plan`.

### Lint rules (`ms lint`) — how the AI is held to the contract

The highest-value check first, because it is the failure mode that actually happens:

1. **No hallucinated time.** Every `start`/`end` must correspond to a real word boundary in `words.json` within tolerance, and `source_text` must match the `words.json` span verbatim after normalization. This catches invented timestamps and invented quotes.
2. `--fix` snaps `start`/`end` to the nearest word/sentence boundary and applies `pad_in`/`pad_out`, writing back in place.
3. Every gate in `gates` checked against `words.json` + `silence.json`.
4. Every rubric `id` present in every clip's `why.scores`; `weighted_score` recomputed and compared; `vetoes` and `min_weighted_score` applied.
5. `visual_dependency: true` → the clip's layout must reference a region of `kind: "slide"`. *(This is the payoff of the region model: "the clip depends on what's on screen" becomes machine-checkable.)*
6. Referential integrity — every region id in every layout exists in `source.regions`; layout spans are ordered and within clip bounds; clip ids unique and well-formed; clips non-overlapping.

`ms render` refuses to run on a job that fails lint.

---

## Module layout

```
makeshorts/
  cli.py                    # typer: prepare · plan · lint · render · caps
  jobs.py                   # slug derivation, job dir paths
  config.py                 # load+validate config/*.yaml (pydantic)

  prepare/                  # ── MECHANICAL. No AI, ever. ──
    probe.py                # ffprobe → media.json
    transcribe.py           # faster-whisper → words.json (word-level timestamps)
    silence.py              # silencedetect → silence.json
    regions.py              # tile-grid temporal variance + cropdetect → regions.json (proposal)
    artifacts.py            # transcript.txt (timestamped), raw.srt

  select/                   # ── EDITORIAL. AI-authored, human-editable. ──
    criteria.py             # load/validate criteria.yaml
    prompt.py               # criteria.yaml + transcript.txt + regions.json → PROMPT.md
    schema.py               # pydantic models for clips.json — THE contract
    lint.py                 # the six rule groups above
    snap.py                 # boundary snapping used by lint --fix

  render/
    engine.py               # RenderEngine protocol + registry. Engines register by name.
    ffmpeg_engine.py        # renders with ffmpeg; the default
    resolve_engine.py       # builds a DaVinci Resolve timeline per clip; plan is pure
    resolve_driver.py       # executes that plan via Resolve's scripting API, in a child process
    layout.py               # pure: normalized rects + mode → pixel geometry. Heavily unit-tested.
    caps.py                 # probe installed ffmpeg for libass/drawtext/videotoolbox; cached
    captions/
      cues.py               # words.json → cues. Backend-agnostic, shared.
      base.py               # CaptionBackend protocol
      ass_backend.py        # subtitles filter path (unused on this machine)
      pillow_backend.py     # RGBA PNG per cue + overlay enable='between(t,a,b)'

config/
  render.yaml               # codecs, crf, encoder choice, output dir — ffmpeg's business
  criteria.yaml             # the rubric
  styles.yaml               # caption styles: font, size_pct, fill, stroke, highlight, safe area

jobs/<slug>/
  source.mp4  media.json  words.json  silence.json  regions.json
  transcript.txt  raw.srt  PROMPT.md  clips.json
  out/<slug>--<clip.id>.mp4        # e.g. acme-q3-webinar--01-cac-payback-math.mp4
  out/<slug>--<clip.id>.json       # receipt: engine, ffmpeg version, command, source hash
```

**Swappability contract** — `render/engine.py`:

```python
class RenderEngine(Protocol):
    name: str
    def capabilities(self) -> set[str]: ...          # {"hero_inset", "karaoke_captions", …}
    def render(self, clip: Clip, source: Path, out: Path, cfg: RenderConfig) -> Receipt: ...
```

Everything ffmpeg-shaped lives under `render/`. `select/` and `prepare/` never import it. A second engine implements this protocol and registers a name; `ms render --engine X` selects it. `layout.py` is pure geometry and is shared by any engine.

### Notable implementation details

- **Pillow caption backend:** `cues.py` groups `words.json` into cues (max chars/line, max lines, sentence-aware). Each cue renders to an RGBA PNG with the active word highlighted; ffmpeg composites with `overlay=…:enable='between(t,s,e)'`. Overlay stages are chunked to keep the filtergraph manageable on long clips.
- **Mid-clip layout spans** render as separate **video-only** segments, concatenated with the concat demuxer, then muxed against **one continuous audio stream** for the whole clip. Rendering audio per-segment would produce audible seams at every layout change; this avoids the problem entirely rather than trying to crossfade it.
- **Region detection** samples frames on a tile grid and computes per-tile temporal variance: camera regions vary continuously, slide/screenshare regions are near-static with step changes (which also yields `change_points`, useful to the editor). Output carries `confidence`; low confidence means the editorial step is expected to correct it.
- `faster-whisper` on Apple Silicon runs CTranslate2 on CPU (no Metal/CoreML). Default to `distil-large-v3` int8; `medium` as the faster fallback. Word timestamps are required — `word_timestamps=True`.
- Encoder default `libx264` for determinism and portability; `h264_videotoolbox` available in `render.yaml` for speed.

---

## Build order

Each phase ends in something inspectable.

1. **Skeleton + mechanical stage.** `pyproject.toml` (uv), `config.py`, `jobs.py`, `prepare/*`, `ms prepare`, `ms caps`. → Real `words.json`, `silence.json`, `regions.json`, `transcript.txt`, `raw.srt` on a real webinar. **Checkpoint: inspect these before any AI or render code exists.**
2. **Contract + gate.** `select/schema.py`, `criteria.py`, `prompt.py`, `snap.py`, `lint.py`. → `PROMPT.md` generated from the rubric; lint proven against deliberately-broken fixture `clips.json` files.
3. **Editorial pass.** Read `PROMPT.md` in Claude Code, write a real `clips.json`, run `ms lint`. Tune `criteria.yaml` and repeat — cheap, no rendering involved.
4. **Render.** `engine.py`, `layout.py`, `caps.py`, `captions/*`, `ffmpeg_engine.py`, `ms render`. Single-span first, then multi-span concat.
5. **Polish.** Receipts, `--only` clip selection, re-render idempotency.

---

### Parallelizing across agents

Two files are shared contracts and must land **before** any parallel work, or agents will invent conflicting shapes:

- `select/schema.py` — the `clips.json` pydantic models
- `render/engine.py` — the `RenderEngine` protocol

Write those two first, alone. After that these tracks are genuinely independent — different directories, no shared imports:

| Track | Owns | Depends on |
|---|---|---|
| A — mechanical | `prepare/*` (probe, transcribe, silence, regions, artifacts) | nothing |
| B — contract & gate | `select/*` (criteria, prompt, snap, lint) | schema.py |
| C — geometry & captions | `render/layout.py`, `render/caps.py`, `render/captions/*` | schema.py, engine.py |
| D — config & CLI | `config/*.yaml`, `config.py`, `jobs.py`, `cli.py` | schema.py |

`render/ffmpeg_engine.py` is the join point — it consumes C and is the last thing built. Track B's linter needs Track A's `words.json` shape, so agree that shape in `schema.py` up front rather than letting B guess it.

Assign one agent per track. Do not split a track across agents; each track is small enough to hold in one context, and the seams inside a track are tighter than the seams between them.

## Verification

- **Unit, pure functions:** `layout.py` geometry (every mode × region config, including `frame` fallback and `contain_blur`), `cues.py` grouping, `snap.py` boundary snapping, weighted scoring.
- **Lint, fixture-driven:** hand-written broken `clips.json` files — hallucinated timestamp, `source_text` mismatch, missing rubric key, wrong `weighted_score`, veto violation, `visual_dependency` without a slide region, unknown region id, overlapping clips. Each must fail with a specific message.
- **Render, synthetic source:** build a deterministic two-up fixture with `ffmpeg -f lavfi` (`testsrc2` beside a static `color` + burned counter, plus `sine` audio) — no large binary fixture in the repo. Assert via `ffprobe` that outputs are exactly 1080×1920, correct duration, correct stream count. Assert a multi-span clip's duration equals the sum of its spans and that audio is continuous.
- **End-to-end on real input:** `ms prepare` a real webinar → read `transcript.txt` and `regions.json` → write `clips.json` → `ms lint` → `ms render` → watch the clips.
- **Engine-agnosticism, enforced:** a test that greps `select/` and `prepare/` for ffmpeg vocabulary (`crop=`, `-vf`, `libx264`, `overlay=`, `scale=`) and fails on a hit. Plus a schema test asserting no codec/filter/path keys exist anywhere in `clips.json`.

## Non-goals

Auto-posting or platform APIs; music beds; B-roll; a GUI; compilation output.

*(Multi-source input and transitions were both listed here as non-goals and were built anyway — see below.)* `full` as a layout mode (subsumed by `focus` on the implicit `frame` region).

---

## What changed during implementation

Every divergence below was forced by contact with a real 58-minute Zoom
recording. They are recorded because the *reason* generalises even where the
specific number does not.

### Multi-source input — was a non-goal, was built

A Zoom cloud recording exports several frame-aligned renders of the same
meeting. Measured on a real one, the speaker occupied:

| view | speaker pixels | upscale to fill 1080 wide |
|---|---|---|
| `_avo_` active speaker | 1280×720 | 2.67× |
| `_gvo_` gallery | 640×360 | 5.35× |
| `_gallery_` PiP tile | 322×181 | 10.7× |
| `_as_` shared screen | *no camera at all* | — |

The sharp slides and the only usable face were in **different files**. No
single-source choice could produce a clip with both, so the non-goal had to go.

It cost less than expected, and that is the design paying off: `source` became
a named `sources` map, regions gained a `source` field — and **layouts did not
change at all**, because they already referred to regions by name. Old
single-source edit lists still validate untouched. New lint rules
(`source.clip_out_of_range`, `source.duration_mismatch`) check the assumption
that makes mixing safe: that the files share a clock.

### Three gate values were wrong, and the data said so

- **`max_internal_silence` 1.2 → 2.0s.** The original figure was a guess. It
  rejected eight of sixteen good clips for pauses of 1.2–1.9s, every one a
  breath at a clause boundary. Measured, that talk's median inter-word pause
  was 0.76s and the 90th percentile 1.55s — so 1.2 was cutting into the top
  fifth of ordinary phrasing. 2.0s is where the distribution turns (90 spans
  over the threshold become 23), and it barely moves when the silence detector
  threshold changes, so it is a property of speech and not of the detector.
- **`max_clips` 8 → 40.** Arbitrary. How many good clips a talk holds is a
  property of the talk. `diversity.max_per_theme` is the cap that actually
  protects quality.
- **`duration.min` 20 → 15s.** The tightest moments are often the shortest.

### A gate the plan never imagined: `gate.opens_on_filler`

A cold open beginning on "So", "And" or "Um" spends the hook window saying
nothing — and it is mechanically checkable, so the rubric should not have to
notice it by hand every time. Added after five of sixteen clips shipped with
openings like *"And they forklift…"* despite being scored and passed. Four had
a markedly better opener one sentence away.

Deliberately **not** in `FIXABLE_RULES`: trimming the leading word leaves the
clip mid-sentence, and picking a different span is an editorial decision about
what the clip *is*.

### Region detection localises content, not subjects

The plan assumed detected regions would be enough to frame a speaker. They are
not. Detection correctly reports "this whole frame is a camera" — true, and
useless when the subject sits off-centre, which cropped the presenter's face in
half.

Worse, the subject **moved**: measured across the talk, his face drifted from
0.688 to 0.761 of frame width. No fixed centre suits every clip.

The fix was entirely in `clips.json` — a region rect roughly 1.2× wider than a
bare 9:16 crop, centred on the measured face position, with
`fit: contain_blur`. The extra width both adds real pixels (`cover` always
crops to 9:16, so widening it only pans) and makes the residual drift stop
mattering. **No code changed**, which is the clearest evidence the
mechanical/editorial seam is in the right place.

### Captions burn in one pass, not per span

The plan had captions applied per layout span. That would cut a caption cue in
half whenever a layout change landed mid-sentence. They are instead burned once
over the concatenated clip — the same reasoning the plan already applied to
audio seams, which turns out to apply to captions too.

### Transitions and an outro bumper — also once non-goals

Both came from presenting the clips: hard cuts between layouts read as abrupt,
and a set of clips wants a consistent ending. They fit the existing pattern
without a new concept: a span names a `transition` preset, `render.yaml` says
what that preset is. Which cuts deserve one is editorial; what a dissolve looks
like is presentation.

Edge fades and the outro have no `clips.json` counterpart at all, because they
are uniform across a set rather than decisions about a particular clip. The
outro is a supplied video file rather than a generated card — the design
belongs in a design tool, and normalising an existing video is far less code
than reimplementing motion graphics badly.

The one genuinely tricky part is timing. `xfade` *consumes* its overlap, so
joining segments of length La and Lb with a D-second crossfade yields
`La + Lb - D`. Each span followed by a transition is therefore rendered D
seconds longer, and the xfade offset pulls it back. Get it wrong and every
transition silently shortens the clip and drifts it against its own audio —
which no test asserting "it rendered" would catch.

Compilation transitions stay out of scope: there is no compilation output yet,
so it would be building the transition before the thing it transitions
between.
