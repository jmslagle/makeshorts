# makeshorts

Turn a long-form webinar recording into a set of short vertical clips for social.

The point of this tool is a seam. Transcription, silence detection and region
detection are mechanical: deterministic, re-runnable, no AI. Choosing *which
45 seconds are worth posting* is judgment, and it produces one reviewable file.
Rendering is mechanical again, and consumes that file without re-running any of
the thinking.

```
ms prepare <input>   →  words.json · silence.json · regions.json · transcript.txt · PROMPT.md
   ── you read PROMPT.md and write clips.json ──
ms lint <slug>       →  the gate; nothing renders until this passes
ms render <slug>     →  out/<slug>--<clip.id>.mp4 + a receipt beside it
```

`clips.json` is the deliverable of the thinking step. A human can read it,
disagree with it, edit it, and re-render **without re-running the AI or the
transcription**.

---

## Why it is built this way

**The edit list is engine-agnostic.** ffmpeg is today's renderer, not a concept
`clips.json` knows about. No codec, bitrate, preset, filter string, pixel crop,
font path or output directory appears anywhere in it. Times are absolute
seconds; geometry is normalized 0–1; styles and branding are *names* resolved at
render time. A second engine implements the `RenderEngine` protocol, registers a
name, and `ms render --engine X` selects it — with the same edit list.

Two tests enforce this rather than trusting it: one greps `select/` and
`prepare/` for ffmpeg vocabulary, the other walks a fully-populated `clips.json`
and fails if anything engine-shaped appears.

**Selection criteria are data, not a prompt string.** `config/criteria.yaml`
holds weighted criteria with 1/3/5 anchors, hard gates, veto thresholds and a
diversity cap. `PROMPT.md` is *generated from it*, and the score keys `ms lint`
demands are read from it. Edit the YAML and the prompt, the rubric and the gate
move together. A test proves it by swapping in a nonsense criterion and
asserting the old text is gone.

**Layouts compose over named regions**, so speaker count is irrelevant to the
code. One speaker, two speakers side by side, and speaker-plus-slides are the
same three modes over different regions:

| mode | shape |
|---|---|
| `focus` | one region fills the frame (`fit: cover` or `contain_blur`) |
| `hero_inset` | one region fills, a second sits inset — slides + talking head |
| `stack` | regions tiled vertically |

A layout may be a list of timed spans, so a clip can cut from the speaker to a
slide partway through.

**Several source files, one clip.** A Zoom export gives frame-aligned renders of
the same meeting — sharp slides in one file, the only usable face in another.
`sources` names them; regions say which file they are measured against; layouts
are unchanged, because they already referred to regions by name. Lint checks
every source actually spans the clip and warns when durations disagree, because
mixing files is only safe while they share a clock.

---

## `ms lint` is the load-bearing piece

It is what makes the mechanical/editorial seam real rather than aspirational.
The highest-value rule first, because it is the failure that actually happens:

- **`time.mid_word` / `text.mismatch`** — every timestamp must land on a real
  word boundary in `words.json`, and `source_text` must match that span
  verbatim. This catches invented timestamps and invented quotes, and prints a
  word-level diff.
- **gates** — duration, sentence-boundary starts and ends, internal silence,
  separation between clips, clip count.
- **rubric** — every criterion scored, `weighted_score` recomputed and compared,
  vetoes and thresholds applied, theme diversity enforced.
- **`visual.no_slide_region`** — a clip marked `visual_dependency: true` must
  have a slide region in its layout. An editorial claim about meaning, checked
  by machine.
- **referential integrity** — region and source ids, span ordering, overlaps.
- **`source.clip_out_of_range`** — with several sources, every one must cover
  the clip.

`ms render` refuses to run on a job with errors. `ms lint --fix` snaps
timestamps to real boundaries but deliberately will **not** rewrite
`source_text` — doing so would make every file self-consistent including the
ones where the model invented a quote, erasing the most valuable finding.

---

## Install

Requires Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and `ffmpeg`.

```bash
uv venv && uv pip install -e .
ms caps          # what your ffmpeg can actually do
```

Full setup, model download, fonts and troubleshooting: **[INSTALL.md](INSTALL.md)**.

**Captions do not need a special ffmpeg build.** Many ffmpeg packages ship
without `libass`/`libfreetype`, which means no `subtitles` and no `drawtext`
filter. `ms caps` probes for this and the caption layer picks a backend
accordingly: Pillow renders each cue to an RGBA PNG and ffmpeg composites it
with timeline-gated `overlay`, which every build has. Word-level karaoke
highlighting works either way.

Transcription uses `faster-whisper`. On Apple Silicon CTranslate2 runs on CPU
(no Metal), which is still ~4.7× realtime with `distil-large-v3`.

---

## Use

```bash
# single file
ms prepare videos/webinar.mp4

# several frame-aligned views of one recording
ms prepare --source cam=videos/rec_avo.mp4 --source slides=videos/rec_as.mp4

ms plan   my-webinar        # regenerate PROMPT.md from criteria.yaml
ms lint   my-webinar        # exits non-zero on error
ms lint   my-webinar --fix  # snap timestamps to real boundaries
ms render my-webinar --only 01,03
ms jobs                     # what stage every job is at
```

Skip stages you have already paid for: `ms prepare ... --skip transcribe` lets
you re-run region detection without transcribing an hour of audio again.

## Configuration

| file | holds |
|---|---|
| `config/criteria.yaml` | the rubric — what "compelling" means, gates, vetoes, diversity |
| `config/render.yaml` | codecs, crf, loudness, branding presets — everything ffmpeg-shaped |
| `config/styles.yaml` | caption styles: font, size, colours, pill, safe area |

## Layout

```
makeshorts/
  artifacts.py          shapes of the mechanical artifacts
  prepare/              MECHANICAL — probe, transcribe, silence, regions
  select/               EDITORIAL — criteria, prompt, schema, snap, lint
  render/               layout geometry, caps probing, captions, engines
    engine.py           the RenderEngine protocol — the swappability contract
    layout.py           pure geometry, shared by every engine
```

`select/` and `prepare/` never import `render/`.

## Tests

```bash
uv run pytest
```

Fixtures are synthesized with `ffmpeg -f lavfi` rather than committed, so the
suite carries no binary media. Region detection is tested against sources whose
correct answer is known by construction.

## Documentation

| | |
|---|---|
| [INSTALL.md](INSTALL.md) | setup, ffmpeg notes, model download, troubleshooting |
| [CLAUDE.md](CLAUDE.md) | invariants and traps, for changing the code |
| [.claude/skills/makeshorts](.claude/skills/makeshorts/SKILL.md) | the workflow, for driving the tool on a new recording |
| [docs/PLAN.md](docs/PLAN.md) | design record — the forks, and what implementation changed |

## Non-goals

Auto-posting or platform APIs; music beds; B-roll; transitions beyond hard cuts;
a GUI.

## License

[AGPL-3.0](LICENSE). If you run a modified version as a network service, the
AGPL requires you to offer its source to your users.

Copyright is held by the author, so this can be relicensed to something more
permissive later; going the other way once outside contributors hold copyright
in merged changes is much harder.
