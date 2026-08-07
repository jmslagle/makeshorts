# Working on makeshorts

Read this before changing code. It records the invariants that make the design
work and the mistakes that have already been made once.

## The one idea

Three stages, three trust levels, one artifact between each:

```
prepare/   MECHANICAL   deterministic, no AI      -> words.json, silence.json, regions.json
select/    EDITORIAL    AI-authored, reviewable   -> clips.json
render/    MECHANICAL   deterministic             -> mp4 + receipt
```

`clips.json` is the product of the thinking step. Every design rule below
exists to keep it readable, arguable and re-renderable by a human who did not
run the AI.

## Invariants — do not break these

**`clips.json` is engine-agnostic.** No codec, bitrate, preset, filter string,
pixel crop, font path, image path or output directory. Times are absolute
seconds; geometry is normalized 0–1; styles and branding are *names* resolved
at render time. If you want to add a field, ask whether a Resolve engine would
also need it. `tests/test_engine_agnostic.py` enforces this two ways and will
fail you.

**`select/` and `prepare/` never import `render/`.** Same test enforces it.
`prepare/` may shell out to ffmpeg — it is an ingest stage — but `select/` may
not even name it.

**Rubric text lives in `config/criteria.yaml`, never in Python.** `PROMPT.md`
is generated from it and `lint` reads its score keys from it, so the prompt and
the gate cannot drift apart. `tests/select/test_prompt.py` proves this by
swapping in a nonsense criterion and asserting the old text is gone. Adding a
criterion should require no code change at all.

**`layout.py` is pure.** No I/O, no ffmpeg vocabulary, stdlib + pydantic only.
Every engine shares it.

**`lint --fix` must never rewrite `source_text`.** Doing so would make every
file self-consistent *including* the ones where the model invented a quote,
which erases the single most valuable finding the linter produces. It prints
the verbatim replacement instead and lets a human paste it.

## Gotchas already paid for

**ffmpeg's `color` filter is an infinite source** unless given `:d=<duration>`,
and `-t` *before* `-i` bounds only a file input. An unbounded canvas plus
`overlay` yields an encode that never terminates — it hangs rather than fails,
so a test asserting on output would never fire. Duration is threaded through
`_span_filtergraph`; keep it that way.

**This project targets ffmpeg builds without `libass`/`libfreetype`**, where
`subtitles` and `drawtext` do not exist. `caps.py` probes and the caption layer
picks a backend. Do not add a code path that assumes either filter.

**Import optional submodules with `importlib.import_module`**, not
`from package import submodule`. A from-import reads the attribute already
bound on the parent package once the submodule has been imported anywhere in
the process, silently ignoring `sys.modules` — which broke test isolation in
`prepare/run.py` in a way that only appeared when the whole suite ran.

**Boundary checks must test containment, not proximity.** For a padded clip
end, the numerically nearest word edge is often the *next* word's onset, since
`pad_out` legitimately runs into the gap before it. Anchoring on proximity made
correct padding fail `end_on_sentence_end`, so `--fix` could never produce a
passing file. Ask "which word was being spoken", not "which edge is closest".

**Region detection classifies content, not subjects.** It answers "this area is
a slide, that one is a camera" — it does not locate a face. Framing a speaker
is an editorial decision expressed as a region rect in `clips.json`.

## Tests

```bash
uv run pytest          # 708 tests, ~20s
```

Media fixtures are synthesized with `ffmpeg -f lavfi`, never committed. Region
detection is tested against sources whose correct answer is known by
construction (a moving half beside a static half). Keep it that way — a
committed sample video is both a licensing problem and a repo-size problem.

## What is deliberately not here

Auto-posting, music beds, B-roll, transitions beyond hard cuts, a GUI. And
`full` is not a layout mode: it is `{"mode": "focus", "region": "frame",
"fit": "contain_blur"}`.
