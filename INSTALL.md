# Installing makeshorts

## Requirements

| | |
|---|---|
| Python | 3.12 or newer |
| [uv](https://docs.astral.sh/uv/) | package + venv manager |
| ffmpeg / ffprobe | any recent build — see [ffmpeg notes](#ffmpeg-notes) |
| Disk | ~1.5 GB for the Whisper model, plus room for jobs (a 1-hour job is ~300 MB before renders) |

No GPU is required. No paid or licensed dependency is used anywhere.

## Install

```bash
git clone https://github.com/jmslagle/makeshorts.git
cd makeshorts
uv venv
uv pip install -e .
```

That puts `ms` in `.venv/bin/`. Either activate the venv, or call it directly:

```bash
source .venv/bin/activate    # then: ms caps
# or
.venv/bin/ms caps
```

`ms` reads `config/` relative to the working directory, so **run it from the
repo root** — or point it elsewhere with `--config-dir` / `MS_CONFIG_DIR`.

## Verify

```bash
.venv/bin/ms caps      # what your ffmpeg can actually do
.venv/bin/ms jobs      # should print an empty job list
uv run pytest -q       # 708 tests, ~20s
```

If `ms caps` prints a caption backend and your layout capabilities, you are
ready.

## ffmpeg notes

**You do not need a special build.** Many distributions ship ffmpeg without
`libass` and `libfreetype`, which means the `subtitles` and `drawtext` filters
do not exist. That is a normal, supported configuration here: `ms caps` probes
your binary and the caption layer picks a backend to match.

| your ffmpeg | caption backend | karaoke |
|---|---|---|
| with libass | ASS subtitles (fewer moving parts) | yes |
| without libass | Pillow → RGBA PNG → `overlay` | yes |

Both burn captions into the video. The Pillow path needs only `crop`, `scale`
and `overlay`, which every build has.

Check yours:

```bash
ffmpeg -hide_banner -filters | grep -E '(^| )(subtitles|drawtext) '
```

macOS via Homebrew:

```bash
brew install ffmpeg
```

## Transcription model

The first `ms prepare` downloads a Whisper model — `distil-large-v3` by
default, about **1.4 GB**, cached in `~/.cache/huggingface/`. This happens once
and takes several minutes. Pre-fetch it if you would rather not wait mid-run:

```bash
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('distil-large-v3', device='cpu', compute_type='int8')"
```

`distil-large-v3` is **English-only**. For other languages use
`ms prepare --model large-v3`, which is multilingual and slower.

Transcription runs on CPU. CTranslate2 has no Metal backend, so Apple Silicon
does not use the GPU — expect roughly **4–5× realtime** (a 58-minute recording
took ~12 minutes on an M-series laptop). Transcription happens once per job;
`ms prepare --skip transcribe` re-runs the other stages for free afterwards.

## Configuration

Three files in `config/`, all optional to edit:

| file | what it controls |
|---|---|
| `criteria.yaml` | the rubric — what "compelling" means, gates, vetoes, diversity |
| `render.yaml` | codecs, quality, loudness, branding presets |
| `styles.yaml` | caption appearance |

A job may override any of them by dropping its own `config/render.yaml` (or
the others) inside `jobs/<slug>/config/`. Job config layers on top of the
repo's, so a three-line file is enough to change one thing — and since `jobs/`
is gitignored, local branding and tuning stay out of version control.

### Adding a watermark

The repo ships no logo. Put a transparent PNG in `assets/` (also gitignored)
and add a branding preset — see the commented example in `config/render.yaml`.
Pick a variant that suits the footage: a dark logo disappears over a dark
camera feed, a white one disappears over a white slide. If the image is
missing, `ms render` warns and renders without it rather than failing.

### Fonts

Caption styles reference a font path. The shipped styles use macOS system
fonts; on Linux or Windows, point `font` in `config/styles.yaml` at a font you
have — any TTF/OTF works, since Pillow does its own text rendering.

## Troubleshooting

**`error: config file not found: config/render.yaml`** — you are not in the
repo root. `cd` there, or set `MS_CONFIG_DIR`.

**`ms render` refuses to run** — that is the gate. `ms lint <slug>` explains
each finding and names the fix; `ms lint <slug> --fix` repairs the mechanical
ones. Nothing renders until lint passes, by design.

**A render seems to hang** — check `ms caps` first. If you are on a build with
unusual filter support, run with a single clip (`--only 01`) to isolate it.

**Transcription is slower than expected** — it is CPU-bound and single-job.
`--model medium` trades some accuracy for speed; `--skip transcribe` avoids
paying for it twice.

## Uninstall

```bash
rm -rf .venv                                    # the environment
rm -rf ~/.cache/huggingface/hub/models--Systran--faster-distil-whisper-large-v3
```

Job data lives in `jobs/` and is never written anywhere else.
