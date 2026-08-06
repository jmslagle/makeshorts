"""Build the lint fixtures from one hand-written transcript.

Run with `.venv/bin/python tests/select/fixtures/_build_fixtures.py`. The
outputs are committed; this script exists so they can be regenerated and so it
is obvious that words.json, silence.json and clips.json all describe the same
imaginary webinar. Hand-maintaining three files that must agree to the
millisecond is how fixtures come to be quietly wrong.

The transcript is written here as sentences. Timings are synthesised with a
crude length-proportional model — the linter cares only that the numbers are
internally consistent and that sentence flags land where sentences do.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[2]

# ── timing model ───────────────────────────────────────────────────────────
WORD_BASE = 0.14
WORD_PER_CHAR = 0.05
WORD_GAP = 0.05
SENTENCE_GAP = 0.50

PAD_IN = 0.12
PAD_OUT = 0.25

PAUSE = "__pause__"  # ("__pause__", seconds) between sentences


# ── the transcript ─────────────────────────────────────────────────────────
# Blocks marked with a clip id become clips; the rest is the webinar around
# them — filler that gives the clips somewhere to not be, and gaps that give
# min_separation something to measure.

BLOCKS: list[dict] = [
    {
        "clip": None,
        "gap_before": 0.0,
        "sentences": [
            "Alright, I think everyone can see the deck now.",
            "We will keep questions until the end, and yes, we will send the slides out.",
        ],
    },
    {
        "clip": "01-cac-payback-math",
        "gap_before": 4.0,
        "sentences": [
            "Eighteen months.",
            "That is the number that kills companies, and almost nobody checks it.",
            "If your CAC payback runs past eighteen months, you are financing growth "
            "out of a hole you cannot see.",
            (PAUSE, 2.6),
            "We looked at four hundred deals last year, and the pattern held every single time.",
            "The ones that died had a payback of twenty two months and a churn rate "
            "of four percent.",
            "Eighteen months is where companies die.",
        ],
    },
    {
        "clip": None,
        "gap_before": 5.0,
        "sentences": [
            "Someone in the chat is asking about the sample, and I will come back to that.",
            "Let me get through the hiring part first, because it is the one people argue with.",
            "There is a poll coming up in a minute as well.",
        ],
    },
    {
        "clip": "02-first-sales-hire",
        "gap_before": 4.0,
        "sentences": [
            "Do not hire a head of sales before you can sell the thing yourself.",
            "I have watched eleven companies make that exact mistake in three years.",
            "You cannot debug a sales process you have never run, and a new VP cannot either.",
            "The number I use is thirty closed deals by a founder before the first sales hire.",
            "Thirty is not a magic number, it is just long enough to stop guessing.",
        ],
    },
    {
        "clip": None,
        "gap_before": 5.0,
        "sentences": [
            "Okay, the poll is open, give it a second.",
            "While that runs, let me take the churn question that came in earlier.",
            "I promise this connects back to the payback math from the beginning.",
        ],
    },
    {
        "clip": "03-churn-is-pricing",
        "gap_before": 4.0,
        "sentences": [
            "Churn is a pricing problem wearing a product costume.",
            "When customers leave in month four, they did not hate the software, "
            "they hated the bill.",
            "Move one dollar of that price into onboarding and watch what happens to retention.",
            "We did exactly that in two thousand twenty four and cut logo churn by a third.",
            "That is the whole trick, and it costs almost nothing to try.",
        ],
    },
    {
        "clip": None,
        "gap_before": 5.0,
        "sentences": [
            "That is time, thanks everyone for coming.",
            "The recording and the deck go out tomorrow morning.",
        ],
    },
]


# ── editorial content for the three clips ──────────────────────────────────

CLIP_META: dict[str, dict] = {
    "01-cac-payback-math": {
        "title": "The CAC payback number nobody checks",
        "theme": "unit-economics",
        "one_line": "Contrarian claim on a hard number, resolves inside the clip, opens cold.",
        "scores": {
            "hook_strength": (5, "First two seconds are a bare number plus a stake."),
            "standalone": (5, "Defines CAC payback inline; no back-references."),
            "single_idea": (4, "One claim, one worked example."),
            "payoff": (5, "Closes on the corrective rule it opened with."),
            "specificity": (5, "Eighteen months, four hundred deals, twenty two months, four percent."),
            "quotability": (4, "'Eighteen months is where companies die.'"),
            "housekeeping": (5, "No logistics, no chat wrangling."),
        },
        "visual_dependency": False,
        "speaker": "cam_a",
        "layout": {"mode": "focus", "region": "cam_a", "fit": "cover"},
    },
    "02-first-sales-hire": {
        "title": "Do not hire a head of sales yet",
        "theme": "hiring",
        "one_line": "Direct instruction, a count, and a threshold a founder can act on today.",
        "scores": {
            "hook_strength": (5, "Opens on the imperative: 'Do not hire a head of sales'."),
            "standalone": (4, "One soft reference to 'that exact mistake', resolved immediately."),
            "single_idea": (5, "One rule, developed and closed."),
            "payoff": (4, "Delivers the thirty-deal threshold."),
            "specificity": (5, "Eleven companies, three years, thirty deals."),
            "quotability": (4, "'You cannot debug a sales process you have never run.'"),
            "housekeeping": (5, "No poll talk, no crosstalk."),
        },
        "visual_dependency": True,
        "speaker": "cam_a",
        "layout": [
            {"at": 0.0, "mode": "focus", "region": "cam_a", "fit": "cover"},
            {
                "at": 8.0,
                "mode": "hero_inset",
                "hero": "slides",
                "inset": "cam_a",
                "inset_corner": "bottom_right",
                "inset_scale": 0.28,
            },
        ],
    },
    "03-churn-is-pricing": {
        "title": "Churn is a pricing problem",
        "theme": "unit-economics",
        "one_line": "Reframing plus a cheap experiment with a real result attached.",
        "scores": {
            "hook_strength": (4, "'Wearing a product costume' lands, but it is a metaphor first."),
            "standalone": (4, "Self-contained; the churn question it answers is not needed."),
            "single_idea": (4, "One reframing, one experiment."),
            "payoff": (4, "Ends on the result and the cost of trying."),
            "specificity": (4, "Month four, one dollar, two thousand twenty four, a third."),
            "quotability": (3, "The opening line needs its follow-up to make sense."),
            "housekeeping": (5, "The poll aside stays outside the clip."),
        },
        "visual_dependency": False,
        "speaker": "cam_a",
        "layout": [
            {"at": 0.0, "mode": "focus", "region": "frame", "fit": "contain_blur"},
            {"at": 10.0, "mode": "stack", "regions": ["cam_a", "slides"]},
        ],
    },
}

WEIGHTS = {
    "hook_strength": 3,
    "standalone": 3,
    "single_idea": 2,
    "payoff": 2,
    "specificity": 2,
    "quotability": 1,
    "housekeeping": 1,
}


# ── build ──────────────────────────────────────────────────────────────────


def word_duration(token: str) -> float:
    core = re.sub(r"[^\w]", "", token)
    return round(WORD_BASE + WORD_PER_CHAR * max(len(core), 1), 3)


def build_words() -> tuple[list[dict], dict[str, tuple[int, int]], list[tuple[float, float, str]]]:
    words: list[dict] = []
    clip_ranges: dict[str, tuple[int, int]] = {}
    dead_air: list[tuple[float, float, str]] = []
    t = 100.0

    for block in BLOCKS:
        if block["gap_before"]:
            dead_air.append((t, t + block["gap_before"], "gap"))
            t += block["gap_before"]
        first = len(words)
        for item in block["sentences"]:
            if isinstance(item, tuple) and item[0] == PAUSE:
                dead_air.append((t, t + item[1], "pause"))
                t += item[1]
                continue
            tokens = item.split()
            for i, tok in enumerate(tokens):
                dur = word_duration(tok)
                words.append(
                    {
                        "text": tok,
                        "start": round(t, 3),
                        "end": round(t + dur, 3),
                        "sentence_start": i == 0,
                        "sentence_end": i == len(tokens) - 1,
                    }
                )
                t = round(t + dur + WORD_GAP, 3)
            t = round(t - WORD_GAP + SENTENCE_GAP, 3)
        t = round(t - SENTENCE_GAP, 3)
        if block["clip"]:
            clip_ranges[block["clip"]] = (first, len(words) - 1)
    return words, clip_ranges, dead_air


def pad(words: list[dict], first: int, last: int) -> tuple[float, float]:
    start = words[first]["start"] - PAD_IN
    if first > 0:
        start = max(start, words[first - 1]["end"])
    end = words[last]["end"] + PAD_OUT
    if last + 1 < len(words):
        end = min(end, words[last + 1]["start"])
    return round(start, 3), round(end, 3)


def span_text(words: list[dict], first: int, last: int) -> str:
    return " ".join(w["text"] for w in words[first : last + 1])


def weighted(scores: dict[str, tuple[int, str]]) -> float:
    total = sum(WEIGHTS.values())
    return round(sum(WEIGHTS[k] * v[0] for k, v in scores.items()) / total, 2)


def dump_words(words: list[dict]) -> str:
    """One word per line: a fixture nobody can read is a fixture nobody
    checks."""
    body = ",\n".join("    " + json.dumps(w) for w in words)
    head = '{\n  "model": "distil-large-v3",\n  "language": "en",\n  "words": [\n'
    return head + body + "\n  ]\n}\n"


def main() -> None:
    words, clip_ranges, dead_air = build_words()
    source_duration = round(words[-1]["end"] + 30.0, 3)

    # silence.json: the dead air between blocks, plus part of the long pause
    # inside clip 01. The declared span is always shorter than the true gap --
    # a silence detector trims to where the level actually drops, and the
    # mid-clip pause carries breath either side of the quiet second. That
    # leaves the baseline just under max_internal_silence, so the broken
    # variant only has to widen one span.
    spans = []
    for a, b, kind in dead_air:
        if kind == "pause":
            mid = (a + b) / 2
            spans.append({"start": round(mid - 0.5, 3), "end": round(mid + 0.5, 3)})
        elif b - a >= 2.0:
            spans.append({"start": round(a + 0.05, 3), "end": round(b - 0.05, 3)})
    silence = {"threshold_db": -32.0, "min_duration": 0.6, "spans": spans}

    # The same file with that one pause reported at its full length: 2.5s of
    # dead air sitting in the middle of clip 01.
    excessive = json.loads(json.dumps(silence))
    for a, b, kind in dead_air:
        if kind == "pause":
            for s in excessive["spans"]:
                if a < s["start"] < b:
                    s["start"], s["end"] = round(a + 0.05, 3), round(b - 0.05, 3)
    (HERE / "silence.excessive.json").write_text(json.dumps(excessive, indent=2) + "\n")

    criteria_src = (REPO / "config" / "criteria.yaml").read_text()
    (HERE / "criteria.yaml").write_text(
        "# FROZEN COPY of config/criteria.yaml, pinned so the fixtures' recorded\n"
        "# sha256 stays valid when the real rubric is tuned. Regenerate with\n"
        "# tests/select/fixtures/_build_fixtures.py.\n" + criteria_src
    )
    import hashlib

    sha = hashlib.sha256((HERE / "criteria.yaml").read_bytes()).hexdigest()
    version = re.search(r'^version:\s*"([^"]+)"', criteria_src, re.M).group(1)

    clips = []
    for cid, (first, last) in clip_ranges.items():
        meta = CLIP_META[cid]
        start, end = pad(words, first, last)
        clips.append(
            {
                "id": cid,
                "title": meta["title"],
                "start": start,
                "end": end,
                "source_text": span_text(words, first, last),
                "why": {
                    "one_line": meta["one_line"],
                    "theme": meta["theme"],
                    "scores": {
                        k: {"score": v[0], "evidence": v[1]} for k, v in meta["scores"].items()
                    },
                    "weighted_score": weighted(meta["scores"]),
                    "rejected_alternatives": [],
                },
                "visual_dependency": meta["visual_dependency"],
                "speaker": meta["speaker"],
                "layout": meta["layout"],
                "captions": {"style": "pill-karaoke", "position": "lower_third"},
                "audio": {"normalize": True},
            }
        )

    doc = {
        "schema_version": "1.0",
        "job": "acme-q3-webinar",
        "source": {
            "path": "jobs/acme-q3-webinar/source.mp4",
            "duration": source_duration,
            "resolution": [1920, 1080],
            "regions": [
                {
                    "id": "cam_a",
                    "kind": "speaker",
                    "label": "Dana Reyes",
                    "rect": [0.0, 0.0, 0.5, 1.0],
                },
                {"id": "slides", "kind": "slide", "rect": [0.5, 0.0, 0.5, 1.0]},
            ],
        },
        "output": {"width": 1080, "height": 1920, "fps": 30},
        "criteria_ref": {"file": "config/criteria.yaml", "version": version, "sha256": sha},
        "clips": clips,
    }

    (HERE / "words.json").write_text(dump_words(words))
    (HERE / "silence.json").write_text(json.dumps(silence, indent=2) + "\n")
    (HERE / "clips.valid.json").write_text(json.dumps(doc, indent=2) + "\n")

    # -- deliberately broken variants, written out as real files --------------

    # Times invented out of the air: plausible-looking, ~37s further on, in the
    # dead air between blocks. Nothing in words.json is near them.
    bad = json.loads(json.dumps(doc))
    bad["clips"][0]["start"] = round(doc["clips"][0]["start"] + 37.4, 3)
    bad["clips"][0]["end"] = round(doc["clips"][0]["end"] + 37.4, 3)
    (HERE / "clips.hallucinated-time.json").write_text(json.dumps(bad, indent=2) + "\n")

    # Right span, wrong words: a paraphrase of the kind produced by writing the
    # quote from memory rather than copying it out of words.json.
    bad = json.loads(json.dumps(doc))
    bad["clips"][0]["source_text"] = (
        bad["clips"][0]["source_text"]
        .replace("four hundred deals", "five hundred deals")
        .replace("and almost nobody checks it", "")
        .replace("Eighteen months is where companies die.", "Eighteen months is where startups die.")
    )
    (HERE / "clips.text-mismatch.json").write_text(json.dumps(bad, indent=2) + "\n")

    # Edges nudged off their boundaries by more than the tolerance but less
    # than a fabrication, and sitting in the dead air rather than inside a
    # word: exactly the file `--fix` exists for.
    bad = json.loads(json.dumps(doc))
    bad["clips"][0]["start"] = round(doc["clips"][0]["start"] - 0.18, 3)
    bad["clips"][0]["end"] = round(doc["clips"][0]["end"] + 0.20, 3)
    (HERE / "clips.unsnapped.json").write_text(json.dumps(bad, indent=2) + "\n")

    print(f"words: {len(words)}  source_duration: {source_duration}")
    for c in clips:
        mid = (c["start"] + c["end"]) / 2
        print(f"  {c['id']:<24} {c['start']:>8.3f} – {c['end']:>8.3f}  "
              f"dur {c['end'] - c['start']:>6.2f}  mid {mid:>8.2f}  "
              f"weighted {c['why']['weighted_score']}")
    mids = [(c["start"] + c["end"]) / 2 for c in clips]
    for i in range(len(mids)):
        for j in range(i + 1, len(mids)):
            print(f"  separation {i + 1}->{j + 1}: {mids[j] - mids[i]:.2f}")
    print(f"  silence spans: {spans}")


if __name__ == "__main__":
    main()
