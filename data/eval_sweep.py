"""
eval_sweep.py — produce a *measured* accuracy-vs-token-budget curve.

This is the honesty counterpart to the demo's simulated curve. It runs the real
LFM2.5-VL-450M over a labeled set of frames at every vision-token budget the
slider exposes, records the measured latency and the actual vision-token count,
and (optionally) scores each description against its label so you get a real
accuracy curve instead of the illustrative one shipped in simulated mode.

It talks to the SAME local server as the browser demo (server.py), so it exercises
the identical code path — no second, divergent inference path to keep in sync.

Usage
-----
1. Start the model server in another terminal:
       python server.py
2. Put labeled frames + labels in a manifest (see LABELS below or pass --manifest).
3. Run:
       python data/eval_sweep.py                 # latency + token sweep, all budgets
       python data/eval_sweep.py --judge         # also LLM-judge accuracy (needs a judge)

Output: data/sweep_results.json  — per (scene, budget): latency_ms, vision_tokens,
description, and (if judged) pass/fail. Aggregate into the curve however you display it.

Scoring is intentionally left as a stub: wire in a human pass or an LLM judge that
applies data/rubric.md. Do not fabricate an accuracy number — an unjudged run reports
latency + tokens only and leaves accuracy null.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path
from urllib import request as urllib_request

# The slider's budgets — keep in sync with web/index.html DATA.model.budget_stops
# (consumed there as the STOPS const) and data/scenes.json "budget_stops".
BUDGET_STOPS = [64, 96, 128, 160, 192, 224, 256, 512]

SERVER = "http://localhost:8000"
DATA_DIR = Path(__file__).resolve().parent
SCENES_DIR = DATA_DIR / "scenes"

# Minimal built-in manifest: the four scene types the demo ships with. Replace /
# extend with your ~20–40 labeled frames (see data/rubric.md).
LABELS = [
    {"scene": "package", "file": "package.jpg",
     "label": "A parcel is on the porch step; no person present."},
    {"scene": "person", "file": "person.jpg",
     "label": "One person standing at the front door."},
    {"scene": "two-people", "file": "two-people.jpg",
     "label": "Two people at the door."},
    {"scene": "empty-branch", "file": "empty-branch.jpg",
     "label": "Empty porch; a tree branch is moving. No person or package."},
]

PROMPT = "Describe the scene in one sentence."


def _b64(path: Path) -> str:
    # Return a data: URL, exactly like the browser's FileReader.readAsDataURL, so the
    # sweep hits the same _decode_image branch the UI does (identical input contract).
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _infer(image_b64: str, max_image_tokens: int) -> dict:
    body = json.dumps(
        {"image_b64": image_b64, "prompt": PROMPT, "max_image_tokens": max_image_tokens}
    ).encode("utf-8")
    req = urllib_request.Request(
        f"{SERVER}/infer", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib_request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def _judge(description: str, label: str) -> bool | None:
    """Apply data/rubric.md to decide pass/fail.

    STUB: returns None (unjudged). Wire in a human review pass or an LLM judge here —
    give it the label + the three rubric checks and have it return a strict boolean.
    Leaving it None keeps the run honest: latency/tokens are measured, accuracy is not
    invented.
    """
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="store_true",
                        help="also score each description (requires a wired-in judge)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="JSON list of {scene, file, label}; defaults to built-in LABELS")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text()) if args.manifest else LABELS

    missing = [m["file"] for m in manifest if not (SCENES_DIR / m["file"]).exists()]
    if missing:
        print(f"Missing frames in {SCENES_DIR}: {', '.join(missing)}", file=sys.stderr)
        print("Add your footage (see the repo README) before running the sweep.", file=sys.stderr)
        return 1

    results = []
    for item in manifest:
        image_b64 = _b64(SCENES_DIR / item["file"])
        for budget in BUDGET_STOPS:
            out = _infer(image_b64, budget)
            passed = _judge(out["description"], item["label"]) if args.judge else None
            row = {
                "scene": item["scene"],
                "budget": budget,
                "vision_tokens": out["vision_tokens"],
                "latency_ms": out["latency_ms"],
                "device": out["device"],
                "description": out["description"],
                "label": item["label"],
                "passed": passed,
            }
            results.append(row)
            status = "" if passed is None else (" PASS" if passed else " FAIL")
            print(f"{item['scene']:>13}  budget={budget:>4}  "
                  f"tok={out['vision_tokens']:>4}  {out['latency_ms']:>7.1f}ms{status}")

    out_path = DATA_DIR / "sweep_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} rows to {out_path}")
    if not args.judge:
        print("Accuracy not scored (run with --judge and a wired-in grader). "
              "Latency and token counts are measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
