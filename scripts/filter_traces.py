#!/usr/bin/env python3
"""Rejection sampling: turn sampled thinking traces (sample_traces.py) into reasoning-SFT rows.

For each request, keep one trace (by default the median-length one; see --select) that finished and whose patch is correct: an exact match to the reference
(rewards.score, incl. relax/alts), or, for rows with deterministic checks (compile_checks.py), a passing grade, so a
valid alternative to the reference also counts. Requests with no correct trace are dropped (RL can pick them up).

    python scripts/filter_traces.py --dataset user/ds --split train --traces traces.jsonl --out sft_think.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from design_edit.checks import grade_checks  # noqa: E402
from design_edit.schema import strip_fences  # noqa: E402

LEAK = re.compile(r"\bhint\b|private note|\bnote\b says|reference (answer|patch|solution)|given answer|correct answer|i.m ignoring|"
                  r"provided answer|the answer (is )?(given|provided)", re.I)
KEEP = ("prompt", "before", "after", "kind", "instruction", "orig_instruction", "relax", "alts", "refs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Hub dataset or local .jsonl the traces were sampled from")
    ap.add_argument("--split", default="train")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reject-hint-leak", action="store_true",
                    help="for --hint traces: drop any whose text mentions the hint (it is trained without one)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="drop traces longer than this")
    ap.add_argument("--select", choices=["median", "longest", "shortest"], default="median",
                    help="which correct trace to keep. 'shortest' (used through v13) trains the model to think less, "
                         "which cost it badly on the harder eval v3")
    ap.add_argument("--review", action="store_true",
                    help="traces from sample_traces.py --review: build two-turn rows (see review_rows)")
    ap.add_argument("--judge", default="deepseek-ai/DeepSeek-V4.1-Flash",
                    help="--review: judge for items whose checks include {'judge': ...} criteria")
    args = ap.parse_args()
    if args.review:
        return review_rows(args)
    if args.dataset.endswith(".jsonl"):
        rows = [json.loads(line) for line in open(args.dataset)]
    else:
        from datasets import load_dataset

        rows = list(load_dataset(args.dataset, split=args.split))
    out, n_prompts, n_samples, n_correct, lengths = [], 0, 0, 0, []
    for line in open(args.traces):
        t = json.loads(line)
        r = rows[t["index"]]
        assert r["instruction"] == t["instruction"], "traces and dataset rows are out of order"
        n_prompts += 1
        good = []
        for s in t["samples"]:
            n_samples += 1
            ok = s["exact"]
            if not ok and r.get("checks"):
                ok = grade_checks(r["before"], strip_fences(s["text"]), r["checks"], r["allowed"])["pass"]
            if ok:
                n_correct += 1
                if args.reject_hint_leak and LEAK.search(s["text"]):
                    continue
                if s["finished"] and "<channel|>" in s["text"] and s["tokens"] <= args.max_tokens:
                    good.append(s)
        if good:
            good.sort(key=lambda s: s["tokens"])
            best = {"shortest": good[0], "longest": good[-1], "median": good[(len(good) - 1) // 2]}[args.select]
            lengths.append(best["tokens"])
            out.append({**{k: r.get(k, "") for k in KEEP},
                        "completion": [{"role": "assistant", "content": best["text"].strip()}]})
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    lengths.sort()
    print(f"{n_prompts} prompts, sample accuracy {n_correct / max(n_samples, 1):.3f}, "
          f"{len(out)} prompts with a usable correct trace ({len(out) / max(n_prompts, 1):.1%}); "
          f"median kept trace {lengths[len(lengths) // 2] if lengths else 0} tokens -> {args.out}")


def _final(s: dict) -> str:
    """The episode's final answer: the review answer, or the first one if the review came back empty (as at eval)."""
    rv = s.get("review_text") or ""
    return rv if strip_fences(rv).strip() else s["text"]


def review_rows(args) -> None:
    """Review episodes -> SFT rows that match eval_model.py --revise-rounds 1 exactly.

    Per prompt, among episodes whose FINAL patch is correct, keep the median-length one (by --select), then write:
      row B (always): prompt + [assistant: first patch only, user: review message] -> review answer (with thinking)
      row A (only if the FIRST answer was already correct): prompt -> first answer (with thinking)
    so the model never trains on a wrong first attempt, and learns both to fix and to confirm on review."""
    import json as _json

    from design_edit.revise import review_message

    if args.dataset.endswith(".jsonl"):
        rows = [_json.loads(line) for line in open(args.dataset)]
    else:
        from datasets import load_dataset

        rows = list(load_dataset(args.dataset, split=args.split))

    def correct(r: dict, text: str) -> bool:
        patch = strip_fences(text).strip()
        if not patch or not r.get("checks"):
            return False
        g = grade_checks(r["before"], patch, r["checks"], r["allowed"], r.get("design_system"))
        if not g["pass"]:
            return False
        if '"judge"' in r["checks"]:  # subjective criteria: ask the judge on the full criteria
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from build_eval_v2 import grade

            from design_edit.schema import apply_patch, parse

            before = parse(r["before"])
            v = grade(r["design_system"], before, r["instruction"], _json.loads(r["criteria"]),
                      apply_patch(before, patch), model=args.judge)
            return bool(v) and all(v)
        return True

    out, stats = [], {"prompts": 0, "solved": 0, "fixed": 0, "confirmed": 0, "first_ok": 0, "episodes": 0}
    for line in open(args.traces):
        t = _json.loads(line)
        r = rows[t["index"]]
        assert r["instruction"] == t["instruction"], "traces and dataset rows are out of order"
        stats["prompts"] += 1
        good = []
        for s in t["samples"]:
            stats["episodes"] += 1
            first_ok = s["finished"] and correct(r, s["text"])
            stats["first_ok"] += first_ok
            fin = _final(s)
            if not s.get("review_finished") or not correct(r, fin):
                continue
            if s["tokens"] + s.get("review_tokens", 0) > 2 * args.max_tokens:
                continue
            good.append({**s, "first_ok": first_ok, "total": s["tokens"] + s.get("review_tokens", 0)})
        if not good:
            continue
        stats["solved"] += 1
        good.sort(key=lambda s: s["total"])
        best = {"shortest": good[0], "longest": good[-1], "median": good[(len(good) - 1) // 2]}[args.select]
        stats["confirmed" if best["first_ok"] else "fixed"] += 1
        keep = {k: r.get(k, "") for k in KEEP}
        prompt = r["prompt"]
        conv = prompt + [{"role": "assistant", "content": strip_fences(best["text"])},
                         {"role": "user", "content": review_message(prompt, best["text"])}]
        out.append({**keep, "prompt": conv, "turn": "review", "episode": "confirmed" if best["first_ok"] else "fixed",
                    "completion": [{"role": "assistant", "content": _final(best).strip()}]})
        if best["first_ok"]:
            out.append({**keep, "prompt": prompt, "turn": "first", "episode": "confirmed",
                        "completion": [{"role": "assistant", "content": best["text"].strip()}]})
    Path(args.out).write_text("".join(_json.dumps(x) + "\n" for x in out))
    print(f"{stats}; first-answer accuracy {stats['first_ok'] / max(stats['episodes'], 1):.3f}; "
          f"{len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
