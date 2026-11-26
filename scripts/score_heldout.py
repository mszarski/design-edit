#!/usr/bin/env python3
"""Plan v14 go/no-go: solve rate on the held-out RL prompts, scored exactly like the RL reward's pass condition
(exact checks with the item's design system, plus no new rendered-geometry issue).

Sample a checkpoint first (k answers per held-out prompt), e.g.
    hf jobs uv run ... scripts/sample_traces.py --dataset user/design-edits-v14-rl --split heldout --k 4 \
        --temperature 1.0 --max-new-tokens 4096 --max-model-len 12288 --adapter <ckpt> --out-repo ... --out-name ...
then
    python scripts/score_heldout.py --dataset user/design-edits-v14-rl --traces ck25.jsonl [ck50.jsonl ...]
Prints the solve rate (with a 95% interval over answers), the base model's rate on the same prompts (column
base_solve_rate, from the re-pick at k=8), and median thinking words.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from design_edit.checks import grade_checks  # noqa: E402
from design_edit.geometry import Renderer  # noqa: E402
from design_edit.schema import apply_patch, parse, strip_fences  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--traces", nargs="+", required=True)
    a = ap.parse_args()
    from datasets import load_dataset

    rows = list(load_dataset(a.dataset, split=a.split))
    base = sum(r["base_solve_rate"] for r in rows) / len(rows)
    print(f"base model (re-pick, k=8): {base:.3f} on {len(rows)} prompts")
    with Renderer() as R:
        for path in a.traces:
            ok = n = 0
            think = []
            per_prompt = []
            for line in open(path):
                t = json.loads(line)
                r = rows[t["index"]]
                b = parse(r["before"])
                solved = 0
                for s in t["samples"]:
                    n += 1
                    think.append(len(s["text"].rsplit("<channel|>", 1)[0].split()))
                    p = strip_fences(s["text"]).strip()
                    if not p or not s.get("finished", True):
                        continue
                    g = grade_checks(r["before"], p, r["checks"], r["allowed"], r.get("design_system"))
                    if not g["pass"]:
                        continue
                    try:
                        if R.new_issues(b, apply_patch(b, p)):
                            continue
                    except Exception:  # noqa: BLE001
                        continue
                    ok += 1
                    solved += 1
                per_prompt.append(solved / max(len(t["samples"]), 1))
            rate = ok / max(n, 1)
            # interval over prompts (answers to one prompt are correlated)
            sd = statistics.pstdev(per_prompt) if len(per_prompt) > 1 else 0.0
            half = 1.96 * sd / math.sqrt(max(len(per_prompt), 1))
            print(f"{path}: solve rate {rate:.3f} ± {half:.3f} ({ok}/{n} answers, {len(per_prompt)} prompts); "
                  f"vs base {rate - base:+.3f}; median thinking words {statistics.median(think) if think else 0}")


if __name__ == "__main__":
    main()
