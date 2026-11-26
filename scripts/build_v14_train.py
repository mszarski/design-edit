#!/usr/bin/env python3
"""Assemble the plan-v14 training prompts.

* hard items: approved eval-v3-style items from agent work directories (Opus-written; DeepSeek-written and Opus-fixed),
  re-gated with eval_v3/tool.py so only items that still pass are used;
* concrete items: the realistic set re-checked under the v2.2 conventions (recheck_train_real.py), keep/fix verdicts,
  concrete kinds only (vague goal checks were too loose for a reward), and only items graded fully by exact checks.

    python scripts/build_v14_train.py --hard <work> [<work> ...] --concrete train_real_v22.jsonl --out v14_train.jsonl \
        [--push user/design-edits-v14-train]
"""
import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from design_edit.design_system import render_design_system, sample_design_system, user_message_ds  # noqa: E402
from design_edit.prompts import SYSTEM_PROMPT  # noqa: E402
from design_edit.schema import apply_patch, parse, serialize, strip_fences  # noqa: E402

COLS = ["prompt", "completion", "before", "after", "kind", "instruction", "orig_instruction", "relax", "alts", "refs",
        "criteria", "design_system", "screen", "checks", "allowed", "source", "item_id"]


def hard_rows(work: Path, gated: bool = False) -> list[dict]:
    cwd = os.getcwd()
    os.chdir(work)
    spec = importlib.util.spec_from_file_location("v3tool", ROOT / "eval_v3" / "tool.py")
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    rows = []
    ids = (sorted(set((work / "gated.txt").read_text().split())) if gated
           else [f.stem for f in sorted((work / "approved").glob("*.json"))])
    for f in [work / "proposals" / f"{i}.json" for i in ids]:
        _, errs = T.gate(f.stem)
        if errs:
            continue
        p = json.loads((work / "proposals" / f"{f.stem}.json").read_text())
        meta = json.loads((work / "screens" / f"{p['screen_id']}.json").read_text())
        doc = parse((work / "screens" / f"{p['screen_id']}.dsl").read_text().strip())
        ds = render_design_system(sample_design_system(int(meta["ds_seed"])))
        patch = strip_fences(p["reference_patch"]).strip()
        rows.append({
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": user_message_ds(serialize(doc), ds, p["request"])}],
            "completion": [{"role": "assistant", "content": patch}],
            "before": serialize(doc), "after": serialize(apply_patch(doc, patch)), "kind": f"v3_{p['kind']}",
            "instruction": p["request"], "orig_instruction": p["request"], "relax": "[]", "alts": "[]", "refs": "[]",
            "criteria": json.dumps(p["criteria"]), "design_system": ds, "screen": meta["screen"],
            "checks": json.dumps(p["checks"]), "allowed": json.dumps(p["allowed"]),
            "source": f"hard:{work.name}", "item_id": f.stem})
    os.chdir(cwd)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", nargs="*", default=[])
    ap.add_argument("--hard-gated", nargs="*", default=[],
                    help="work dirs whose gate-passed (not yet reviewed) items to include, from gated.txt")
    ap.add_argument("--concrete", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", default=None)
    a = ap.parse_args()
    rows = []
    for w in a.hard:
        rows += hard_rows(Path(w).resolve())
    for w in a.hard_gated:
        rows += hard_rows(Path(w).resolve(), gated=True)
    n_hard = len(rows)
    for line in open(a.concrete) if a.concrete else []:
        r = json.loads(line)
        if r.get("v22") not in ("keep", "fix") or r["kind"] == "v2_goal" or '"judge"' in r.get("checks", ""):
            continue
        rows.append({**{k: r.get(k, "") for k in COLS}, "source": "concrete:v11-real", "item_id": f"real{r['_i']}"})
    rows = [{k: r.get(k, "") for k in COLS} for r in rows]
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{n_hard} hard + {len(rows) - n_hard} concrete = {len(rows)} -> {a.out}")
    print(Counter(r["kind"] for r in rows))
    if a.push:
        from datasets import Dataset
        Dataset.from_list(rows).push_to_hub(a.push, split="train", private=True)
        print("pushed", a.push)


if __name__ == "__main__":
    main()
