#!/usr/bin/env python3
"""Plan v14 phase 1C: re-check the realistic training set (design-edits-v11-real) under the eval-v2.2 conventions.

Its criteria came from the eval-v2 writer (alternatives AND-ed, ids of new nodes, exact values for vague requests)
and its compiled checks predate the conventions. For each item the writer (default DeepSeek-V4-Pro, reasoning off)
returns keep / fix / drop with fresh criteria, checks, "allowed" and alternative answers. An item is kept only if:
the checks have no problems the grader would ignore (checks.check_problems), the reference and every alternative
pass, and the unedited design fails. One repair round with the errors. Rows are written with "checks"/"allowed"
filled, so filter_traces.py and train_grpo.py --reward checks use them directly.

    python scripts/recheck_train_real.py --out data/train_real_v22.jsonl [--n 5383] [--push user/design-edits-v14-real]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_eval_v2 import parse_json  # noqa: E402
from compile_checks import SPEC as CHECK_SPEC  # noqa: E402

from design_edit.checks import check_problems, eval_check, grade_checks  # noqa: E402
from design_edit.judge import chat  # noqa: E402
from design_edit.schema import apply_patch, parse, serialize, strip_fences  # noqa: E402

PROMPT = """You maintain a training set for a model that edits UI designs. Re-grade one item under these conventions.

=== GRADING CONVENTIONS ===
{conventions}

=== CHECK LANGUAGE (only these keys; anything else is ignored by the grader and rejected) ===
{check_spec}

Design system:
{ds}

Design before:
{before}

Request: {request}

Current criteria (may violate the conventions):
{criteria}

Reference answer (patch):
{patch}

Decide "keep" (criteria fair), "fix" (rewrite fair criteria) or "drop" (false premise, ambiguous request, or a
reference a designer would not accept). For keep/fix give criteria (last: "Nothing unrelated to the request
changes."), exact checks, "allowed" (every existing node a valid answer may touch) and 1-2 alternative valid
patches that differ from the reference. Never use ids of nodes the edit creates in checks (use new_node, with
"after"/"before" for position); accept text= or name= for labels stored only in the layer name; use "in"/ranges
where several values are acceptable. Reply with JSON only:
{{"verdict": "keep"|"fix"|"drop", "reason": "...", "criteria": [...], "checks": [...], "allowed": [...], "alternatives": [...]}}"""

REPAIR = """Your answer was rejected by the automatic validator:
{errors}

Your answer:
{answer}

Fix it (usually the checks, "allowed" or an alternative; or change the verdict to "drop" if the item can't be graded
fairly). Reply with the corrected JSON object only."""


def ref_patch(row: dict) -> str:
    c = row["completion"]
    return c[0]["content"] if isinstance(c, list) else c


def validate(row: dict, ans: dict) -> list[str]:
    if ans.get("verdict") == "drop":
        return []
    errs = []
    checks, allowed = ans.get("checks"), ans.get("allowed")
    if not isinstance(checks, list) or not checks or not isinstance(allowed, list):
        return ["checks (non-empty list) and allowed (list) are required"]
    for k, c in enumerate(checks):
        errs += check_problems(c, f"check {k}")
    before = parse(row["before"])
    for name, patch in [("reference", ref_patch(row))] + [(f"alternative {k}", a) for k, a in
                                                            enumerate(ans.get("alternatives") or [])]:
        if not isinstance(patch, str):
            errs.append(f"{name} is not a string")
            continue
        try:
            after = apply_patch(before, strip_fences(patch))
        except Exception as e:  # noqa: BLE001
            errs.append(f"{name} does not apply: {e}")
            continue
        if serialize(after) == serialize(before):
            errs.append(f"{name} changes nothing")
            continue
        g = grade_checks(row["before"], strip_fences(patch), checks, allowed, row.get("design_system"))
        if not g["pass"]:
            errs.append(f"{name} fails checks: {g['why']}")
    if not any(eval_check(c, before, before) is False for c in checks):
        errs.append("the unedited design passes every check")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mszarski/design-edits-v11-real")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--writer", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--push", default=None)
    args = ap.parse_args()
    assert os.environ.get("HF_TOKEN")
    from datasets import load_dataset

    rows = list(load_dataset(args.dataset, split="train"))
    if args.n:
        rows = rows[:args.n]
    conventions = (ROOT / "eval_v3" / "CONVENTIONS.md").read_text()
    out_path = Path(args.out)
    done = {json.loads(line)["_i"] for line in open(out_path)} if out_path.exists() else set()
    lock = threading.Lock()
    stats = {"keep": 0, "fix": 0, "drop": 0, "invalid": 0, "repaired": 0}

    def ask(msgs: list[dict]) -> dict | None:
        out = parse_json(chat(args.writer, msgs, max_tokens=12000, temperature=0.3, timeout=900, retries=2,
                              extra={"reasoning": {"enabled": False}}))
        return out if isinstance(out, dict) else None

    def one(i: int) -> None:
        if i in done:
            return
        r = rows[i]
        crit = json.loads(r["criteria"]) if r.get("criteria") else []
        msgs = [{"role": "user", "content": PROMPT.format(
            conventions=conventions, check_spec=CHECK_SPEC, ds=r["design_system"], before=r["before"],
            request=r["instruction"], criteria="\n".join(f"- {c}" for c in crit), patch=ref_patch(r))}]
        ans = ask(msgs)
        errs = validate(r, ans) if ans else ["no JSON"]
        repaired = False
        if errs and ans:
            fixed = ask(msgs + [{"role": "assistant", "content": json.dumps(ans)},
                                {"role": "user", "content": REPAIR.format(errors="\n".join(errs), answer=json.dumps(ans))}])
            if fixed and not validate(r, fixed):
                ans, errs, repaired = fixed, [], True
        verdict = "invalid" if errs or not ans else ans.get("verdict", "invalid")
        rec = {**r, "_i": i, "v22": verdict, "v22_reason": (ans or {}).get("reason", "; ".join(errs))[:300]}
        if verdict in ("keep", "fix"):
            rec |= {"criteria_v2": r.get("criteria"), "criteria": json.dumps(ans["criteria"]),
                    "checks": json.dumps(ans["checks"]), "allowed": json.dumps(ans["allowed"]),
                    "alternatives": json.dumps(ans.get("alternatives") or [])}
        with lock:
            stats[verdict if verdict in stats else "invalid"] += 1
            stats["repaired"] += repaired
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n = sum(v for k, v in stats.items() if k != "repaired")
            if n % 100 == 0:
                print(n, stats, flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, range(len(rows))))
    print("done", stats)
    if args.push:
        from datasets import Dataset

        kept = [json.loads(line) for line in open(out_path)]
        kept = [r for r in kept if r["v22"] in ("keep", "fix")]
        Dataset.from_list(kept).push_to_hub(args.push, split="train", private=True)
        print("pushed", len(kept), "->", args.push)


if __name__ == "__main__":
    main()
