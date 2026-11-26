#!/usr/bin/env python3
"""Phase 1 of plan v14: hard *training* items written by an LLM (default DeepSeek-V4-Pro), gated like eval v3.

For each screen: the writer designs a large screen (eval_v3/SPEC.md) for a design system, then writes items of the
v3 kinds with criteria, exact checks, a reference, alternative answers and traps. Every item must pass the
eval_v3/tool.py gate (reference and alternatives pass, doing nothing fails, a trap is caught by the checks); a failed
item gets one repair round with the gate's errors. Opus agents then review the gated items (see PLAN_v14.md).

Screen types are disjoint from eval v2 and v3, and seeds start at 60000, so eval v3 stays a held-out test.

    python scripts/write_hard_items.py --work data/hard --n-screens 300
Writes <work>/screens/, <work>/proposals/ and <work>/gated.txt (ids that passed the gate). Resumable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_eval_v2 import parse_json  # noqa: E402
from compile_checks import SPEC as CHECK_SPEC  # noqa: E402

from design_edit.design_system import render_design_system, sample_design_system  # noqa: E402
from design_edit.judge import chat  # noqa: E402
from design_edit.prompts import SYSTEM_PROMPT  # noqa: E402
from design_edit.schema import strip_fences  # noqa: E402

TRAIN_SCREENS = [
    "fitness tracker workout log", "travel itinerary planner", "inventory management table", "newsletter editor",
    "patient intake form", "car rental checkout", "video course catalog", "smart home device dashboard",
    "subscription management page", "issue tracker detail with comments", "warehouse shipment list",
    "conference schedule with sessions", "donation page with amounts", "language learning lesson",
    "freelancer invoice list", "property management tenants table", "photo gallery with albums",
    "grocery delivery cart", "ride booking with options", "wedding RSVP form", "library book detail",
    "time tracking timesheet", "vendor onboarding wizard", "crypto portfolio overview", "museum ticket booking",
    "sales leaderboard", "school grades report", "pet adoption profile", "parking permit application",
    "customer feedback inbox", "game lobby with players", "legal document review list", "farm produce marketplace",
    "IT asset request form", "gym class booking calendar", "news reader with sections", "insurance claim form",
    "volunteer shift sign-up", "wine shop product list", "recruiting pipeline board",
]
KINDS = ["goal", "goal", "structure", "structure", "multi_step", "multi_step", "propagate", "propagate",
         "system_audit", "pattern_copy"]

SCREEN_PROMPT = """{system}

That is the system prompt of a model that edits designs written in the design language above. You are writing a
screen for its training data. Requirements (from the eval spec):

{screen_spec}

Screen type: {screen}
Design system to follow:
{ds}

Reply with the design only (the DSL lines), no commentary."""

ITEM_PROMPT = """You write hard training items for a model that edits UI designs. Each item is a natural request on
the screen below, graded by exact checks and criteria. Follow the spec and the grading conventions exactly: hard
must never mean ambiguous, arbitrary or unfair.

=== SPEC ===
{spec}

=== GRADING CONVENTIONS ===
{conventions}

=== CHECK LANGUAGE ===
{check_spec}
  {{"any": [<check>, ...]}}                            passes when one alternative passes

=== PATCH LANGUAGE AND DSL ===
{system}

=== DESIGN SYSTEM ===
{ds}

=== SCREEN ({sid}) ===
{design}

Write {k} items, one of each kind: {kinds}. Reply with a JSON list only. Each item:
{{"kind": "...", "request": "...", "reference_patch": "...", "criteria": ["...", ..., "Nothing unrelated to the request changes."],
  "checks": [...], "allowed": ["n..", ...], "alternatives": ["<another valid patch>"],
  "traps": [{{"patch": "...", "why_wrong": "...", "caught_by": "checks"}}], "notes": "what makes it hard"}}
Rules the automatic gate enforces: the reference and every alternative apply cleanly, stay in the design language,
change something and pass all checks; the unedited design fails at least one check; every trap applies and at least
one trap fails the checks; "allowed" lists every existing node a valid answer may touch."""

REPAIR_PROMPT = """Your item was rejected by the automatic gate:
{errors}

Item:
{item}

Fix the item (usually the checks, "allowed", an alternative or a trap; change the request only if its premise is
false). Reply with the corrected item as one JSON object only."""


def load_tool(work: Path):
    os.chdir(work)
    spec = importlib.util.spec_from_file_location("v3tool", ROOT / "eval_v3" / "tool.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--n-screens", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--items-per-screen", type=int, default=5)
    ap.add_argument("--writer", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument("--seed-base", type=int, default=60_000, help="eval v2 uses 9000+, train_real 100000+, v3 50000+")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    assert os.environ.get("HF_TOKEN")
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    T = load_tool(work)
    spec_text = (ROOT / "eval_v3" / "SPEC.md").read_text()
    conventions = (ROOT / "eval_v3" / "CONVENTIONS.md").read_text()
    screen_spec = spec_text.split("## Screens", 1)[1].split("##", 1)[0].strip()
    lock = threading.Lock()
    gated_path = work / "gated.txt"
    done_path = work / "done_screens.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    stats = {"screens": 0, "screen_fail": 0, "items": 0, "gated": 0, "repaired": 0}

    def ask(prompt: str, max_tokens: int, think: bool = True) -> str:
        # The writer is a reasoning model: reasoning tokens count against max_tokens, so budgets are generous.
        return chat(args.writer, [{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.7,
                    timeout=1800, retries=2, extra=None if think else {"reasoning": {"enabled": False}})

    def one_screen(i: int) -> None:
        sid = f"t{i:04d}"
        if sid in done:
            return
        rng = random.Random(args.seed_base + i)
        seed = args.seed_base + i
        ds = render_design_system(sample_design_system(seed))
        screen = TRAIN_SCREENS[i % len(TRAIN_SCREENS)]
        (work / "screens" / f"{sid}.json").write_text(json.dumps({"screen": screen, "ds_seed": seed}))
        feedback, ok = "", False
        for _ in range(3):
            out = ask(SCREEN_PROMPT.format(system=SYSTEM_PROMPT, screen_spec=screen_spec, screen=screen, ds=ds)
                      + feedback, 12000, think=False)
            (work / "screens" / f"{sid}.dsl").write_text(strip_fences(out).strip() + "\n")
            try:
                _, errs, _ = T.check_screen(sid)
            except Exception as e:  # noqa: BLE001
                errs = [f"parse error: {e}"]
            if not errs:
                ok = True
                break
            feedback = "\n\nA previous attempt was rejected: " + "; ".join(errs) + ". Fix that."
        if not ok:
            with lock:
                stats["screen_fail"] += 1
            return
        design = (work / "screens" / f"{sid}.dsl").read_text()
        kinds = rng.sample(KINDS, args.items_per_screen)
        items = parse_json(ask(ITEM_PROMPT.format(
            spec=spec_text, conventions=conventions, check_spec=CHECK_SPEC, system=SYSTEM_PROMPT, ds=ds, sid=sid,
            design=design, k=len(kinds), kinds=", ".join(kinds)), 32000))
        gated = []
        for k, it in enumerate(items if isinstance(items, list) else []):
            iid = f"{sid}i{k}"
            if not isinstance(it, dict):
                continue
            it["screen_id"] = sid
            path = work / "proposals" / f"{iid}.json"
            path.write_text(json.dumps(it, indent=1))
            try:
                _, errs = T.gate(iid)
            except Exception as e:  # noqa: BLE001
                errs = [f"{type(e).__name__}: {e}"]
            if errs:
                fixed = parse_json(ask(REPAIR_PROMPT.format(errors="\n".join(errs), item=json.dumps(it, indent=1)), 16000))
                if isinstance(fixed, dict):
                    fixed["screen_id"] = sid
                    path.write_text(json.dumps(fixed, indent=1))
                    try:
                        _, errs = T.gate(iid)
                    except Exception as e:  # noqa: BLE001
                        errs = [f"{type(e).__name__}: {e}"]
                    if not errs:
                        with lock:
                            stats["repaired"] += 1
            if not errs:
                gated.append(iid)
        with lock:
            stats["screens"] += 1
            stats["items"] += len(items) if isinstance(items, list) else 0
            stats["gated"] += len(gated)
            with open(gated_path, "a") as f:
                f.write("".join(g + "\n" for g in gated))
            with open(done_path, "a") as f:
                f.write(sid + "\n")
            print(f"{sid} ({screen}): {len(gated)}/{len(items) if isinstance(items, list) else 0} gated  {stats}",
                  flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one_screen, range(args.start, args.start + args.n_screens)))
    print("done", stats)


if __name__ == "__main__":
    main()
