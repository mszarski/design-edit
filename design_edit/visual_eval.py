"""Visual grading for evals: screenshots of before/after (+ 2x close-ups of the edited area) graded by a VLM.

Mirrors Figma's evaluation ("we combine checks on the design's structure with visual judges"): the judge sees the
rendered result, a plain-language list of every change (some changes are invisible in a screenshot), and grades
fulfilled / preserved / legible / contained / minimal separately (design_edit.visual_judge).
"""

from __future__ import annotations

from .goal_reward import describe_changes, focus_node
from .schema import Node, apply_patch, diff, format_patch, parse, strip_fences


def visual_case(renderer, before_text: str, output: str) -> dict | None:
    """Screenshots and judge inputs for one answer, or None if the patch doesn't apply or changes nothing."""
    before = parse(before_text)
    try:
        after = apply_patch(before, strip_fences(output).strip())
    except Exception:  # noqa: BLE001
        return None
    canon = format_patch(diff(before, after)) if after["id"] == before["id"] else ""
    if not canon:
        return None
    fid = focus_node(before, after)
    zb, za = (renderer.crop(before, fid), renderer.crop(after, fid)) if fid else (None, None)
    return {"before_png": renderer.screenshot(before), "after_png": renderer.screenshot(after),
            "closeups": (zb, za) if zb and za else None,
            "patch_for_judge": canon + "\n\nEvery change this patch makes, including ones the screenshots cannot show:\n"
                               + describe_changes(before, after)}


def visual_grade(renderer, before_text: str, output: str, request: str, model: str) -> dict | None:
    """Per-dimension verdict dict (see visual_judge.DIMENSIONS) or None (invalid patch / judge unreachable)."""
    from .visual_judge import judge

    c = visual_case(renderer, before_text, output)
    if c is None:
        return None
    return judge(c["before_png"], c["after_png"], request, c["patch_for_judge"], model, closeups=c["closeups"])
