# Plan v14: retrain without compressing reasoning, measured on eval v3

## Goal and success criteria
On **eval v3 test** (88 items, never used for selection):
- the trained model beats plain Gemma 4 31B with thinking (0.747) and plain Qwen 3.8 27B (0.767), paired;
- with no regression on **eval v2.2 test**: ≥ 0.93 (plain Gemma thinking 0.865, v13 0.955).

Checkpoints are chosen on the v3 dev and v2.2 dev splits only.

What went wrong in v12–v13: the SFT kept the shortest correct trace; RL trained only on v2-style local edits on small
screens with a 1,024–2,048-token cap; and `rewards.lint` penalized design-system-correct values. The model learned
to think less (155 words against 350 for plain Gemma) and to do part of multi-part edits.

## Phase 0: find where the drop happens (about $20, half a day)
Run eval v3 on the intermediate models to see which stage caused the drop:
- **v12 reasoning SFT** (before any RL);
- **RL v12 checkpoint-150**.

Read the results this way:
- **The SFT alone drops below plain Gemma:** trace selection and data are the main cause, so phases 2–3 are needed
  as planned.
- **The SFT holds and RL causes the drop:** the SFT recipe is fine. Re-run RL only (phase 3) from v12 SFT with the
  new prompts and token cap, and skip most of phase 2.

At the same time, make these code fixes. They need no GPU.
- `rewards.lint`: take the type scale and colors from each item's design system, and treat a rect's fill as a
  background.
- `filter_traces.py --select median|longest|shortest`, defaulting to median.
- A position check in `design_edit/checks.py`: `{"new_node": {...}, "after": "<id>"}` or `"before": "<id>"`, meaning
  directly next to an existing sibling. The v3 reviewers asked for it; it makes structure and pattern_copy items
  exactly gradable.
- Log thinking-length and patch-length statistics in `eval_model.py`'s summary.

## Phase 1: hard training data (about $30–60 of API calls, plus about 1M agent tokens; 1 day)
- **Writer:** a script, not agents. It reuses `eval_v3/SPEC.md`, the conventions and `eval_v3/tool.py`'s gate, with
  DeepSeek-V4-Pro on the HF router as the writer.
  - The gate requires: the reference passes; an alternative passes; doing nothing fails; a trap fails the checks.
  - Training on Opus outputs would need your OK under Anthropic's terms, so Opus stays in the review role.
- **Volume:** about 300 new screens (seeds 60000+), 45–110 nodes each, with about 5 items per screen, so about
  1,500 items. Kinds match v3: goal, structure, multi_step, propagate, system_audit, pattern_copy.
- **Held out from v3:** use a disjoint list of screen types, so v3 keeps measuring generalization to new screens.
- **Quality audit:** one Opus agent reviews a random 100 items blind, as the v3 reviewers did.
  - If the defect rate is above 15%: fix the writer prompt and regenerate.
  - If the writer can't get there: ask you about using Opus as the writer.
- **Also:** re-compile checks for the existing realistic training set (`train_real`) under the v2.2 conventions. It
  currently has stale checks and AND-ed alternatives.

## Phase 2: SFT from plain Gemma (about $50–70; 1 day)
- **Start from plain Gemma**, not on top of v13. The v12–v13 stack has the compressed behavior built in.
- **Rejection sampling:** plain Gemma with thinking on, 4 samples per prompt, 4,096-token cap.
  - Prompts: the ~1,500 hard items, plus about 3,000 existing v2-style items for coverage.
  - Graded by the fresh checks, so alternatives count.
- **Trace selection: the median-length correct trace, not the shortest.**
  - Expected yield is about 70% of prompts.
  - Prompts with no correct sample go to RL as "never solved"; they are not rationalized.
- **Training:** LoRA, 1 epoch, max length 8,192 (larger screens), thinking on, inference-matched rendering.
- **Gate:**
  - v3 dev ≥ plain Gemma (0.747) and v2.2 dev ≥ 0.90;
  - median thinking length on v3 dev ≥ 80% of plain Gemma's.

  If it fails, stop and diagnose before RL.

## Phase 3: RL (about $120–150; 1–1.5 days)
- **Prompts:** re-pick on the SFT model, 8 samples each. Keep the prompts it solves 1–7 of 8 times, from both the
  hard set and the v2 set, about 60% hard.
- **Reward:**
  - check pass plus keep-out, with the fixed lint;
  - no edit-size penalty; minimality comes from keep-out checks, so complete multi-part edits aren't punished;
  - the LLM judge only on items with judge checks.
- **Budget:** 4,096-token completions, 2×H200, checkpoints every 50 steps, about 200 steps.
- **Monitor per checkpoint:**
  - reward, zero-variance groups and clipped ratio;
  - thinking length (must not fall steadily);
  - v3 dev and v2.2 dev.
- **Stop when:** v3 dev is flat for 2 checkpoints, or thinking length drops by more than 25% from the SFT start.

## Phase 4: report (an hour)
- Score v3 test and v2.2 test for the chosen checkpoint, plain Gemma, Qwen and v13.
- Paired comparisons, plus length statistics per round.
- Update the README and RECIPE.md.

## Cost and time
| Phase | GPU | API / agents | Wall clock |
|---|---|---|---|
| 0: diagnose + code fixes | ~$20 | – | 0.5 day |
| 1: hard training data | – | ~$30–60 + ~1M agent tokens | 1 day |
| 2: SFT | ~$50–70 | – | 1 day |
| 3: RL | ~$120–150 | judge calls ~$5 | 1–1.5 days |
| **Total** | **~$190–240** | **~$35–65** | **~3.5–4 days** |

Each phase has a gate. Phase 0's answer may cut phase 2 short, which would save about $50 and a day.

## Risks
- **Writer quality:** DeepSeek-written checks may be noisier than Opus-written ones, and noisy checks mean a noisy
  RL reward. The phase-1 audit and the gate (traps, alternatives) are there to catch this.
- **Still behind plain Qwen:** possible, because Qwen thinks about 1,000 tokens per request. If phase 3 leads plain
  Gemma but not Qwen, a longer budget (8k) and more hard data are the next levers.
- **v3 has 88 test items,** so ±7–8 points is noise on a paired comparison. To confirm a close win, author about
  100 more v3 items with the same workflow (about 1M agent tokens) before claiming it.
