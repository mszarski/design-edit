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

## Phase 0 result: every stage cost a little, in step with shorter thinking

| eval v3 (176 items) | pass | goal | multi_step | pattern_copy | propagate | structure | system_audit | thinking words (median) |
|---|---|---|---|---|---|---|---|---|
| plain Gemma, thinking | **0.747** | 0.857 | 0.647 | 0.667 | 0.703 | 0.763 | 0.867 | 350 |
| v12 reasoning SFT | 0.690 | 0.857 | 0.618 | 0.533 | 0.622 | 0.763 | 0.600 | 282 |
| + RL v12 checkpoint-150 | 0.667 | 0.829 | 0.471 | 0.533 | 0.649 | 0.763 | 0.667 | 206 |
| + RL v12b + v13 hint SFT (v13) | 0.644 | 0.829 | 0.618 | 0.533 | 0.568 | 0.658 | 0.533 | 154 |

Paired, step by step:
- SFT against plain Gemma: −5.8 ± 6.7 points;
- RL v12 against the SFT: −1.7 ± 5.4;
- v13 against RL v12: −2.3 ± 5.7.

No single step is significant, but every step moves the same way. Over the whole chain the loss is −10.3 ± 7.5,
which is significant. Median thinking falls at each stage, 350 → 282 → 206 → 154 words. The biggest drops are on
system_audit and propagate, which need a full sweep of the screen.

**Verdict.** There is no single culprit, so phase 2 can't be skipped.
- The SFT already cost about 6 points and 20% of the thinking. Shortest-trace selection and v2-only data are the
  likely cause.
- RL kept shortening the thinking.
- Run phases 2 and 3 as planned, with these additions:
  - phase 2 must keep median thinking at least 80% of plain Gemma's (≥ 280 words) on v3 dev;
  - phase 3 stops if thinking falls by more than 25%;
  - phase 3 runs both v3 dev and v2.2 dev at every checkpoint.

## Encouraging thinking (applies to phases 2-3)
Thinking shrank because four parts of the pipeline pushed against it. Remove those pressures; don't reward length
directly, because models learn to pad.
1. **SFT trace selection:** keep the median-length correct trace, not the shortest (`filter_traces.py --select`).
   Prefer traces that show the procedure: rule → every affected instance → patch → check.
2. **Tasks where thinking pays:** hard multi-part, propagating and audit items (phase 1). On v2-style local edits,
   RL loses nothing by thinking less.
3. **No truncation penalty in RL:**
   - completion cap of at least 4,096 tokens;
   - truncated completions masked out of the loss (`train_grpo.py --mask-truncated`);
   - the loss is already token-level (`dapo`).
4. **KL to the start model:** `--beta 0.01`–`0.02`. It was 0, so nothing held behaviour near the base.
5. **Inference tests** on plain Gemma before any training (running):
   - a checklist added to the system prompt (`eval_model.py --system-suffix checklist`);
   - one review round (`--revise-rounds 1`).

**Phase 1 writer and reviewer** (decided): DeepSeek-V4-Pro writes the screens and items
(`scripts/write_hard_items.py`), and Opus agents review them blind, approving, fixing or rejecting each.

## Inference tests on plain Gemma 4 31B, eval v3 (result)

| plain Gemma, thinking | all | dev | goal | multi_step | pattern_copy | propagate | structure | system_audit | thinking words |
|---|---|---|---|---|---|---|---|---|---|
| normal prompt | 0.747 | 0.747 | 0.857 | 0.647 | 0.667 | 0.703 | 0.763 | 0.867 | 350 |
| + checklist in the system prompt | 0.747 | 0.736 | 0.714 | 0.765 | 0.733 | 0.757 | 0.816 | 0.600 | 456 |
| **+ one review round** | **0.874** | **0.897** | 0.971 | 0.794 | 0.800 | 0.865 | 0.895 | 0.867 | 513 |

* **The checklist does nothing overall** (+17/−17). It helps sweep-type kinds and hurts goal and audit requests.
  More thinking alone doesn't help.
* **One review round adds +12.6 ± 6.1 points** (+27/−5; dev +14.9 ± 8.7). The model sees its own patch applied:
  the list of changes and the resulting design. Nothing from the grader is shown. It then keeps or corrects the
  patch. Plain Gemma with a review round (0.874) is well above plain Qwen in one pass (0.767). Qwen with a review
  round hasn't been measured.
* **What this changes:**
  * **Phase 2:** generate traces with the review loop. More hard prompts get solved, so more get a correct trace.
    Train on the full two-turn exchange, so the model learns to review, or on the corrected final answer.
  * **Phase 3:** include the review turn in RL, rewarding the final patch.
  * **Ship the review round at inference** regardless; it costs one extra model call.
 hard training data (about $30–60 of API calls, plus about 1M agent tokens; 1 day)
## Phase 1 pilot: DeepSeek-V4-Pro as writer (result)
* **Writer mechanics:**
  * V4-Pro on the router reasons until the token limit on these long prompts and returns nothing, even at
    `reasoning_effort=low`, so it runs with reasoning off.
  * Indented subtree inserts are converted into canonical `insert` lines.
  * The repair round gets the screen and the patch rules.
* **Yield:** 60 screens: 56 valid and 268 items written. 80 passed the gate (30%).
* **Opus review of 48 gated items** (blind): 1 approved as-is, 17 approved after fixes, 30 rejected (62%). So about
  7% of written items survive. The failures, in order:
  1. false premises and off-system "fixes";
  2. checks that pin one answer (ids of new nodes, `text=` on name-only labels, exact layer names, one value
     where several are on-scale) or that silently do nothing (unknown keys);
  3. trivial items labelled hard, or vague "mood" goals whose checks then pick one answer.
* **Gate change:** `checks.check_problems` rejects checks the grader would ignore: unknown keys, node checks with
  nothing to test, unknown `new_node` keys.
* **Decision (with the user): C + B.** A smaller hard set (about 300 items) written by Opus agents and reviewed
  blind, plus the existing realistic set re-checked under the v2.2 conventions. Opus writes the inputs and the
  grader; Gemma is trained only on its own traces.

## DeepSeek writer, second try: thinking on, the model card's settings (result)
* **The model card** recommends temperature 1.0 and top_p 1.0; the modes are Non-think, High and Max. The router
  setup ran out of tokens because 2–5 items per call didn't fit in 32k with reasoning. With one item per call and
  64k tokens, V4-Pro reasons 20–32k tokens (3–4 min) and answers. DeepInfra silently ignores reasoning.
* **Gate yield on 15 screens:**
  * V4-Pro with thinking: 0/15;
  * V4-Pro without thinking: 0/15 (checks too loose to catch their own traps);
  * **V4.1-Flash with thinking: 11/15.**
* **Opus review of Flash's 11 items:** 3 good as-is, 6 fixable, 2 rejected.
  * Premises are much better than before.
  * But checks reject valid answers in 5 items and accept wrong ones in 2, and many items are easy.
  * These flaws only show when other valid answers are tried, so the writer would need a full review, not a sample.
* **Decision:** Opus authors the hard items (B). DeepSeek is used for conversion (1C) and judging, not for authoring.
* **Screen gate fix:** numeric properties must be numbers. The fix rejects `size=w:80,h:20`-style screens that
  DeepSeek wrote and that the old gate let through.

## Phase 1C pilot: re-checking the realistic training set (result)
* DeepSeek re-grading under the v2.2 conventions: 25 of 40 items pass the validator (62%). It dropped none.
* **Blind Opus audit of those 25:**
  * 13 good, 4 with a minor loophole, 7 too loose, 1 should be dropped.
  * The split follows the request type. Concrete requests (an exact value, a token, disable, select) are fine.
    **Vague requests ("stand out", "more subtle", "improve hierarchy") are too loose**: the checks test "changed",
    not the direction or every instance. A policy could earn reward for opposite or one-instance changes.
* **Use:** keep the concrete kinds (direct, state, consistency, structure) from this set. Take goal-type training
  items only from the Opus-written hard set.
* **Grader limits that matter for the RL reward** (fix in phase 3):
  1. Judge checks count toward neither pass nor fail. Items whose key criterion is a judge check need a judge call
     in the reward, or must be excluded.
  2. Props a check doesn't name can change freely on "allowed" nodes, and new nodes are always allowed. Add a small
     penalty per changed property that no check covers, sized so complete multi-part edits aren't punished.
  3. Disable items should also require no opacity on the label, so opacity isn't stacked on the button and its label.

## Phase 1: hard training data (original plan; see the pilot results above)
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
