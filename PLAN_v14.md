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

## Rendered geometry checks (after re-reading Figma's post)
The post combines structural checks with visual judges, and its example failures are overlapping and clipped
elements. Our checks read the DSL, not the rendered page, so they couldn't see those failures.
`design_edit.geometry.Renderer` renders before and after in Chromium (about 20 ms per design). It reports new
clipping, sibling overlap, text overflow and off-page content.
* **References are almost always clean.** New issues come from 4% of eval-v3 references, 4% of hard training
  references, 0% of concrete ones and 8% of Flash ones. RL leaves out the items whose reference introduces an issue.
* **RL reward** (`train_grpo.py --geometry`): an answer that introduces an issue can't pass, and each new issue costs
  −0.05 (up to 4).
* **Eval** (`grade_eval_v2.py --geometry`): reports the share of answers that add issues, and `pass_geo` (passes and
  adds none). On v3, 3–4.5% of answers add issues; pass_geo is 2–4 points below pass (plain Gemma with review
  0.845, Qwen 0.727).
* **A VLM visual judge** (`visual_judge.py`, built in round 7) is an option for eval spot checks. It's too slow and
  costly per RL rollout.

## Visual judge calibration (60 eval-v3 answers, Opus labels from screenshots)
* **Setup:** 30 answers that passed text grading and 30 that failed, from plain Gemma with review, Qwen and v14.
  Each has before and after screenshots, 2x close-ups and a list of every change. Opus labelled each blind to the
  text grade: 43 acceptable, 17 not.
* **Vision judges, agreement with Opus on "acceptable":**

  | judge | agreement | accepts bad (of 17) | rejects good |
  |---|---|---|---|
  | Kimi-K3.1-Pro | 0.763 | 11 | 3 |
  | DeepSeek-V4.1-Flash | 0.754 | 10 | 4 |
  | Mistral-Large-4 | 0.746 | 12 | 3 |
  | Muse-Spark-Open | 0.733 | 10 | 6 |
  | Llama-4-Maverick | 0.559 | 5 | 21 |

* **The vision judges catch few of the bad edits** (6–7 of 17). The bad edits Opus found are mostly related text left
  stale (a counter or label the edit made wrong), unrequested changes and misplaced elements. Screenshots don't
  make those obvious to a VLM. Clipping and overlap are already covered by the geometry check.
* **The text grading agrees with Opus 72% of the time,** about as well as the best vision judge.
  * Of 30 text passes, Opus rejects only 2.
  * Of 30 text failures, Opus accepts 15. Many of those 15 fail on details a screenshot hides (an exact price, a
    label, #71717A against a near-identical grey); in those cases the text grading is right to be strict.
  * The rest are strict-but-debatable criteria: renumbering questions after a reorder, and "nothing unrelated
    changes".
* **Decision:**
  * Add a vision judge (Kimi-K3.1-Pro or DeepSeek-V4.1-Flash) as an extra eval column, `--visual`.
  * Keep it out of the RL reward: at about 75% agreement it adds little beyond the exact checks and the geometry.
  * The visual lever worth testing is the model seeing its own screenshot in the review round (running).

## Visual review round (Gemma sees a screenshot of its edit): worse, because it over-edits
| plain Gemma, eval v3 | all | dev | test | goal | multi_step | pattern_copy | propagate | structure | system_audit |
|---|---|---|---|---|---|---|---|---|---|
| review round, text only | **0.874** | 0.897 | 0.851 | 0.971 | 0.794 | 0.800 | 0.865 | 0.895 | 0.867 |
| review round + screenshot and close-up | 0.806 | 0.793 | 0.828 | 0.857 | 0.824 | 0.867 | 0.892 | 0.667 | 0.733 |

* **Paired:** −6.3 ± 5.6 (+7/−18).
* **It helps a little on pattern_copy, multi_step and propagate, but loses on structure (−23) and goal.** Of the 18
  losses, most fail "Nothing unrelated to the request changes". Seeing the rendered page, the model "polishes" what it
  notices, for example shrinking a title to improve "vertical balance" or snapping nearby values. The screenshot
  invites over-editing.
* **Retry with a stricter prompt** (judge the screenshot only for problems the patch caused; leave pre-existing details
  alone):
  * Overall 0.845 against 0.874 text-only: −2.9 ± 5.1 (+8/−13), dev −6.9, test +1.1.
  * "Nothing unrelated" failures fell from 18 to 7 (text-only: 8), so the over-editing is fixed.
  * It's a trade-off: multi_step 0.912 against 0.794, but system_audit 0.600 against 0.867 (exact values aren't
    visible) and structure 0.763 against 0.895.
* **Decision:** the review round stays text-only. Revisit the screenshot after RL, or in two-turn RL where the model
  can learn when to trust it.

## Two more visual tests at inference (plain Gemma, eval v3, text review round as baseline 0.874)
* **Screenshot of the original design in the first turn:** 0.856 (−1.7 ± 5.6). Multi_step 0.912 (+12), goal −11,
  audit −13.
* **Completeness check after the review** (a screenshot turn that may only add lines): 0.862 (−1.1 ± 2.8). It
  changed only 3 of 176 answers, so it's effectively inert.
* **The pattern across all four visual tests:** the image reliably helps multi-part requests (+12 each time) and hurts
  goal and audit requests. Net, nothing beats the text-only review. Vision stays in the geometry reward and the eval
  column. Revisit it in two-turn RL, where the model can learn when the picture helps.

## Phase 3, stage 1: prompt re-pick on plain Gemma (result)
* **Pool:** 1,399 prompts (802 hard, 597 concrete). 39 were dropped because their reference breaks the layout.
* **Sampling:** 8 single-turn samples each at temperature 1.0, 4,096 tokens. Scored by exact checks plus rendered
  geometry, as the RL reward does.
* **Solve rates:**
  * hard: 414 always, 314 sometimes, 74 never;
  * concrete: 488 always, 62 sometimes, 47 never.
  * 376 sometimes-solved, well above the 150 gate. Median thinking 164 words.
* **Held out:** 51 sometimes-solved items that need a judge check (the reward ignores judge checks).
* **Dataset `mszarski/design-edits-v14-rl`:**
  * train: 225 prompts (182 hard, 43 concrete), base solve rate 0.643;
  * heldout: 100 prompts, base solve rate 0.615. That's 400 graded answers per checkpoint at 4 samples each, so
    about ±5 points.

## Phase 3, stage 2: RL pilot (plain Gemma, 25 steps) — gates not met
* **Run:** 225 prompts; 4 prompts × 8 rollouts per step; reward 0.7·pass + 0.3·score with the design-system lint and
  4px geometry; cap 4,096 with truncated completions masked; KL 0.01; about 9.5 min/step on 2×H200.
* **Training curves, steps 1–5 → 21–25:**
  * reward 0.599 → 0.731;
  * completion length 1,208 → 1,636 tokens (thinking grew, it didn't shrink);
  * clipped 2.5% → 3.7%;
  * zero-std groups 20% throughout;
  * KL 0.038.
* **Go/no-go at checkpoint 25:**
  * **held-out:** 0.642 → 0.670, paired +2.8 ± 5.2 (30 prompts better, 22 worse). Gate +4: **not met**;
  * **thinking:** 370 → 389 words. Gate: met;
  * **eval v3 + review:** 0.850 against 0.874 (−2.9 ± 5.2); dev 0.837 against 0.907 (−7.0 ± 6.3); test +1.1.
    Gate dev ≥ 0.897: **not met**;
  * **by kind:** propagate, multi_step and pattern_copy up a little; system_audit 0.867 → 0.667 and goal
    0.971 → 0.886 down.
* **Training reward rose about 13 points but held-out only about 3:** RL is fitting the 225 training prompts more than
  it is gaining general skill. The pilot was cancelled at about step 30 (no step-50 checkpoint could be saved before
  the timeout).

## Why design-system audits keep failing (Opus analysis of the 15 eval-v3 audit items)
* **Pass counts:** Gemma + review 13/15, plain Gemma 13/15, v14 SFT 11/15, RL pilot 10/15, Qwen 10/15.
* **The 18 failures:**
  * missed deviation: 6;
  * changed an on-system value: 4–5;
  * wrong token: 0;
  * scope error: 0;
  * item at fault: about 1–3;
  * empty or cut-off answer: 6 (Qwen 4, plain Gemma 2; the review round fixes Gemma's).
* **The pilot regressed on 3 items and improved on none.** All three are systematic:
  * it checks spacing against a generic grid instead of the item's scale (pad 12 or 20 where the scale has neither);
  * it trusts the type scale over the component spec (a primary label at 13/500 should be weight 600);
  * it adds weights to every text to match a size/weight pair, which also flattens states. The SFT model makes the
    identical over-edit, so v14 training data may teach it.
* **Reward blind spot:** `rewards.lint` flags only odd spacing, so off-scale 12 or 20 never costs anything.
* **Fixes for the next round:**
  * lint reads the item's spacing scale and radius;
  * a per-property penalty for changes no check covers;
  * contrastive audit training items (component spec over type scale; never add props a node doesn't have; same-role
    siblings must match; scales that leave out 12 or 20);
  * eval fixes: item 70 (the stepper fill is acceptable; secondary labels keep their style), item 8 (the
    primary/secondary swap is acceptable, not required), and a written convention on whether the type scale is
    size/weight pairs or sizes only.

## Phase 2 result: SFT on plain Gemma's own review episodes does not add capability

| | eval v3 single pass | eval v3 + review round | eval v2.2 | thinking (median words) |
|---|---|---|---|---|
| plain Gemma, thinking | **0.747** | **0.874** (dev 0.897) | 0.861 | 350 single / 513 review |
| **v14 SFT** (3,396 rows; LoRA from plain Gemma) | 0.686 | 0.834 (dev 0.828) | 0.864 | 276 single / 487 review |
| v13 (old pipeline) | 0.644 | – | 0.957 | 154 |

* **Paired against plain Gemma:** single pass −6.3 ± 6.4 (+11/−22); with a review round −4.0 ± 5.8 (+10/−17).
  On v2.2 it is level (0.864 against 0.861). Better than v13 on v3 (+4.0 ± 7.4), but not significantly.
* **Thinking still shrank by 21% in a single pass,** although traces were the median length. Most rows came from
  easy concrete items, where Gemma thinks about 60 words.
* **Goal requests lose most** (0.971 → 0.829 with review); structure gains in a single pass (0.763 → 0.821).
* **Conclusion:** fine-tuning on the model's own correct traces does not improve on the model. Here it costs a
  little, as it did in round 12 (+1 ± 3 there). The gains in this project came from RL (+8 points in round 12) and
  from the review round at inference (+12.6). Next: RL directly from plain Gemma, skipping SFT.

## Flash batch: cheap DeepSeek items vetted by check-vs-judge disagreement (result)
* **V4.1-Flash with thinking:** 140 screens, 579 items through the gate (about 83%). Plain Gemma sampled 4 review
  episodes per item; each final answer was graded by the exact checks and by the LLM judge on the criteria.
* **Flagged:** 112 items (19%), where the checks and the judge disagreed.
  * 228 answers failed the checks but passed the judge; 60 passed the checks but failed the judge.
  * Opus review of the flagged items: the judge was right in 86 items (mostly checks too narrow, 13 too loose),
    the checks were right in 23. Result: 109 approved (86 after fixes), 3 rejected.
* **Unflagged:** an Opus spot-check of 30 gave 24 good, 6 fixable and none to reject. So the filter works.
* **The most common defect** was a label check that reads only `text=` on screens that keep labels in the layer
  name. `tool.text_or_name` repairs it automatically, and the gate now rejects checks that name ids not in the
  design.
* **Result:** 522 usable Flash items (413 unflagged and auto-approved after the repair, 109 reviewed). 49
  never-solved, unflagged items are set aside, since RL gets no signal from them.
* **Cost:** about $5 of API calls, 3 GPU-hours of sampling, and Opus review of about 140 items (about 0.9M tokens).
  That's roughly 4× cheaper per usable item than Opus authoring, at similar quality.

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
