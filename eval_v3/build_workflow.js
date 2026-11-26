export const meta = {
  name: 'eval-v3-build',
  description: 'Opus authors write large screens and hard eval items; a blind Opus reviewer per batch accepts, fixes or rejects each',
  phases: [
    { title: 'Author', detail: '5 authors, 8 screens and ~36 hard items each', model: 'opus' },
    { title: 'Review', detail: 'one adversarial reviewer per author batch', model: 'opus' },
  ],
}
const W = args.dir, T = args.tool, D = args.docs
const SCREENS = [
  ['calendar week view with events', 'shopping cart with recommendations', 'admin user detail with activity log', 'flight search results with filters', 'analytics dashboard with stat cards and a table', 'multi-step checkout with sidebar summary', 'team settings with roles table', 'mail inbox with message list and preview'],
  ['course lesson page with outline', 'invoice editor with line items', 'CRM deal detail with timeline', 'pricing plans with feature comparison', 'project overview with tabs and task list', 'notification preferences with toggles', 'file manager with folders and details panel', 'support ticket thread with reply box'],
  ['recipe detail with ingredients and steps', 'music player with queue', 'job application form with sections', 'kanban board with three columns', 'account security settings', 'product detail page with reviews', 'onboarding checklist', 'billing page with plans and invoices'],
  ['hotel booking detail with rooms', 'social feed with composer and posts', 'expense report form', 'API keys management table', 'event registration form with ticket types', 'chat app with conversation list', 'search results with facets', 'order tracking with steps'],
  ['restaurant menu with categories', 'healthcare appointment booking', 'learning dashboard with progress cards', 'real estate listing detail', 'survey builder with questions', 'podcast episode page', 'banking transactions list with filters', 'workspace members invite modal'],
]
const MIX = 'about 8 goal, 8 structure, 7 multi_step, 7 propagate, 3 system_audit, 3 pattern_copy'
const SUMMARY = { type: 'object', properties: { items: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' } }, required: ['items', 'notes'] }
const REVIEW = { type: 'object', properties: { approved: { type: 'array', items: { type: 'string' } }, rejected: { type: 'array', items: { type: 'string' } }, fixed: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' } }, required: ['approved', 'rejected', 'fixed', 'notes'] }
const base = `Work directory: ${W} (cd there; run the tool as: python3 ${T} <cmd> ...). Read ${D}/SPEC.md and ${D}/CONVENTIONS.md first, then the tool's docstring (${T}), the check language in /home/user/claude_playground/design_edit/scripts/compile_checks.py (SPEC string; also {"any": [...]}), and design_edit/schema.py for the DSL and patch language. Do not read any model outputs, training data or other eval sets; do not modify the repo.`

const run = await pipeline(SCREENS.map((s, k) => ({ k, s })),
  ({ k, s }) => agent(`${base}

You are author ${k}. You write part of a new, hard evaluation set for a model that edits UI designs.
1. Write 8 screens, one per screen type below, ids a${k}s0..a${k}s7: screens/<sid>.dsl (the design) and screens/<sid>.json {"screen": "<type>", "ds_seed": ${50000 + 100 * k} + j}. Get each design system with "tool.py ds <seed>" and follow it. Validate with "tool.py screen <sid>" until OK.
Screen types: ${s.map((x, j) => `${j}: ${x}`).join('; ')}.
2. Write about 36 items, 4-5 per screen, ids a${k}i00, a${k}i01, ...: proposals/<iid>.json, then "tool.py save <iid>" until it prints OK. Mix of kinds: ${MIX}.
Aim for items a strong model fails 30-60% of the time because they need care (several parts, every instance, implied consequences, the system's rules, a pattern to follow), never because they are ambiguous or graded on arbitrary specifics. Vary phrasing and difficulty. Make traps realistic: what would a capable but careless model actually do?
Return the item ids that passed the gate, and notes on anything you struggled with.`,
    { label: `author ${k}`, phase: 'Author', model: 'opus', effort: 'high', schema: SUMMARY }),
  (res, { k }) => agent(`${base}

You are an adversarial reviewer. Another agent wrote items a${k}i* on screens a${k}s* (in proposals/ and screens/). Review every proposal a${k}i*.json independently, as a skeptical senior product designer:
- Is the request unambiguous enough that good designers agree what counts as correct? Is its premise true on this screen?
- Is the reference actually good (follows the design system, complete, no needless changes)? Apply it mentally against the design.
- Are the criteria and checks fair per CONVENTIONS (accept every valid answer; reject the traps; no arbitrary specifics)? Try at least one valid answer of your own that differs from the reference with "tool.py try <iid> '<patch>'"; if fair answers fail, the checks are wrong.
- Is it hard for a real reason (not a one-property edit)?
Then: "tool.py approve <iid> '<note>'" if good; fix the proposal (edit proposals/<iid>.json, rerun save) and approve if fixable; otherwise "tool.py reject <iid> '<reason>'". Be strict: a smaller, fair, hard set beats a larger noisy one. The author reported: ${res ? res.notes.slice(0, 1500) : 'no report'}.
Return approved, rejected and fixed ids, and notes on recurring problems.`,
    { label: `review ${k}`, phase: 'Review', model: 'opus', effort: 'high', schema: REVIEW }))
return run