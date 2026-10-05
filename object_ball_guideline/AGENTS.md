# AGENTS.md — Codex Operating Contract for 8-Ball Pool Target-Line Segmentation

## Mandatory project memory

Before doing any meaningful work in this repository, Codex must read the canonical project memory files in this order:

1. `docs/experiments/Supervised_Target_Line_Progress.md`
2. `docs/experiments/HANDOFF.md`
3. `docs/experiments/EXPERIMENT_LOG.md`
4. `docs/experiments/DECISIONS.md`

`docs/experiments/Supervised_Target_Line_Progress.md` is the canonical source of truth for task-specific facts in this repository.

If `docs/experiments/Supervised_Target_Line_Progress.md` conflicts with older notes, generic instructions, stale comments, old experiment names, or assumptions in code, prefer `docs/experiments/Supervised_Target_Line_Progress.md` unless the user explicitly says it has been superseded.

Codex must consult `docs/experiments/Supervised_Target_Line_Progress.md` before changing or judging:

- target definition,
- label rules,
- mask generation,
- dataset construction,
- data splits,
- negative folders,
- augmentation logic,
- candidate generation,
- ball/crop proposal logic,
- inference arbitration,
- rescue policies,
- reranker logic,
- checkpoint promotion,
- benchmark gates,
- end-to-end evaluation,
- conditioned-model work.

Do not promote a checkpoint, manifest, rescue rule, target-definition change, or model-policy change unless it is consistent with `docs/experiments/Supervised_Target_Line_Progress.md`.

## Canonical target definition

The positive target is:

- the object-ball outgoing white guideline after cue-ball contact,
- the line that belongs to the target object ball and indicates that object ball's travel direction.

The following are negative/reject material:

- cue-ball-connected aiming line,
- long cue-ball-to-contact aiming line,
- cue stick,
- forbidden-circle line,
- unrelated white UI lines,
- any other white line when the true object-ball outgoing guideline is absent.

Do not mark a prediction as correct merely because it detects a white line. It must detect the object-ball outgoing guideline.

If a visual example is ambiguous, stop and request the relevant raw frame, mask, overlay, prediction, or crop before changing labels or judging model behavior.

## Repository mission

Build, train, evaluate, and improve a local ML system that detects and segments the object-ball outgoing white guideline in 8-ball pool.

The target can be:

- extremely thin,
- very short,
- faint,
- anti-aliased,
- partially visible,
- disconnected from the cue-ball aiming line,
- visually confused with other white UI/game elements.

The goal is to train and control our own model locally. Do not replace the core project with a hosted external model or API unless the user explicitly asks.

Classical computer-vision logic, external checkpoints, and heuristic rescue paths may be used as baselines, diagnostics, rescue branches, or comparison tools, but promotion must be justified by end-to-end benchmark evidence.

## Hardware assumptions

Optimize for the local laptop first:

- NVIDIA RTX 5090 Laptop GPU with 24 GB VRAM,
- 64 GB RAM,
- Intel Core Ultra 9 275HX,
- fast Samsung SSD,
- Windows host with possible WSL/Linux workflows.

Use local GPU training when relevant. Do not recommend cloud training unless there is a specific technical reason.

## Operating mode

Work autonomously, but be evidence-led.

For every non-trivial task:

1. Read the mandatory project memory files.
2. Inspect relevant repository files before proposing edits.
3. Identify the current data flow, model flow, and evaluation flow.
4. Separate verified facts from hypotheses.
5. Generate multiple plausible explanations or improvement paths when uncertainty exists.
6. Rank options by evidence, expected information gain, implementation cost, regression risk, and compute cost.
7. Choose the highest-value next step.
8. Make the smallest defensible change.
9. Validate the change as much as feasible.
10. Update persistent experiment memory.

Do not guess when you can verify.

Do not anchor on the user's current theory if repository evidence, logs, metrics, or visual artifacts point elsewhere. Push back clearly and explain why.

Ask direct questions only when missing information materially affects the next step.

High-value artifacts to request when needed:

- raw frames,
- masks,
- prediction masks,
- overlays,
- zoomed failure crops,
- false positives,
- false negatives,
- hard negatives,
- metric tables,
- training logs,
- config files,
- checkpoints,
- dataset manifests,
- split definitions.

## Repository inspection requirements

Before making meaningful changes, inspect the relevant subset of:

- `docs/experiments/Supervised_Target_Line_Progress.md`,
- `docs/experiments/HANDOFF.md`,
- `docs/experiments/EXPERIMENT_LOG.md`,
- `docs/experiments/DECISIONS.md`,
- training entry points,
- dataset classes,
- mask-loading logic,
- split-generation logic,
- augmentation code,
- crop/proposal code,
- model definitions,
- loss functions,
- metric implementations,
- inference code,
- reranker code,
- rescue/arbitration logic,
- benchmark/evaluation scripts,
- visualization/overlay tools,
- tests.

If a file, command, checkpoint, or path mentioned in older notes no longer exists, say so and adjust the plan using current repository evidence.

## Research policy

Use repository evidence first for project-specific behavior.

Use current external research or documentation when the answer depends on:

- package/API behavior,
- PyTorch/CUDA/AMP behavior,
- model architecture choices,
- loss or metric definitions,
- current best practices,
- unstable dependency behavior,
- Codex/OpenAI configuration,
- unclear library/version behavior.

Preferred sources:

1. official documentation,
2. primary papers,
3. upstream repositories,
4. issue trackers only when needed for active breakage.

Use Context7 for library documentation when available. Use OpenAI Developer Docs for Codex/OpenAI behavior. Use web search if MCP documentation and repository evidence are insufficient.

When research affects a decision, summarize the conclusion in the experiment notes so future sessions do not rediscover the same thing.

## Thin-line segmentation rules

This is a sparse thin-structure segmentation problem with a strong white-color prior and high risk of false positives.

Do not rely on a single scalar metric.

Evaluation should consider:

- IoU,
- Dice/F1,
- precision,
- recall,
- false-positive pixels,
- false-negative pixels,
- zero-IoU failures,
- short-line recall,
- line continuity,
- mask shift,
- endpoint quality,
- threshold sensitivity,
- visual overlays,
- hard-case subsets.

Important failure categories:

- object-ball outgoing line missed,
- cue-ball-connected aiming line falsely accepted,
- long cue-ball line selected instead of outgoing object-ball line,
- cue stick falsely accepted,
- forbidden-circle line falsely accepted,
- hollow reticle or ring selected,
- white UI text/icon selected,
- ball/table highlight selected,
- line broken or shifted,
- mask/image misalignment,
- inconsistent label thickness,
- candidate proposal misses the true target ball/line.

Always inspect overlays when available. For this project, qualitative visual evidence is not optional when judging subtle segmentation behavior.

## Baseline and promotion policy

Use a baseline-first methodology.

Default development order:

1. Read canonical progress and handoff files.
2. Audit current repository and dataset state.
3. Validate images, masks, overlays, and split logic.
4. Establish or preserve a classical CV sanity baseline where relevant.
5. Establish or preserve a small neural segmentation baseline.
6. Improve data, loss, sampling, thresholding, candidate generation, rescue logic, and evaluation.
7. Escalate architecture complexity only when evidence justifies it.

Do not promote from crop-level training metrics alone.

Promotion requires end-to-end validation against the current promoted benchmark described in `docs/experiments/Supervised_Target_Line_Progress.md`.

Before promotion, compare:

- normal validation split,
- hard validation split,
- zero-IoU failures,
- qualitative overlays,
- known canonical validation examples,
- known rejected branches or regressions.

A checkpoint or branch that improves one subset but materially regresses another should usually become a targeted rescue candidate, not a promoted primary replacement.

## Local training rules

Prefer short, information-dense experiments over long blind runs.

Before a long run:

1. validate dataset and masks,
2. run or inspect split sanity checks,
3. run a tiny overfit test if model/training code changed,
4. run a short smoke training run,
5. generate overlays,
6. inspect metrics and failure cases,
7. only then run a longer experiment.

Do not launch multiple full GPU training jobs in parallel on the laptop.

Use mixed precision, gradient accumulation, cropping, tiling, persistent workers, caching, or WSL/Linux where they are technically justified.

For every proposed run, specify:

- hypothesis,
- exact command,
- config or args,
- input data/split,
- checkpoint source,
- expected outputs,
- expected runtime class: smoke / short / medium / long,
- success criteria,
- failure criteria,
- artifacts to inspect afterward.

## Debugging policy

For every non-trivial bug:

1. capture the exact symptom,
2. capture the exact command,
3. inspect the stack trace or failing output,
4. identify the involved files,
5. generate ranked root-cause hypotheses,
6. test or eliminate hypotheses with evidence,
7. implement the smallest safe fix,
8. run the narrowest valid verification,
9. document the result.

Check for:

- stale paths,
- missing files,
- wrong checkpoint format,
- incompatible model config,
- tensor shape mismatch,
- dtype/device mismatch,
- mask polarity inversion,
- train/val leakage,
- empty split,
- empty mask bug,
- threshold bug,
- wrong target definition,
- reranker/candidate mismatch,
- WSL/Windows path mismatch,
- package/API version mismatch,
- NaNs/infs,
- silent metric bugs.

Do not claim a bug is fixed unless validation supports it.

## Code-change rules

Make the smallest defensible change that advances the objective.

Keep unrelated files untouched.

Do not perform broad refactors unless the design itself is the root cause and the user has enough context to approve the direction.

Do not commit, push, delete datasets, delete checkpoints, delete large artifacts, or rewrite experiment history unless the user explicitly asks.

Never overwrite history in experiment logs. Append new information.

Before finalizing a change, report:

- files changed,
- why the change should work,
- commands run,
- validation result,
- what remains unverified,
- next recommended command or experiment.

## Experiment memory requirements

Maintain these files as part of normal work:

- `docs/experiments/Supervised_Target_Line_Progress.md`,
- `docs/experiments/EXPERIMENT_LOG.md`,
- `docs/experiments/DECISIONS.md`,
- `docs/experiments/HANDOFF.md`.

`docs/experiments/Supervised_Target_Line_Progress.md` is the canonical historical record and should not be casually rewritten. Append or update it only when the user asks or when a major promoted result, target-rule change, benchmark result, or project-state correction occurs.

For normal work, update:

- `EXPERIMENT_LOG.md` for detailed experiment notes,
- `DECISIONS.md` for accepted/rejected technical decisions,
- `HANDOFF.md` for the current state and next action.

Each meaningful update should include:

- date/time,
- objective,
- hypothesis,
- exact change,
- files touched,
- command(s),
- metrics,
- visual evidence,
- result,
- what helped,
- what failed,
- open questions,
- exact next step.

Write for future Codex/GPT sessions that may start with no conversation history.

## Subagent use

Use specialized subagents for complex or parallelizable work.

Preferred subagents:

- `data_loader`: dataset, masks, overlays, augmentations, split logic, leakage, label quality.
- `trainer`: training loop, loss, optimizer, scheduler, checkpointing, AMP, reproducibility, local hardware.
- `evaluator`: metrics, overlays, thresholding, post-processing, hard-case analysis, promotion gates.
- `debugger`: crashes, NaNs, shape/device/dtype/path bugs, package/API regressions.
- `experiment_tracker`: experiment logs, decisions, handoff updates, future-session memory.
- `patch_generator`: final minimal patch after the technical direction is chosen.

When using subagents:

1. give each subagent a narrow job,
2. wait for results,
3. synthesize into one ranked recommendation,
4. do not blindly apply conflicting subagent recommendations,
5. validate before claiming success.

## Required response structure for major tasks

For major debugging, training, evaluation, or design work, respond with:

1. Current understanding
2. Mandatory memory files checked
3. Evidence inspected
4. Ranked hypotheses or options
5. Recommended next action
6. Exact implementation or experiment plan
7. Verification commands and expected outputs
8. Files changed or to be changed
9. Risks and remaining uncertainty
10. Experiment-memory update summary

For small tasks, be concise, but still obey the mandatory project-memory rule if the task touches target definition, labels, training, evaluation, inference, or promotion.

## Definition of done

A task is not done until:

- relevant project memory was checked,
- relevant code/artifacts were inspected,
- the change or recommendation is evidence-backed,
- validation was performed or clearly marked as pending,
- experiment memory was updated when appropriate,
- remaining uncertainty is stated,
- the next best step is clear.
