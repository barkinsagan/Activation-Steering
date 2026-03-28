# Research Project Template
# Claude Code ↔ Claude.ai Sync System

This template gives you two things:
1. A CLAUDE.md for your code/experiments (Claude Code side)
2. Project Instructions for your article writing (Claude.ai side)
3. A shared docs/ folder structure that bridges both

---

## PART 1: Your Repo Structure

```
research-project/
├── CLAUDE.md                          ← Claude Code reads this
├── src/                               ← Your experiment code
├── data/                              ← Datasets
├── results/                           ← Output files, plots, logs
│
├── docs/                              ← THE BRIDGE (shared with claude.ai)
│   ├── research-log.md                ← Running log of what you did & found
│   ├── experiment-index.md            ← Quick reference of all experiments
│   ├── findings.md                    ← Key results in plain language
│   ├── article-drafts/               ← Article drafts (optional)
│   └── references.md                  ← Papers, links, sources
│
└── .claude/
    └── commands/
        └── log-experiment.md          ← Custom command to auto-log
```

The docs/ folder is the single source of truth.
- Claude Code WRITES to it after experiments.
- You UPLOAD it to your claude.ai Project for article writing.
- Both environments read the same context.

---

## PART 2: CLAUDE.md (for Claude Code)

Copy this into your project root as CLAUDE.md.
Fill in the [BRACKETS].

```markdown
# [Project Name]

[One-line description of what this research is about.]

## Tech Stack
- Language: [Python 3.12 / R / Julia / etc.]
- Key libraries: [pytorch, numpy, pandas, matplotlib, etc.]
- Environment: [conda / venv / docker]
- Data storage: [local / S3 / etc.]

## Project Structure
- /src — experiment code, models, utilities
- /data — raw and processed datasets
- /results — outputs, plots, checkpoints, logs
- /docs — research log, findings, article drafts (SHARED WITH CLAUDE.AI)

## Commands
- [pip install -r requirements.txt] — setup
- [python src/train.py] — run training
- [python src/evaluate.py] — run evaluation
- [pytest tests/] — run tests
- [python src/plot_results.py] — generate plots

## Conventions
- All experiments get a unique ID: exp_YYYYMMDD_shortname
- Results go in /results/{experiment_id}/
- Every experiment must have a logged entry in docs/research-log.md
- Use seed=42 for reproducibility unless testing variance
- Plots: save as both .png and .svg, always include axis labels and titles

## Experiment Workflow
When I ask you to run an experiment:
1. Create a unique experiment ID
2. Write/modify the code
3. Run it and capture results
4. Auto-generate an entry in docs/research-log.md using this format:

### exp_YYYYMMDD_shortname
**Date:** YYYY-MM-DD
**Goal:** What we were testing
**Method:** What we did (brief)
**Code:** Which files were involved
**Results:** Key numbers, observations
**Plots:** Links to generated plots
**Verdict:** What this means / what to try next
**Status:** [success / partial / failed / inconclusive]

5. Update docs/experiment-index.md with a one-line summary
6. If the result is significant, add it to docs/findings.md

## Writing Bridge
When I say "prep for writing" or "sync for article":
1. Update docs/research-log.md with any unlogged experiments
2. Update docs/findings.md with latest key results
3. Generate a docs/writing-brief.md that summarizes:
   - What we've proven so far
   - Key numbers and results
   - What plots are available
   - Open questions remaining
   - Suggested narrative arc for an article
4. Remind me to upload docs/ to my claude.ai Project

## Important Rules
- NEVER delete or overwrite previous experiment logs — append only
- Always save reproducible configs (hyperparams, seeds, data splits)
- When results are surprising, re-run with different seed before concluding
- Commit after every successful experiment
```

---

## PART 3: Custom Command (optional but powerful)

Save this as .claude/commands/log-experiment.md in your repo.
Then you can type /project:log-experiment in Claude Code.

```markdown
---
description: Log the most recent experiment to docs/research-log.md
---

Review the most recent code changes and results. Then:

1. Generate a unique experiment ID (exp_YYYYMMDD_shortname)
2. Append a complete entry to docs/research-log.md with:
   - Date, goal, method, code files, results, plots, verdict, status
3. Add a one-line entry to docs/experiment-index.md
4. If the result changes our understanding, update docs/findings.md
5. Commit the doc changes with message: "docs: log {experiment_id}"
```

---

## PART 4: Project Instructions for Claude.ai (Article Writing)

Create a Project in claude.ai for this research.
Upload the docs/ folder contents as Project knowledge.
Paste this into Project Instructions:

```
## My Research Project: [NAME]

I'm writing articles and papers based on experiments I run in Claude Code.
My experiment logs, findings, and research context are in the uploaded docs.

### Your role
You are my research writing partner. You help me:
- Turn experiment findings into clear, engaging articles
- Structure arguments from my data
- Write in my voice (see style notes below)
- Suggest visualizations that would strengthen the narrative
- Catch logical gaps in my arguments
- Cite my own experiments properly

### Key uploaded docs
- research-log.md — Chronological log of every experiment I've run
- experiment-index.md — Quick reference of all experiments
- findings.md — Key results in plain language
- writing-brief.md — Latest summary prepared for writing (if available)
- references.md — External sources and papers

### Writing rules
- Write in [first person / third person / academic style]
- Target audience: [developers / researchers / general tech audience]
- Tone: [conversational but rigorous / formal academic / tutorial-style]
- Always reference specific experiment IDs when citing my results
- Never invent results — only use data from my experiment logs
- Flag when my data doesn't fully support a claim I want to make
- Suggest where I need more experiments before publishing

### Session commands
- "outline [topic]" — Create an article outline using my findings
- "draft [section]" — Write a section draft based on my experiment data
- "strengthen this" — Find experiment data that supports or challenges my argument
- "what's missing" — Identify gaps where I need more experiments before this article is ready
- "cite my work" — Reference specific experiment IDs from my logs
- "simplify" — Rewrite the last section for a less technical audience

### Current article status

(Update this after each writing session)

**Working on:** [article title or topic]
**Stage:** [outlining / drafting / revising / ready for review]
**Sections done:** [list]
**Sections remaining:** [list]
**Experiments I still need to run:** [list — take these back to Claude Code]
```

---

## PART 5: The Sync Workflow

### After running experiments (Claude Code → Claude.ai):

1. In Claude Code, say: "prep for writing" or "sync for article"
2. Claude updates docs/research-log.md, findings.md, writing-brief.md
3. Commit the changes
4. Go to your claude.ai Project
5. Delete old docs, upload the updated docs/ files
6. Start writing with full experiment context

### After writing sessions (Claude.ai → Claude Code):

1. In claude.ai, if you realize you need more experiments, say: "what experiments do I need?"
2. Claude generates a list based on gaps in your article
3. Copy that list
4. Go to Claude Code and say: "Here are experiments I need to run for my article: [paste list]"
5. Run them, log them, cycle back to writing

### The cycle:

```
┌─────────────────┐         docs/ folder         ┌─────────────────┐
│   Claude Code   │ ──── write logs & findings ──→│   Claude.ai     │
│                 │                               │                 │
│  Run experiments│         upload docs/          │  Write articles │
│  Log results    │                               │  Structure args │
│  Generate plots │                               │  Draft sections │
│                 │←── "need more experiments" ───│                 │
└─────────────────┘      copy experiment list      └─────────────────┘
```

---

## PART 6: Starter Templates for docs/

### docs/research-log.md
```markdown
# Research Log

## exp_20260325_baseline
**Date:** 2026-03-25
**Goal:** Establish baseline performance
**Method:** [describe]
**Code:** src/train.py, src/evaluate.py
**Results:** Accuracy: X%, Loss: Y
**Plots:** results/exp_20260325_baseline/loss_curve.png
**Verdict:** Baseline established. Ready to test [hypothesis].
**Status:** success
```

### docs/experiment-index.md
```markdown
# Experiment Index

| ID | Date | Goal | Status | Key Result |
|----|------|------|--------|------------|
| exp_20260325_baseline | 2026-03-25 | Establish baseline | success | Acc: X% |
```

### docs/findings.md
```markdown
# Key Findings

## Finding 1: [Title]
**Supported by:** exp_20260325_baseline
**Summary:** [What we found in plain language]
**Confidence:** [high / medium / low — based on how many experiments confirm this]
**Caveats:** [Limitations, edge cases, things to verify]
```

### docs/references.md
```markdown
# References

## Papers
- [Author, "Title", Year](URL) — [one-line relevance note]

## Tools & Libraries
- [Library name](URL) — [what we use it for]

## Datasets
- [Dataset name](URL) — [description, size, license]
```
