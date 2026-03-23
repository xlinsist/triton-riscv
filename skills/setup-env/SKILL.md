---
name: setup-env
description: Build a reusable repository setup workflow and produce executable setup documentation. Use when users ask to research a repo, generate or refresh SETUP_GUIDE.md, record step-by-step installation attempts in INSTALLATION.md, and finish with a one-click minimal verification command that can be copied and run directly.
---

# Setup Env

## Overview

Create setup documentation from repository facts, then validate and document a reproducible build path.

## Workflow

1. Research repository setup signals.
2. Draft `SETUP_GUIDE.md` from research results.
3. Execute build steps from `README.md` (or primary docs) and record every attempt in `INSTALLATION.md`.
4. Capture a one-click minimal verification path in `INSTALLATION.md`.
5. Run consistency checks and finalize.

## Step 1: Research the Repository

Prefer this discovery order:

- `README.md` (primary source of intended setup path)
- Build/package manifests (`pyproject.toml`, `setup.py`, `requirements*.txt`, `CMakeLists.txt`, etc.)
- Existing setup docs (if present): `SETUP_GUIDE.md`, `INSTALLATION.md`
- Helper scripts under `scripts/` (only if referenced by docs or needed to complete setup)

Extract:

- Prerequisites and required tool versions
- Build/install command sequence
- Required environment variables
- Smallest runnable validation target (single command/test)

## Step 2: Draft SETUP_GUIDE.md

Write a clean setup guide with:

- Scope and assumptions
- Prerequisites
- Step-by-step setup and build commands
- Runtime environment variables
- Troubleshooting hints

Use variables/placeholders instead of hardcoded machine paths unless the user explicitly requests fixed paths.

## Step 3: Execute and Record in INSTALLATION.md

Follow commands from `README.md` first. If adjustments are needed, apply the smallest change and record it.

In `INSTALLATION.md`, keep a chronological log:

- Timestamp (UTC)
- Command
- Result (`pass`/`fail`)
- Notes (why changed, if changed)

Do not hide failed attempts; keep them with resolution notes.

## Step 4: Add One-Click Minimal Verification

At the end of `INSTALLATION.md`, add a copy-paste block that:

- sets required environment variables with overridable shell variables
- runs the smallest end-to-end verification command
- states expected success output

Template shape:

```bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
# ... derive or export required variables
# ... run one minimal verification command
```

## Step 5: Consistency Checks

Before finishing:

```bash
rg -n "TODO|TBD|FIXME" SETUP_GUIDE.md INSTALLATION.md
```

Also verify:

- variable names are consistent between `SETUP_GUIDE.md` and `INSTALLATION.md`
- commands in the one-click block are directly runnable
- dates are concrete (UTC)

## Reference

For reusable wording and checklists, use:

- `references/setup-doc-template.md`
