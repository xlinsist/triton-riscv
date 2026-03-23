# Setup Documentation Template

## Goal

Produce two executable documents:

- `SETUP_GUIDE.md`: clean setup instructions for first-time users
- `INSTALLATION.md`: real build log + final one-click verification command

## Research Checklist

1. Read `README.md` first.
2. Extract build/install commands from repository manifests and scripts.
3. Identify required env vars.
4. Pick the smallest runnable validation target.

## INSTALLATION.md Log Format

Use a chronological table:

| Time (UTC) | Command | Result | Notes |
|---|---|---|---|

Rules:

- Keep failures with resolution notes.
- Prefer copy-paste commands.
- Use concrete dates.

## One-Click Minimal Verification Block

```bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
# export required env vars (overridable)
# run one minimal verification command
```

Expected output should be written explicitly (for example: `1 passed`).
