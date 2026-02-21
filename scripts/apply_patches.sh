#!/usr/bin/env bash
# Apply triton-shared patches to the triton main repository.
#
# Usage:
#   ./scripts/apply_patches.sh [TRITON_DIR]
#
# If TRITON_DIR is not specified, the script looks for a sibling directory
# named "triton" next to the triton-shared repo root.
#
# The patches remove the hard-coded NVIDIA/AMD backend assumptions from
# triton's build system so that triton can be built as a pure frontend
# (AST -> TTIR) with triton-shared acting as the only backend.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCHES_DIR="${REPO_ROOT}/patches"

# Resolve the triton source directory.
if [ $# -ge 1 ]; then
    TRITON_DIR="$(realpath "$1")"
else
    TRITON_DIR="$(realpath "${REPO_ROOT}/../triton")"
fi

if [ ! -d "${TRITON_DIR}" ]; then
    echo "ERROR: triton directory not found at '${TRITON_DIR}'"
    echo "  Pass the path explicitly:  $0 /path/to/triton"
    exit 1
fi

echo "Applying triton-shared patches to: ${TRITON_DIR}"

for patch in "${PATCHES_DIR}"/*.patch; do
    echo ""
    echo "  Applying $(basename "${patch}") ..."
    # --check first to give a helpful message if already applied.
    if git -C "${TRITON_DIR}" apply --check "${patch}" 2>/dev/null; then
        git -C "${TRITON_DIR}" apply "${patch}"
        echo "  OK"
    else
        # Try reverse-check: already applied?
        if git -C "${TRITON_DIR}" apply --check --reverse "${patch}" 2>/dev/null; then
            echo "  SKIPPED (patch already applied)"
        else
            echo "  ERROR: patch does not apply cleanly. Check for conflicts."
            exit 1
        fi
    fi
done

echo ""
echo "All patches applied successfully."
echo ""
echo "Next steps:"
echo "  export TRITON_PLUGIN_DIRS=${REPO_ROOT}"
echo "  pip install --no-build-isolation -e ${TRITON_DIR}"
