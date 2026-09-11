#!/usr/bin/env bash
# Per-boot setup for role-confusion: restore the venv from its snapshot, kernel, JupyterLab.
# pod_bootstrap.sh runs every named project's runpod/boot.sh.
exec bash "$(dirname "${BASH_SOURCE[0]}")/../setup_python_runpod.sh" --fast
