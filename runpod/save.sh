#!/usr/bin/env bash
# Save the role-confusion pod venv so the next pod gets it: rewrite
# runpod/requirements.lock.txt from the venv, then re-snapshot the venv to the volume
# if its package set changed.
# Run it before stopping the pod, or right after installing something worth keeping
# (pod_save.sh runs it for you; so does a full setup_python_runpod.sh run).
#
# The lockfile is the venv's exact package set (`uv pip freeze`), so anything installed
# by hand is recorded too; setup_python_runpod.sh installs it with `uv pip sync` on a
# full rebuild. It is left modified for you to commit.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
VENV_DIR=/opt/role-venv
SNAPSHOT=/workspace/code/prompt-injection-as-role-confusion/venv-snapshot.tar
SNAPSHOT_PATHS=(opt/role-venv)
# uv's own Python, when the image lacked a matching one: the venv links into it.
[[ -d /opt/uv-python ]] && SNAPSHOT_PATHS+=(opt/uv-python)
CHECK_PACKAGES=(torch transformers kernels)
LOCK="$REPO_DIR/runpod/requirements.lock.txt"
LOCK_INDEXES=(--extra-index-url https://download.pytorch.org/whl/cu128
              --extra-index-url https://pypi.nvidia.com)
LOG="$REPO_DIR/dev.log"

log() { echo "[$(date +%H:%M:%S)] [save] $*" | tee -a "$LOG"; }

[[ -x "$VENV_DIR/bin/python" ]] || { log "no venv at $VENV_DIR - nothing to save"; exit 0; }
PY="$VENV_DIR/bin/python"

# ---------- check: interpreter starts, key packages installed (metadata only, ~0.1s) ----------
if ! "$PY" -c 'import importlib.metadata as m, sys; [m.version(p) for p in sys.argv[1:]]' \
        "${CHECK_PACKAGES[@]}" 2>>"$LOG"; then
    log "CHECK FAILED: venv is broken or missing one of: ${CHECK_PACKAGES[*]} - not saving"
    exit 1
fi

# ---------- lockfile: the venv's exact package set ----------
FREEZE="$(uv pip freeze --python "$PY" 2>>"$LOG")"
NEW_LOCK="$(
    echo "# Exact package set of $VENV_DIR, written by runpod/save.sh from \`uv pip freeze\`."
    echo "# setup_python_runpod.sh installs it with \`uv pip sync\`. Don't edit by hand:"
    echo "# install into the venv, then run runpod/save.sh."
    printf '%s %s\n' "${LOCK_INDEXES[@]}"
    printf '%s\n' "$FREEZE" | python3 -c '
import re, sys
canon = lambda n: re.sub(r"[-_.]+", "-", n).lower()
out = []
for line in sys.stdin.read().splitlines():
    m = re.match(r"([A-Za-z0-9._-]+)(\s*(==|@).*)", line)
    out.append(canon(m.group(1)) + m.group(2) if m else line)
print("\n".join(sorted(out)))'
)"
if [[ "$NEW_LOCK" != "$(cat "$LOCK" 2>/dev/null)" ]]; then
    printf '%s\n' "$NEW_LOCK" > "$LOCK"
    log "updated $LOCK - commit it"
fi

# ---------- snapshot, only if the package set changed ----------
# The freeze recorded inside the venv travels with the snapshot, so after a restore it
# describes exactly what the snapshot holds.
MARK="$VENV_DIR/.snapshot-freeze.txt"
if [[ -f "$SNAPSHOT" && -f "$MARK" && "$FREEZE" == "$(cat "$MARK")" ]]; then
    log "packages unchanged since the last snapshot - nothing to save"
    exit 0
fi
[[ -f "$MARK" ]] && diff "$MARK" <(printf '%s\n' "$FREEZE") | sed -n 's/^</  -/p; s/^>/  +/p' | tee -a "$LOG"
printf '%s\n' "$FREEZE" > "$MARK"
log "snapshotting ${SNAPSHOT_PATHS[*]} -> $SNAPSHOT"
rm -f "$SNAPSHOT.tmp"
if tar -cf "$SNAPSHOT.tmp" -C / "${SNAPSHOT_PATHS[@]}" 2>>"$LOG" && mv -f "$SNAPSHOT.tmp" "$SNAPSHOT"; then
    log "snapshot saved ($(du -sh "$SNAPSHOT" | cut -f1))"
else
    rm -f "$SNAPSHOT.tmp" "$MARK"  # no marker, so the next save retries
    log "SNAPSHOT FAILED - the previous snapshot is untouched"
    exit 1
fi
