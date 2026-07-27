#!/bin/bash
set -euo pipefail

# RunPod variant of setup_python.sh, for running demo/role-probe-demo_runpod.ipynb
# on a single RTX 4090 pod (RunPod PyTorch template, network volume mounted at /workspace).
#
# Differences from setup_python.sh:
#  - Paths live on /workspace (the network volume), so the venv, uv caches, and HF model
#    cache all survive pod stop/terminate. Everything outside /workspace is wiped.
#  - transformers v5 instead of the 4.57.5 pin: the demo notebook hard-requires v5.
#  - `kernels` is installed into the version window transformers actually accepts
#    (it enforces a floor AND a ceiling; outside the window it silently dequantizes
#    gpt-oss-20b's MXFP4 weights to bf16 (~48 GB), which cannot fit on a 4090).
#  - No flash-attn wheel: the runpod notebook uses eager attention (FA3 is Hopper-only).
#
# Re-run this on every fresh pod: the venv on /workspace persists, but the Jupyter kernel
# registration lives in the container's /root and does not. Use `--fast` for those re-runs:
# it skips all package installs/audits (even no-op audits cost minutes on the network
# filesystem) and only redoes the per-boot container state. Run the full script (no flag)
# the first time on a volume, or after changing any package pins.

# Set constants
PROJECT_DIR="/workspace/prompt-injection-as-role-confusion"
VENV_DIR="$PROJECT_DIR/.venv"
KERNEL_NAME="role-analysis-uv"

# Persist across restarts
export UV_CACHE_DIR="/workspace/.uv-cache"
export UV_PYTHON_INSTALL_DIR="/workspace/.uv-python"
export UV_HTTP_TIMEOUT=120


# Install the Hugging Face token if one is stored on the volume. huggingface_hub reads
# ~/.cache/huggingface/token automatically; that path is wiped on every pod boot, so the
# durable copy lives at /workspace/secrets/hf_token (create it once: paste the token into
# that file over SSH, chmod 600).
install_hf_token() {
  if [ -f /workspace/secrets/hf_token ]; then
    mkdir -p "$HOME/.cache/huggingface"
    install -m 600 /workspace/secrets/hf_token "$HOME/.cache/huggingface/token"
    echo "HF token installed from /workspace/secrets/hf_token."
  fi
}

# Point SSH at the repo's GitHub deploy key (generated on the pod, stored on the volume,
# registered as a deploy key on the GitHub repo). ~/.ssh is wiped on every pod boot, and the
# network volume forces mode 666 on everything (chmod is ignored), which ssh refuses for
# private keys - so copy the key to local disk with real 600 perms each boot.
install_github_key() {
  if [ -f /workspace/secrets/github_deploy_key ]; then
    mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
    install -m 600 /workspace/secrets/github_deploy_key "$HOME/.ssh/github_deploy_key"
    if ! grep -q github_deploy_key "$HOME/.ssh/config" 2>/dev/null; then
      printf "Host github.com\n  IdentityFile ~/.ssh/github_deploy_key\n  IdentitiesOnly yes\n  StrictHostKeyChecking accept-new\n" >> "$HOME/.ssh/config"
      chmod 600 "$HOME/.ssh/config"
    fi
    echo "GitHub deploy key configured."
  fi
}


# ---------- 0. Fast mode: --fast redoes only what a pod boot wipes ----------
if [ "${1:-}" = "--fast" ]; then
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "ERROR: no venv at $VENV_DIR - run once without --fast first." >&2
    exit 1
  fi
  install_hf_token
  install_github_key
  "$VENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Role analysis (uv)"
  SITE_DIR="$("$VENV_DIR/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  printf "%s\n" "$PROJECT_DIR" > "$SITE_DIR/add_path_analysis.pth"
  echo "Fast setup done. Kernel: $KERNEL_NAME  |  Python: $("$VENV_DIR/bin/python" -V)"
  exit 0
fi


# ---------- 1. Install UV (idempotent) ----------
cd "$PROJECT_DIR"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
case ":$PATH:" in *":$HOME/.local/bin:"*) :;; *) export PATH="$HOME/.local/bin:$PATH";; esac # Make sure uv is in PATH


# ---------- 2. Install venv ----------
# Set Python 3.12 location (download if needed)
UVPY="$(uv python find 3.12 2>/dev/null || (uv python install 3.12 >/dev/null 2>&1 && uv python find 3.12))"

# Create the venv with Python 3.12 (uv will download 3.12 if needed)
if [ -x "$VENV_DIR/bin/python" ]; then
  # Check that it actually runs and is 3.12
  if ! "$VENV_DIR/bin/python" -c 'import sys; exit(0 if sys.version_info[:2]==(3,12) else 1)'; then
    echo "Repairing venv Python links (keeping installed packages)..."
    "$UVPY" -m venv "$VENV_DIR" --upgrade --without-pip   # relink python, keep site-packages
    "$VENV_DIR/bin/python" -m ensurepip --upgrade || true
  else
    echo "Using existing venv at $VENV_DIR"
  fi
else
  if [ -d "$VENV_DIR" ]; then
    echo "Repairing existing venv (python missing)…"
    "$UVPY" -m venv "$VENV_DIR" --upgrade --without-pip
    "$VENV_DIR/bin/python" -m ensurepip --upgrade || true
  else
    echo "Creating venv…"
    uv venv "$VENV_DIR" --python 3.12 --seed
  fi
fi


# ---------- 3. Install packages ----------
uv pip install --python "$VENV_DIR/bin/python" --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1

uv pip install --python "$VENV_DIR/bin/python" \
  "transformers>=5.0.0,<6" hf_transfer==0.1.9 accelerate==1.12.0 triton==3.5.1 \
  tiktoken==0.12.0 blobfile==3.1.0 \
  compressed-tensors==0.13.0 \
  plotly pandas "scikit-learn==1.7.*" python-dotenv pyyaml tqdm termcolor \
  datasets zstandard

# `kernels` must land inside the window the installed transformers accepts. transformers checks
# KERNELS_MIN_VERSION <= v < KERNELS_MAX_VERSION but only ever complains about the floor, so an
# unconstrained install can sit above the ceiling and trigger the silent bf16 dequantize fallback.
KERNELS_SPEC="$("$VENV_DIR/bin/python" - <<'PYEOF'
from transformers.utils import import_utils as u
print(f'kernels>={u.KERNELS_MIN_VERSION},<{u.KERNELS_MAX_VERSION}')
PYEOF
)"
echo "Installing '$KERNELS_SPEC' (window read from installed transformers)"
uv pip install --python "$VENV_DIR/bin/python" "$KERNELS_SPEC"

# Verify transformers will actually use it (a fresh interpreter, since the check is cached)
"$VENV_DIR/bin/python" - <<'PYEOF'
from transformers.utils import is_kernels_available
import transformers, kernels
print(f'transformers {transformers.__version__} | kernels {kernels.__version__}')
assert is_kernels_available(), 'transformers rejects the installed kernels version - MXFP4 would silently dequantize'
print('is_kernels_available(): True')
PYEOF

# Optional, needed for ReAct loop agent testing
uv pip install --python "$VENV_DIR/bin/python" openai

# RAPIDS (search NVIDIA index for cudf/cuML — PyPI/extra index support is documented)
uv pip install --python "$VENV_DIR/bin/python" libucx-cu12==1.18.1 ucx-py-cu12==0.45.0 # Dependencies from pypi for RAPIDS - install separately to avoid error
# 25.10 rather than the original 25.9 pin: pypi.nvidia.com no longer serves 25.9 (only <25.9 and >25.10).
uv pip install --python "$VENV_DIR/bin/python" --extra-index-url https://pypi.nvidia.com "cudf-cu12==25.10.*" "cuml-cu12==25.10.*"

# Optional: only needed for exporting plotly figures to static images (the demo renders inline).
# uv pip install --python "$VENV_DIR/bin/python" kaleido
# uv run --python "$VENV_DIR/bin/python" plotly_get_chrome -y
# apt update && apt-get install -y libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2

# ---------- 4. Setup Jupyter ----------
# Jupyter (server + kernel + widgets + nbformat)
uv pip install --python "$VENV_DIR/bin/python" jupyterlab jupyter_server ipykernel ipywidgets nbformat notebook

# Jupyter kernel (visible to any server, including the JupyterLab the RunPod template starts)
"$VENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Role analysis (uv)"


# ---------- 5. Add import paths ----------
SITE_DIR="$("$VENV_DIR/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf "%s\n" "$PROJECT_DIR" > "$SITE_DIR/add_path_analysis.pth"

# Final
install_hf_token
install_github_key
echo "Done. Kernel: $KERNEL_NAME  |  Python: $("$VENV_DIR/bin/python" -V)"
echo "Open demo/role-probe-demo_runpod.ipynb in JupyterLab and select the 'Role analysis (uv)' kernel."
