#!/bin/bash
set -euo pipefail

# Local (macOS / Apple Silicon) READ-ONLY environment.
#
# This is NOT the environment the experiments run in -- see setup_python.sh (CUDA) or
# setup_python_runpod.sh for that. The purpose here is to give an editor/IDE a real
# interpreter so imports resolve, go-to-definition works, and type hints show up while
# reading the code on a machine with no NVIDIA GPU.
#
# Deliberately omitted, because they have no macOS build and would abort the install:
#   triton, flash-attn, cupy, cudf-cu12 / cuml-cu12 (RAPIDS), the apt-get block.
# utils/probes.py imports cupy at module scope, so that one module will not import
# locally. Everything else will.
#
# torch here is the default PyPI build: CPU plus Apple's MPS backend. Any code path
# guarded by torch.cuda.is_available() will take the CPU branch.

PROJECT_DIR="/Users/clayjones/code/prompt-injection-as-role-confusion"
VENV_DIR="$PROJECT_DIR/.venv"
KERNEL_NAME="role-analysis-local"

export UV_CACHE_DIR="$PROJECT_DIR/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/.uv-python"
export UV_HTTP_TIMEOUT=120

if [ "$(uname -s)" != "Darwin" ]; then
  echo "This script is for macOS. On Linux/CUDA use setup_python.sh." >&2
  exit 1
fi

cd "$PROJECT_DIR"


# ---------- 1. Install UV (idempotent) ----------
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
case ":$PATH:" in *":$HOME/.local/bin:"*) :;; *) export PATH="$HOME/.local/bin:$PATH";; esac


# ---------- 2. Create venv ----------
if [ -x "$VENV_DIR/bin/python" ]; then
  echo "Using existing venv at $VENV_DIR"
else
  echo "Creating venv at $VENV_DIR ..."
  uv venv "$VENV_DIR" --python 3.12 --seed
fi


# ---------- 3. Install packages ----------
# Versions pinned to match setup_python.sh so the API surface you read locally is the
# same one that runs remotely.
uv pip install --python "$VENV_DIR/bin/python" torch==2.9.1

uv pip install --python "$VENV_DIR/bin/python" \
  transformers==4.57.5 hf_transfer==0.1.9 accelerate==1.12.0 \
  tiktoken==0.12.0 blobfile==3.1.0 kernels==0.11.5 \
  compressed-tensors==0.13.0 \
  plotly pandas numpy kaleido python-dotenv pyyaml tqdm termcolor requests \
  datasets openai

# scikit-learn is imported by the demo notebooks but is absent from setup_python.sh,
# where it arrives preinstalled on the Kaggle/RunPod images.
uv pip install --python "$VENV_DIR/bin/python" scikit-learn


# ---------- 4. Setup Jupyter ----------
uv pip install --python "$VENV_DIR/bin/python" jupyterlab jupyter_server ipykernel ipywidgets nbformat notebook

# Registering the kernel writes to ~/Library/Jupyter, which a sandboxed shell may not be
# allowed to touch. Not fatal: an IDE can run notebooks against the venv interpreter
# directly without a globally registered kernelspec.
if ! "$VENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Role analysis (local, CPU)"; then
  echo "WARNING: could not register the Jupyter kernelspec (permissions)." >&2
  echo "         Select $VENV_DIR/bin/python as the notebook interpreter instead." >&2
fi


# ---------- 5. Add import paths ----------
# Lets `import utils...` and `from demo.simple_test_helpers import ...` resolve from
# anywhere, matching the remote environment.
SITE_DIR="$("$VENV_DIR/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf "%s\n" "$PROJECT_DIR" > "$SITE_DIR/add_path_analysis.pth"


echo
echo "Done. Kernel: $KERNEL_NAME  |  Python: $("$VENV_DIR/bin/python" -V)"
echo "Interpreter for your IDE: $VENV_DIR/bin/python"
