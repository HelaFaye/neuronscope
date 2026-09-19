#!/usr/bin/env bash
# Installer for NeuronScope.
#
# Targets EndeavourOS / Arch, where AMD does not officially support ROCm. The
# strategy is to try the PyTorch ROCm wheels first (they bundle the ROCm
# userspace, so all they need from the host is the amdgpu/amdkfd kernel driver,
# which Arch ships in mainline), and fall back to Docker if that fails.
#
#   ./install.sh              # detect, install into ./venv, verify
#   ./install.sh --rocm 6.4   # force a ROCm wheel index
#   ./install.sh --cpu        # no GPU, CPU-only torch
#   ./install.sh --docker     # just print the Docker recipe and exit

set -euo pipefail

ROCM_VERSION=""
MODE="auto"
VENV="${VENV:-venv}"

FORCE_ROCM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rocm) ROCM_VERSION="$2"; FORCE_ROCM=1; shift 2 ;;
    --cpu) MODE="cpu"; shift ;;
    --system-torch) MODE="system"; shift ;;
    --docker) MODE="docker"; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!! \033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx \033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- environment

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  info "distro: ${PRETTY_NAME:-$ID}"
  case "${ID_LIKE:-$ID}" in
    *arch*|arch) warn "Arch is not an AMD-supported ROCm distro. The pip wheels" ;
                 warn "usually work anyway; ./install.sh --docker if they don't." ;;
  esac
fi

command -v python3 >/dev/null || die "python3 not found"
PYVER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
info "python: $PYVER"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
  || die "python 3.10+ required"

# ------------------------------------------------------------ docker shortcut

if [[ "$MODE" == "docker" ]]; then
  cat <<'DOCKER'

Docker fallback. Kernel driver comes from your host, all of ROCm userspace
comes from the container, so Arch's lack of official support stops mattering.

  docker run -it --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ipc=host --shm-size 16G \
    -v "$PWD":/work -w /work \
    rocm/pytorch:latest bash

  # then, inside:
  pip install -r requirements.txt
  python scripts/preflight.py --model_path <model>

DOCKER
  exit 0
fi

# ------------------------------------------------------------ GPU detection

GFX=""
if [[ "$MODE" != "cpu" ]]; then
  if [[ ! -e /dev/kfd ]]; then
    warn "/dev/kfd missing. The amdgpu kernel driver is not loaded, or you are"
    warn "in a container without --device=/dev/kfd. Falling back to CPU torch."
    MODE="cpu"
  else
    for grp in video render; do
      id -nG | tr ' ' '\n' | grep -qx "$grp" \
        || warn "you are not in the '$grp' group: sudo usermod -aG $grp \$USER (re-login)"
    done
    if command -v rocminfo >/dev/null; then
      # Read the real target: HSA_OVERRIDE_GFX_VERSION changes what rocminfo
      # reports, so leaving it set detects the masquerade instead. The {3,}
      # also stops a truncated match like a bare "gfx9".
      GFX=$(env -u HSA_OVERRIDE_GFX_VERSION rocminfo 2>/dev/null \
            | grep -oP 'gfx[0-9a-f]{3,}' | sort -u | grep -v gfx000 | head -1 || true)
    fi
    if [[ -n "$GFX" ]]; then
      info "detected: $GFX"
    else
      warn "could not detect gfx target (rocminfo not installed?)"
      if command -v lspci >/dev/null && lspci 2>/dev/null | grep -qiE "vega|cezanne|barcelo|renoir|picasso|raven"; then
        warn "lspci suggests a Vega-class integrated GPU, which ROCm does not"
        warn "support. Installing CPU torch; use the Vulkan path for extraction."
        MODE="cpu"
      fi
    fi
  fi
fi

# Map architecture to guidance. Only the wheel index is decided here; the
# override advice is printed for the user to apply if the smoke test fails.
OVERRIDE=""
case "$GFX" in
  gfx1200|gfx1201)
    info "RDNA4. Needs ROCm 7.x; older wheels will not have kernels for this."
    [[ -z "$ROCM_VERSION" ]] && ROCM_VERSION="7.0" ;;
  gfx1100|gfx1101|gfx1102)
    info "RDNA3, officially supported."
    [[ -z "$ROCM_VERSION" ]] && ROCM_VERSION="6.4" ;;
  gfx1030|gfx1031|gfx1032|gfx1034|gfx1035|gfx1036)
    info "RDNA2. gfx1030 is supported; the smaller dies need an override."
    [[ "$GFX" != "gfx1030" ]] && OVERRIDE="10.3.0"
    [[ -z "$ROCM_VERSION" ]] && ROCM_VERSION="6.4" ;;
  gfx1010|gfx1011|gfx1012)
    warn "RDNA1 is not supported by ROCm. Community builds exist but expect"
    warn "to compile PyTorch yourself. CPU extraction may be the faster path."
    [[ -z "$ROCM_VERSION" ]] && ROCM_VERSION="6.4" ;;
  gfx90c|gfx902|gfx909|gfx90b)
    if (( FORCE_ROCM )); then
      # The gfx900 masquerade does work on some Vega APUs -- verify with
      # scripts/try_rocm.sh, which checks results against CPU rather than
      # just checking that a matmul returned.
      info "Vega APU ($GFX) with --rocm given: installing ROCm torch."
      info "Export HSA_OVERRIDE_GFX_VERSION=9.0.0 before using it."
    else
      warn "Vega-based APU ($GFX). ROCm has no support for this target."
      warn "Installing CPU torch, which is correct for the main pipeline:"
      warn "cett-dump runs on Vulkan and a 9B in bf16 would not fit anyway."
      warn ""
      warn "The gfx900 masquerade does work on some of these. To try it:"
      warn "  ./scripts/try_rocm.sh          then  ./install.sh --rocm 6.4"
      MODE="cpu"
    fi ;;
  gfx90a|gfx942|gfx908) info "CDNA, fully supported."
    [[ -z "$ROCM_VERSION" ]] && ROCM_VERSION="6.4" ;;
  "") [[ -z "$ROCM_VERSION" && "$MODE" != "cpu" ]] && ROCM_VERSION="6.4" ;;
  *) warn "unrecognised target $GFX; defaulting the wheel index"
     [[ -z "$ROCM_VERSION" ]] && ROCM_VERSION="6.4" ;;
esac

# ---------------------------------------------------------------- install

if [[ "$MODE" == "system" ]]; then
  # Use the distro's torch instead of a wheel. On Arch, python-pytorch-rocm is
  # built against the same ROCm as /opt/rocm, so its rocBLAS and its Tensile
  # kernels match -- no ABI mixing across major versions.
  if ! python3 -c 'import torch' 2>/dev/null; then
    die "no system torch found. Install it first:
   sudo pacman -S python-pytorch-rocm"
  fi
  info "creating venv at $VENV with access to system packages"
  python3 -m venv --system-site-packages "$VENV"
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  pip install --quiet --upgrade pip wheel
  info "using system torch: $(python -c 'import torch;print(torch.__version__)')"
  info "installing the rest"
  pip install --quiet -r requirements.txt
  info "verifying"
  # fall through to the smoke test below
  SKIP_TORCH_INSTALL=1
fi

if [[ -z "${SKIP_TORCH_INSTALL:-}" ]]; then
info "creating venv at $VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
pip install --quiet --upgrade pip wheel

if [[ "$MODE" == "cpu" ]]; then
  INDEX="https://download.pytorch.org/whl/cpu"
  info "installing CPU torch"
  info "  (re-run with --rocm 6.4 if scripts/try_rocm.sh passes on your GPU)"
else
  INDEX="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"
  info "installing torch from $INDEX"
fi

if ! pip install torch --index-url "$INDEX"; then
  warn "that wheel index failed. Available ROCm indexes are listed at"
  warn "  https://pytorch.org/get-started/locally/"
  warn "Retry with ./install.sh --rocm <version>, or use ./install.sh --docker"
  exit 1
fi

info "installing the rest"
pip install --quiet -r requirements.txt
fi

# ---------------------------------------------------------------- smoke test

info "verifying"
cat > /tmp/ns_smoke.py <<'PY'
import torch
print("torch     :", torch.__version__)
print("hip       :", getattr(torch.version, "hip", None))
if torch.cuda.is_available():
    print("device    :", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print("vram      : %.1f GiB total, %.1f GiB free" % (total/2**30, free/2**30))
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    print("bf16 matmul:", tuple((x @ x).shape), "ok")
    print("\nsuggested --gpu_mem: %dGiB" % max(1, int(total/2**30) - 2))
else:
    print("no GPU visible to torch; extraction will run on CPU")
PY

if python /tmp/ns_smoke.py; then
  :
else
  warn "smoke test failed."
  if [[ -n "$OVERRIDE" ]]; then
    warn "Your $GFX needs an architecture override. Try:"
    warn "  export HSA_OVERRIDE_GFX_VERSION=$OVERRIDE"
    warn "and re-run. Add it to ~/.bashrc if it works."
  else
    warn "Try ./install.sh --docker, or ./install.sh --cpu to proceed without GPU."
  fi
  exit 1
fi
rm -f /tmp/ns_smoke.py

cat <<EOF

Done. Activate with:  source $VENV/bin/activate

Next step, which costs no VRAM and downloads no weights:

  python scripts/preflight.py --model_path ornith-ai/Ornith-1.0-9B --n_pairs 400

Then read PIPELINE in README.md.
EOF
