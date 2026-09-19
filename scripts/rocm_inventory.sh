#!/usr/bin/env bash
# What ROCm is on this machine, and which matrix-multiply kernels it has.
#
# There are usually two rocBLAS installs: one from your distro under /opt/rocm,
# and one bundled inside the PyTorch wheel. They ship different GPU targets,
# and PyTorch uses its own by default. If the system one has kernels the wheel
# lacks, ROCBLAS_TENSILE_LIBPATH can point PyTorch at it -- which is cheaper
# than downgrading the wheel.

set -uo pipefail
say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!! \033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok \033[0m %s\n' "$*"; }

targets_in() {   # directory -> sorted unique gfx targets
  # ROCm 7.x ships generic targets like gfx9-generic that cover a whole family,
  # so the pattern has to allow letters and hyphens, not just hex.
  ls "$1" 2>/dev/null | grep -oP 'gfx[0-9a-f]+(-generic)?' | sort -u | tr '\n' ' '
}

covers() {       # "available list" "needed target" -> 0 if covered
  local avail="$1" want="$2"
  grep -qw "$want" <<< "$avail" && return 0
  # gfx90c, gfx900, gfx906... are all gfx9; a gfx9-generic kernel serves them.
  local fam="gfx${want#gfx}"; fam="gfx${fam:3:1}"
  grep -qw "${fam}-generic" <<< "$avail" && return 0
  return 1
}

say "hardware"
REAL=$(env -u HSA_OVERRIDE_GFX_VERSION rocminfo 2>/dev/null \
       | grep -oP 'gfx[0-9a-f]{3,}' | sort -u | grep -v gfx000 | head -1)
printf '     real target      : %s\n' "${REAL:-unknown (install rocminfo)}"
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  printf '     override active  : %s -> gfx%s\n' \
    "$HSA_OVERRIDE_GFX_VERSION" "$(tr -d . <<< "$HSA_OVERRIDE_GFX_VERSION")"
fi
[[ -e /dev/kfd ]] && ok "/dev/kfd present" || warn "/dev/kfd missing: no compute"

say "system ROCm"
if [[ -d /opt/rocm ]]; then
  VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)
  printf '     /opt/rocm        : %s\n' "$VER"
  if command -v pacman >/dev/null; then
    n=$(pacman -Qq 2>/dev/null | grep -c '^rocm\|hip\|rocblas' || true)
    printf '     pacman packages  : %s installed (pacman -Qs rocm to list)\n' "$n"
  fi
  SYS_LIB=""
  for d in /opt/rocm/lib/rocblas/library /opt/rocm/lib/librocblas/library; do
    [[ -d "$d" ]] && SYS_LIB="$d" && break
  done
  if [[ -n "$SYS_LIB" ]]; then
    SYS_T=$(targets_in "$SYS_LIB")
    printf '     rocBLAS kernels  : %s\n' "${SYS_T:-none}"
  else
    warn "no system rocBLAS library dir found"
  fi
else
  printf '     /opt/rocm        : not installed\n'
fi

say "PyTorch"
TORCH_ERR=$(python - <<'PY' 2>&1 >/dev/null
import torch
PY
)
TORCH_LIB=$(python - <<'PY' 2>/dev/null
import os
try:
    import torch
except BaseException:
    raise SystemExit
print(torch.__version__)
print(getattr(torch.version, "hip", "") or "")
print(os.path.join(os.path.dirname(torch.__file__), "lib", "rocblas", "library"))
PY
)
if [[ -z "$TORCH_LIB" ]]; then
  warn "torch not importable in this python. The error was:"
  printf '%s\n' "${TORCH_ERR:-(no output)}" | tail -6 | sed 's/^/     /'
  echo
  warn "Common causes with a --system-site-packages venv:"
  warn "  - the venv was built from a different python than the one that owns"
  warn "    the distro torch. Compare:"
  warn "      python -c 'import sys; print(sys.version, sys.path)'"
  warn "      pacman -Ql python-pytorch-rocm | grep -m1 site-packages/torch/__init__"
  warn "  - a package installed into the venv shadows a system one"
  exit 0
fi
mapfile -t T <<< "$TORCH_LIB"
printf '     version          : %s\n' "${T[0]}"
printf '     hip              : %s\n' "${T[1]:-none (this is a CPU build)}"
WHEEL_LIB="${T[2]}"
if [[ -d "$WHEEL_LIB" ]]; then
  WHEEL_T=$(targets_in "$WHEEL_LIB")
  printf '     rocBLAS kernels  : %s\n' "${WHEEL_T:-none}"
fi

say "verdict"
TARGET="gfx$(tr -d . <<< "${HSA_OVERRIDE_GFX_VERSION:-}")"
[[ "$TARGET" == "gfx" ]] && TARGET="$REAL"
printf '     need kernels for : %s\n' "${TARGET:-unknown}"

have_wheel=0; have_sys=0
[[ -n "${WHEEL_T:-}" ]] && covers "$WHEEL_T" "$TARGET" && have_wheel=1
[[ -n "${SYS_T:-}"   ]] && covers "$SYS_T"   "$TARGET" && have_sys=1
# If the system covers the REAL target generically, the override is unnecessary.
if [[ -n "${SYS_T:-}" && -n "${REAL:-}" ]] && covers "$SYS_T" "$REAL"; then
  ok "system rocBLAS covers $REAL directly -- you may not need the override"
  echo "        unset HSA_OVERRIDE_GFX_VERSION && ./scripts/try_rocm.sh"
fi

if (( have_wheel )); then
  ok "the wheel has $TARGET -- PyTorch should work as installed"
elif (( have_sys )); then
  warn "the wheel lacks $TARGET but your system rocBLAS has it. Try:"
  echo
  echo "  export ROCBLAS_TENSILE_LIBPATH=$SYS_LIB"
  echo "  ./scripts/try_rocm.sh"
  echo
  warn "Mixing a wheel's rocBLAS with the system Tensile library is not"
  warn "supported by AMD. try_rocm.sh checks results against CPU, so run it"
  warn "before trusting any number that comes out."
else
  warn "neither has kernels for $TARGET."
  echo
  echo "  ./install.sh --rocm 6.0      # older wheels still shipped gfx900"
  echo "                               # this only changes the venv; the pip"
  echo "                               # wheel bundles its own ROCm libs and"
  echo "                               # never touches /opt/rocm"
  echo
  warn "Do not remove system ROCm to 'downgrade': the wheel's ROCm version is"
  warn "independent of it, and rocsolver, rccl, magma-hip and others depend on"
  warn "what you have."
  echo
  warn "Nothing in the pipeline needs this: cett-dump runs on Vulkan, and a"
  warn "9B in bf16 will not fit a 12GB carve-out even if ROCm works."
fi
