#!/usr/bin/env bash
# Source this before running anything local:  source env.sh
#
# The model lives on an auto-mounted volume, so the path under /run/media can
# change between reboots or if the drive is reattached. Everything references
# $NS_GGUF rather than embedding the path, and the check below fails loudly
# rather than letting a stage run against a missing file.

# Resolve by filesystem UUID so a different mount point still works.
NS_VOLUME_UUID="19c7eeb5-52f1-4c37-8483-e37745655fc7"
NS_MODEL_REL=".models/lmstudio-community/Ornith-1.0-9B-GGUF/Ornith-1.0-9B-Q6_K.gguf"

if [[ -e "/dev/disk/by-uuid/${NS_VOLUME_UUID}" ]]; then
    _dev=$(readlink -f "/dev/disk/by-uuid/${NS_VOLUME_UUID}")
    _mnt=$(findmnt -n -o TARGET --source "${_dev}" 2>/dev/null | head -1)
else
    _mnt=""
fi
# Fall back to the path as it stood when this was written.
: "${_mnt:=/run/media/hela/${NS_VOLUME_UUID}}"

export NS_GGUF="${_mnt}/${NS_MODEL_REL}"
export NS_TOKENIZER="ornith-ai/Ornith-1.0-9B"
export NS_LLAMA="${HOME}/llama.cpp"
export NS_CETT="${NS_LLAMA}/build/bin/llama-cett-dump"

# Ornith is 32 decoder layers; confirm with preflight.py before a real run.
# AMDVLK Pro is installed and failing to initialise (-3) on this box. RADV
# handles it fine, but llama.cpp and wgpu enumerate ICDs themselves and may not
# skip as gracefully, so pin RADV explicitly.
_radv=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
[[ -r "$_radv" ]] && export VK_DRIVER_FILES="$_radv"
unset _radv

export NS_LAYERS=32
# Must exceed the longest sequence: cett-dump needs one decode per sample.
export NS_BATCH=4096

if [[ ! -r "${NS_GGUF}" ]]; then
    echo "!! model not readable at:" >&2
    echo "   ${NS_GGUF}" >&2
    if [[ -z "${_mnt}" || ! -d "${_mnt}" ]]; then
        echo "   the volume does not appear to be mounted." >&2
        echo "   udisksctl mount -b /dev/disk/by-uuid/${NS_VOLUME_UUID}" >&2
    fi
else
    _sz=$(stat -c %s "${NS_GGUF}")
    echo "model : ${NS_GGUF}"
    echo "size  : $(( _sz / 1024 / 1024 )) MB"
    # An 8.3GB file read over USB on every load is worth avoiding. It fits in
    # page cache on a 32GB machine, so only the first load is slow -- but if
    # the drive is external and slow, copying to internal storage once is
    # cheaper than discovering that mid-run.
    _src=$(findmnt -n -o SOURCE --target "${NS_GGUF}" 2>/dev/null)
    case "${_src}" in
        /dev/sd*|/dev/nvme*) : ;;
        *) echo "note  : unusual backing device ${_src}; check read speed" >&2 ;;
    esac
fi
unset _dev _mnt _sz _src
