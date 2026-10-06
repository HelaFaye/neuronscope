"""Device inventory and placement across CUDA, ROCm, Vulkan, Metal and CPU,
from recorded tool output (no hardware needed)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import accelerators as acc  # noqa: E402

GIB = acc.GIB
RAM = (32 * GIB, 20 * GIB)

VK_APU = """==========
VULKANINFO
==========
Devices:
========
GPU0:
\tapiVersion         = 1.3.274
\tvendorID           = 0x1002
\tdeviceID           = 0x1681
\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\tdeviceName         = AMD Radeon 680M (RADV REMBRANDT)
\tdriverName         = radv
GPU1:
\tvendorID           = 0x10005
\tdeviceType         = PHYSICAL_DEVICE_TYPE_CPU
\tdeviceName         = llvmpipe (LLVM 17.0.6, 256 bits)
\tdriverName         = llvmpipe
"""

ROCMINFO = """*******
Agent 1
*******
  Name:                    AMD Ryzen 9
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1100
  Device Type:             GPU
"""


def card(tmp, n, vendor, vram, gtt, slot, name=None):
    d = tmp / f"card{n}" / "device"
    d.mkdir(parents=True)
    (d / "vendor").write_text(vendor + "\n")
    (d / "mem_info_vram_total").write_text(str(vram))
    (d / "mem_info_vram_used").write_text(str(vram // 8))
    (d / "mem_info_gtt_total").write_text(str(gtt))
    (d / "mem_info_gtt_used").write_text("0")
    (d / "uevent").write_text(f"DRIVER=amdgpu\nPCI_SLOT_NAME={slot}\n")
    if name:
        (d / "product_name").write_text(name)
    (tmp / f"card{n}-eDP-1").mkdir()            # a connector, not a card


def test_parsers(tmp_path):
    card(tmp_path, 0, "0x1002", 512 * 2**20, 16 * GIB, "0000:04:00.0")
    cards = acc.drm_cards(str(tmp_path))
    assert len(cards) == 1 and cards[0]["vendor"] == "amd" and cards[0]["pci"] == "0000:04:00.0"
    vk = acc.parse_vulkan_summary(VK_APU)
    assert vk[0] == {"index": 0, "vendor": "amd", "type": "integrated_gpu",
                     "name": "AMD Radeon 680M (RADV REMBRANDT)", "driver": "radv"}
    assert vk[1]["type"] == "cpu"
    assert acc.parse_rocminfo(ROCMINFO) == ["gfx1100"]


def test_amd_apu_laptop_without_rocm_uses_vulkan_with_gtt(tmp_path):
    card(tmp_path, 0, "0x1002", 512 * 2**20, 16 * GIB, "0000:04:00.0")
    devs = acc.detect({}, smi=[], drm=acc.drm_cards(str(tmp_path)), vulkan=acc.parse_vulkan_summary(VK_APU),
                      rocm=[], system=("Linux", "x86_64"), ram=RAM)
    assert [d["id"] for d in devs] == ["vulkan:0", "cpu:0"]          # llvmpipe is not a GPU
    apu = devs[0]
    assert apu["unified"] and apu["enabled"] and apu["memory_total"] == 512 * 2**20 + 16 * GIB
    pl = acc.place(devs, 6 * GIB)
    assert pl["devices"][0]["id"] == "vulkan:0" and pl["ngl"] == 99
    assert acc.launch_env(pl["devices"]) == {"GGML_VK_VISIBLE_DEVICES": "0"}
    assert acc.place(devs, 40 * GIB) is None                        # neither APU nor RAM can hold it


def test_rocm_desktop_disables_the_vulkan_twin_and_amd_comes_first(tmp_path):
    card(tmp_path, 1, "0x1002", 24 * GIB, 16 * GIB, "0000:03:00.0", "Radeon RX 7900 XTX")
    vk = [{"index": 0, "vendor": "amd", "type": "discrete_gpu", "name": "AMD Radeon RX 7900 XTX (RADV NAVI31)"},
          {"index": 1, "vendor": "nvidia", "type": "discrete_gpu", "name": "NVIDIA GeForce RTX 3060"}]
    smi = [{"index": 0, "name": "NVIDIA GeForce RTX 3060", "memory_total": 12 * GIB, "memory_free": 11 * GIB,
            "bus_id": "00000000:0A:00.0"}]
    cfg = {"servers": {"rocm": "~/llama-rocm/llama-server"},
           "devices": {"rocm:0": {"env": {"HSA_OVERRIDE_GFX_VERSION": "11.0.0", "LD_PRELOAD": "/evil.so"},
                                  "settings": {"threads": 8, "rm": "-rf"}}}}
    devs = {d["id"]: d for d in acc.detect(cfg, smi=smi, drm=acc.drm_cards(str(tmp_path)), vulkan=vk,
                                           rocm=["gfx1100"], system=("Linux", "x86_64"), ram=RAM)}
    assert devs["rocm:0"]["name"] == "Radeon RX 7900 XTX" and not devs["rocm:0"]["unified"]
    assert devs["vulkan:0"]["twin_of"] == "rocm:0" and not devs["vulkan:0"]["enabled"]
    assert devs["vulkan:1"]["twin_of"] == "cuda:0" and not devs["vulkan:1"]["enabled"]
    assert devs["rocm:0"]["env"] == {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"}       # LD_PRELOAD refused
    assert devs["rocm:0"]["settings"] == {"threads": 8}
    assert devs["rocm:0"]["server"].endswith("llama-rocm/llama-server") and "~" not in devs["rocm:0"]["server"]
    lst = list(devs.values())
    # Fits on both: AMD first, even though the NVIDIA card is the tighter fit.
    assert acc.place(lst, 8 * GIB)["devices"][0]["id"] == "rocm:0"
    assert acc.place(lst, 8 * GIB, prefer=("nvidia",))["devices"][0]["id"] == "cuda:0"
    # The AMD card is booked: the next model goes to NVIDIA.
    assert acc.place(lst, 8 * GIB, reserved={"rocm:0": 20 * GIB})["devices"][0]["id"] == "cuda:0"
    # A project limited to the CPU stays there.
    assert acc.place(lst, 8 * GIB, allowed=["cpu:0"])["devices"][0]["id"] == "cpu:0"


def test_multi_gpu_split_within_one_backend():
    devs = [{"id": f"rocm:{i}", "backend": "rocm", "index": i, "vendor": "amd", "memory_total": 8 * GIB,
             "memory_free": 8 * GIB, "unified": False, "enabled": True, "reserve": 0} for i in range(3)]
    devs.append({"id": "cpu:0", "backend": "cpu", "index": 0, "vendor": "cpu", "memory_total": 64 * GIB,
                 "memory_free": 64 * GIB, "unified": True, "enabled": True, "reserve": 0})
    pl = acc.place(devs, 20 * GIB)
    assert [d["id"] for d in pl["devices"]] == ["rocm:0", "rocm:1", "rocm:2"] and pl["split"] == [8, 8, 8]
    assert acc.launch_env(pl["devices"])["HIP_VISIBLE_DEVICES"] == "0,1,2"
    assert acc.place(devs, 30 * GIB)["devices"][0]["id"] == "cpu:0"


def test_rocm_apu_gets_unified_memory_env(tmp_path):
    card(tmp_path, 0, "0x1002", 2 * GIB, 24 * GIB, "0000:c1:00.0")
    devs = acc.detect({}, smi=[], drm=acc.drm_cards(str(tmp_path)), vulkan=[], rocm=["gfx1103"],
                      system=("Linux", "x86_64"), ram=RAM)
    apu = devs[0]
    assert apu["id"] == "rocm:0" and apu["unified"] and "APU" in apu["name"]
    assert apu["enabled"] and "HSA_OVERRIDE_GFX_VERSION=11.0.0" in apu["note"]
    assert acc.launch_env([apu]) == {"HIP_VISIBLE_DEVICES": "0", "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1",
                                     "HSA_OVERRIDE_GFX_VERSION": "11.0.0"}


def test_metal_and_manual_devices():
    devs = acc.detect({"manual": [{"id": "vulkan:5", "backend": "vulkan", "index": 5, "name": "eGPU",
                                   "memory_total_gib": 8}],
                       "devices": {"metal:0": {"memory_total_gib": 40}}},
                      smi=[], drm=[], vulkan=[], rocm=[], system=("Darwin", "arm64"), ram=(64 * GIB, 48 * GIB))
    ids = {d["id"]: d for d in devs}
    assert ids["metal:0"]["memory_total"] == 40 * GIB and ids["metal:0"]["unified"]
    assert ids["vulkan:5"]["memory_total"] == 8 * GIB
    assert acc.launch_env([ids["metal:0"]]) == {}


VK_VEGA = [{"index": 0, "vendor": "amd", "type": "integrated_gpu", "name": "AMD Radeon Graphics (RADV RENOIR)"}]


def vega_laptop(tmp_path, cfg=None):
    """A Ryzen 7030U-class laptop: Vega 8 iGPU (gfx90c), 2 GiB UMA carve-out,
    16 GiB GTT, 32 GiB of DDR4, ROCm installed but not supporting the chip."""
    card(tmp_path, 0, "0x1002", 2 * GIB, 16 * GIB, "0000:04:00.0")
    return {d["id"]: d for d in acc.detect(cfg or {}, smi=[], drm=acc.drm_cards(str(tmp_path)), vulkan=VK_VEGA,
                                           rocm=["gfx90c"], system=("Linux", "x86_64"), ram=(32 * GIB, 24 * GIB))}


def test_vega_apu_defaults_to_vulkan_and_rocm_is_opt_in(tmp_path):
    devs = vega_laptop(tmp_path)
    assert not devs["rocm:0"]["enabled"] and "not supported by ROCm" in devs["rocm:0"]["note"]
    assert devs["vulkan:0"]["enabled"] and devs["vulkan:0"]["twin_of"] == "rocm:0"
    assert devs["vulkan:0"]["unified"] and devs["vulkan:0"]["memory_total"] == 18 * GIB
    pl = acc.place(list(devs.values()), 6 * GIB)
    assert pl["devices"][0]["id"] == "vulkan:0"
    assert acc.launch_env(pl["devices"]) == {"GGML_VK_VISIBLE_DEVICES": "0"}


def test_vega_apu_with_a_pinned_rocm_release(tmp_path):
    rp = tmp_path / "rocm-6.2.4"
    (rp / "lib").mkdir(parents=True)
    cfg = {"devices": {"rocm:0": {"enabled": True, "rocm_path": str(rp),
                                  "server": str(tmp_path / "llama-rocm62" / "llama-server")}}}
    devs = vega_laptop(tmp_path / "sys", cfg)
    r = devs["rocm:0"]
    assert r["enabled"] and not devs["vulkan:0"]["enabled"]          # the twin steps aside
    assert r["server"].endswith("llama-rocm62/llama-server")
    env = acc.launch_env([r])
    assert env["HSA_OVERRIDE_GFX_VERSION"] == "9.0.0" and env["ROCM_PATH"] == str(rp) == env["HIP_PATH"]
    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(rp / "lib")
    assert env["HIP_VISIBLE_DEVICES"] == "0" and env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] == "1"
    gone = vega_laptop(tmp_path / "sys2", {"devices": {"rocm:0": {"enabled": True, "rocm_path": "/nope"}}})
    assert not gone["rocm:0"]["enabled"] and "does not exist" in gone["rocm:0"]["note"]


def test_apu_and_cpu_share_one_memory_pool(tmp_path):
    devs = list(vega_laptop(tmp_path).values())
    # 24 GiB of RAM free, 1 GiB kept back on each: a 12 GiB model on the iGPU
    # leaves ~11 GiB, so a second 12 GiB model fits nowhere, not on the CPU either.
    first = acc.place(devs, 12 * GIB)
    assert first["devices"][0]["id"] == "vulkan:0"
    assert acc.place(devs, 12 * GIB, reserved={"vulkan:0": 12 * GIB}) is None
    assert acc.place(devs, 8 * GIB, reserved={"vulkan:0": 12 * GIB})["devices"][0]["id"] == "cpu:0"


def test_m10_on_arm_board_splits_over_measured_gpus_before_the_mali(tmp_path):
    """Tesla M10 (4 x 8 GiB Maxwell) on an RK3588 board with a Mali GPU and the
    non-coherent driver patch: a model too big for one M10 is split over two,
    not put on the Mali whose memory is only an estimate."""
    params = tmp_path / "params"
    params.mkdir()
    (params / "arm_force_uncached").write_text("1\n")
    smi = [{"index": i, "name": "Tesla M10", "memory_total": 8 * GIB, "memory_free": int(7.9 * GIB),
            "bus_id": f"00000000:0{4 + i}:00.0"} for i in range(4)]
    vk = [{"index": 0, "vendor": "arm", "type": "integrated_gpu", "name": "Mali-G610 MC4"}] + \
         [{"index": i + 1, "vendor": "nvidia", "type": "discrete_gpu", "name": "Tesla M10"} for i in range(4)]
    devs = acc.detect({}, smi=smi, drm=[], vulkan=vk, rocm=[], system=("Linux", "aarch64"),
                      ram=(32 * GIB, 28 * GIB), sysfs=str(params))
    ids = {d["id"]: d for d in devs}
    assert [ids[f"vulkan:{i}"]["twin_of"] for i in range(1, 5)] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert ids["vulkan:0"]["vendor"] == "arm" and ids["vulkan:0"]["estimated"]
    assert "GGML_CUDA_NO_PINNED" in ids["cuda:0"]["note"]
    pl = acc.place(devs, 10 * GIB)
    assert [d["id"] for d in pl["devices"]] == ["cuda:0", "cuda:1"] and pl["split"] == [7, 7]
    assert acc.place(devs, 5 * GIB)["devices"][0]["id"] == "cuda:0"
    # With every M10 booked, a small model may still use the Mali.
    full = {f"cuda:{i}": 7 * GIB for i in range(4)}
    assert acc.place(devs, 4 * GIB, reserved=full)["devices"][0]["id"] == "vulkan:0"
