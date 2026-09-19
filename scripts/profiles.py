"""
Suppression profiles: save a tuned H-Neuron intervention and re-apply it.

A profile is small (a few hundred integers and a float) and is keyed to a model
fingerprint, so applying one to the wrong model fails loudly instead of quietly
scaling arbitrary neurons.

The intervention itself is applied as a forward pre-hook rather than a weight
edit. Scaling column j of down_proj is identical to scaling input element j
before the matmul:

    W @ (s * a) = sum_j W[:, j] * s_j * a_j = (W @ diag(s)) @ a

So a hook gives the same result with no weight mutation, instant toggle, and a
scale you can change at runtime. Baking becomes an export step, not the
mechanism.

Layout:

    profiles/
      <fingerprint>/
        <config_name>.json
"""

import hashlib
import json
import os
import time

import torch

from ns_common import TEXT_LAYER_RE, text_config

PROFILE_VERSION = 1


def fingerprint(config):
    """Stable id for a model's architecture.

    Deliberately excludes the model name: a fine-tune with identical geometry
    is a different model and neuron indices will not transfer, so the name is
    recorded separately as provenance and checked as a warning, not a gate.
    The geometry check is the hard gate.
    """
    t = text_config(config)
    parts = [
        str(getattr(config, "model_type", "?")),
        str(t.num_hidden_layers),
        str(t.hidden_size),
        str(t.intermediate_size),
        str(getattr(t, "num_attention_heads", "?")),
        str(getattr(t, "vocab_size", "?")),
    ]
    blob = "|".join(parts)
    return hashlib.sha256(blob.encode()).hexdigest()[:16], blob


class Profile:
    def __init__(self, data):
        self.data = data

    # ---------------------------------------------------------------- build

    @classmethod
    def create(cls, fp, geometry, model_name, by_layer, scale,
               n_layers, n_neurons, config_name="default", provenance=None):
        by_layer = {str(k): sorted(int(n) for n in v)
                    for k, v in by_layer.items() if v}
        total = sum(len(v) for v in by_layer.values())
        return cls({
            "version": PROFILE_VERSION,
            "config_name": config_name,
            "fingerprint": fp,
            "geometry": geometry,
            "model_name": model_name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "scale": scale,
            "n_layers": int(n_layers),
            "n_neurons": int(n_neurons),
            "total_neurons": total,
            "by_layer": by_layer,
            "provenance": provenance or {},
            "evaluations": [],
        })

    # ----------------------------------------------------------------- io

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        if data.get("version") != PROFILE_VERSION:
            raise ValueError(f"profile version {data.get('version')} not supported")
        return cls(data)

    def save(self, root="profiles"):
        d = os.path.join(root, self.data["fingerprint"])
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{self.data['config_name']}.json")
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
        return path

    @staticmethod
    def find(config, root="profiles", config_name=None):
        """List profiles matching this model's geometry."""
        fp, _ = fingerprint(config)
        d = os.path.join(root, fp)
        if not os.path.isdir(d):
            return []
        out = []
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            if config_name and name != f"{config_name}.json":
                continue
            out.append(os.path.join(d, name))
        return out

    # ------------------------------------------------------------- checks

    def check(self, model, strict_name=False):
        """Refuse to apply a profile whose geometry does not match."""
        fp, geom = fingerprint(model.config)
        if fp != self.data["fingerprint"]:
            raise ValueError(
                "profile does not match this model.\n"
                f"  profile : {self.data['geometry']}\n"
                f"  model   : {geom}\n"
                "Neuron indices are model-specific and do not transfer."
            )
        name = getattr(model.config, "_name_or_path", None)
        if name and self.data.get("model_name") and \
                name != self.data["model_name"]:
            msg = (f"geometry matches but the model differs: profile was tuned "
                   f"on {self.data['model_name']}, this is {name}. A fine-tune "
                   f"with the same shape has different neurons.")
            if strict_name:
                raise ValueError(msg)
            print(f"warning: {msg}")

    @property
    def by_layer(self):
        return {int(k): v for k, v in self.data["by_layer"].items()}

    def add_evaluation(self, record):
        self.data["evaluations"].append(record)


class SuppressionHandle:
    """Applies a profile via forward pre-hooks. Toggle and rescale at will."""

    def __init__(self, model, profile, scale=None):
        profile.check(model)
        self.profile = profile
        self.scale = float(profile.data["scale"] if scale is None else scale)
        self._hooks = []
        self._targets = []

        by_layer = profile.by_layer
        for name, module in model.named_modules():
            if "down_proj" not in name or not isinstance(module, torch.nn.Linear):
                continue
            m = TEXT_LAYER_RE.search(name)
            if not m:
                continue  # never the vision tower
            layer = int(m.group(1))
            if layer not in by_layer:
                continue
            cols = by_layer[layer]
            if max(cols) >= module.in_features:
                raise ValueError(
                    f"{name} has {module.in_features} inputs but the profile "
                    f"references neuron {max(cols)}"
                )
            # Device is resolved inside the hook: with device_map offload the
            # activation may not arrive on the device the weight was on at
            # construction time.
            idx = torch.tensor(cols, dtype=torch.long)
            self._targets.append((layer, idx))
            self._hooks.append(
                module.register_forward_pre_hook(self._make_hook(idx))
            )

        missing = set(by_layer) - {l for l, _ in self._targets}
        if missing:
            raise ValueError(f"profile layers never matched a module: {missing}")

    def _make_hook(self, idx):
        def pre(module, args):
            if self.scale == 1.0:
                return None  # no-op, cheapest path when disabled
            x = args[0].clone()
            x[..., idx.to(x.device)] *= self.scale
            return (x,) + tuple(args[1:])
        return pre

    def set_scale(self, scale):
        """Change suppression strength with no reload."""
        self.scale = float(scale)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    def summary(self):
        n = sum(len(i) for _, i in self._targets)
        return (f"{n} neurons across {len(self._targets)} layers "
                f"at scale {self.scale}")
