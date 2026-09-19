#!/usr/bin/env python3
"""Export a trained classifier for NS_CLASSIFIER.

Flat little-endian float32: [n_layers * n_ff] coefficients, then one intercept.
Deliberately dumb -- the server should not need numpy or a parser to read it.

    python llama-tools/server-activations/export_classifier_bin.py \
        models/classifier.npz models/classifier.bin
"""
import sys
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit(__doc__)
src, dst = sys.argv[1], sys.argv[2]
blob = np.load(src)
coef = np.asarray(blob["coef"], dtype="<f4").ravel()
intercept = np.float32(float(blob["intercept"]))
with open(dst, "wb") as f:
    f.write(coef.tobytes())
    f.write(np.asarray([intercept], dtype="<f4").tobytes())
n_layers = int(blob["n_layers"]) if "n_layers" in blob.files else None
print(f"wrote {dst}: {coef.size} weights + intercept "
      f"({coef.nbytes + 4} bytes)")
if n_layers:
    print(f"  expects a model with {n_layers} layers x "
          f"{coef.size // n_layers} neurons")
print("  the server checks this length against the model and refuses on a "
      "mismatch")
