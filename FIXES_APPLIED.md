# NeuronScope fixes applied — 2026-10-02

This patch set is deliberately limited to stabilizing and correcting the existing NeuronScope research/intervention project. It does **not** add a coding-model training pipeline.

## Fixes

- Split lightweight shared constants/token normalization into `scripts/ns_constants.py` so `extract_activations_gguf.py` can run without importing PyTorch or Transformers.
- Added `scripts/gguf_utils.py` as the single compatibility path for GGUF scalar/string metadata fields. This fixes the prior first-byte/string-field failure and updates the affected utility/visualizer consumers.
- Removed dead/unreachable fallback code from `scripts/gguf_tokenizer.py`.
- Made `tests/test_audit.py` repository-relative instead of hard-coding `/home/claude/neuronscope`.
- Added regression coverage for negative neuron indices, duplicate neuron indices, and out-of-range layer indices.
- Added shared selection validation to profiles and the direct intervention/merge consumers so malformed selections fail before tensor indexing.
- Added profile required-field and geometry checks during load/application.
- Added a classifier train/test QID-overlap guard to prevent accidental leakage.
- Fixed the GGUF extractor's per-sample prompt fallback bug: the fallback previously used the final sample's prompt instead of the current sample's prompt.
- Made GGUF tokenizer/activation subprocess failures fatal instead of silently continuing with incomplete intermediate data.
- Added `requests` and `jinja2` to `requirements.txt`, matching the code paths already treated as core by `doctor.py`.
- Added `.env.example` and excluded the uploaded `.env` from the release archive so credentials are not redistributed.
- Kept mergekit's NeuronScope plugin standalone rather than introducing a fragile dependency on the local `profiles` module.

## Verification

Passed:

- `python tests/test_audit.py`
- Python bytecode compilation for `scripts`, `viz`, and `tests`
- `bash -n` for all shell scripts
- `git diff --check`
- `--help` smoke tests for the GGUF extractor, classifier, VRAM utility, doctor, Studio, and weights tools
- focused classifier train/test overlap integration check

The supplied archive already contained unrelated working-tree modifications; the fixed source archive preserves those current files while omitting Git metadata, the `.env` credential file, generated `__pycache__`, and log files.
