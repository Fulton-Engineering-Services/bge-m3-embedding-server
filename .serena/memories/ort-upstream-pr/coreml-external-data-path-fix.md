# Upstream ORT PR: CoreML external data path fix

## Bug

`TensorProtoWithExternalDataToTensorProto` in
`onnxruntime/core/framework/tensorprotoutils.cc` passes `model_path` (a
model **file** path, e.g. `.../onnx/model.onnx`) directly to
`ReadExternalDataForTensor`, which expects a **directory**.  This causes
`GetExternalDataInfo` to construct `.../onnx/model.onnx/model.onnx_data`,
and the subsequent `open(2)` fails with `ENOTDIR`.

Triggered by CoreML EP's `RegisterInitializers` in `model_builder.cc`,
which passes `graph_viewer_.ModelPath()` — a file path, not a directory.

Confirmed present in **both** v1.23.2 (line 291) **and** `main` (line 320).

## Fix (3 lines of logic)

```cpp
const auto tensor_proto_dir =
    model_path.has_filename() ? model_path.parent_path() : model_path;
ORT_RETURN_IF_ERROR(ReadExternalDataForTensor(ten_proto, tensor_proto_dir, unpacked_data));
```

Mirrors the pattern used by every other external-data code path in the
same file (`UnpackTensor` at line ~1053, `GetExtDataFromTensorProto` at
line ~1303).

## Our internal fix (applied and running in production)

- Fork: `https://github.com/Fulton-Engineering-Services/onnxruntime`
- Branch: `fix/coreml-tensorproto-external-data-path`
- Commit: `1e37c3583d05992bc1419269f87d941e8642248c`
- Base tag: `v1.23.2`
- Validated against BGE-M3 (fastembed-rs) on Apple Silicon M-series;
  service runs healthy with `{"status":"ok","workers":{"live":2,"total":2}}`
- Local ORT build at `~/.local/share/ort-build/` with `ORT_LIB_LOCATION`
  used in `bge-m3-axum-fastembed-rs` build

## Upstream PR — TODO

1. **Rebase onto `main`** (not v1.23.2 — PRs must target main)
2. **Write unit test** in `onnxruntime/test/framework/tensorutils_test.cc`
   - Use `PathValidationTest`-style temp-dir fixture
   - Call `TensorProtoWithExternalDataToTensorProto` with `model_path` =
     file path (e.g. `/tmp/dir/model.onnx`)
   - No existing test covers this function with a file-path model_path
3. **Sign Microsoft CLA** (automated bot on first PR submission)
4. **No existing upstream issue** — bug search came up empty; file a
   bug report or go direct to PR (CONTRIBUTING.md allows direct PR for bugs)
5. **No existing test** for `TensorProtoWithExternalDataToTensorProto`
   in the test file; our test is entirely new ground

## Key notes

- `ValidateExternalDataPath` (new in `main`, not in v1.23.2) is
  **orthogonal** — called from `graph.cc`, not from this call chain.
  Its comment even explicitly mentions HuggingFace Hub symlinks, same
  scenario we hit. Does not need changes for our fix.
- `main` has ~30 extra lines around the function vs v1.23.2, all from
  the new `ValidateExternalDataPath` addition; the fix applies cleanly.
- ORT PR guidelines: keep small, tests mandatory, describe motivation,
  no self-approval.
