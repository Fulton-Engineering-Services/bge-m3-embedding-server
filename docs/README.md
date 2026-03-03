# Documentation

Design documentation for `bge-m3-embedding-server`.

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture.md) | Component diagram, module layout, worker pool design, middleware stack |
| [Request Flow](request-flow.md) | End-to-end request lifecycle for dense and sparse endpoints |
| [Health State Machine](health-state-machine.md) | Health endpoint states, decision logic, Docker HEALTHCHECK integration |
| [Cold Start](cold-start.md) | Leader–follower startup pattern, failure modes, idle-reload comparison |
| [The BGE-M3 Model](bge-m3-model.md) | Model provenance, vocabulary, dense/sparse capabilities, hybrid scoring, vector storage compatibility |
| [CoreML Execution Provider](coreml-ep.md) | Apple Silicon compute units, custom ORT build, ENOTDIR fix, BGE-M3 op coverage, CoreML EP configuration |
| [Performance](performance.md) | MLAS vs CoreML benchmarks, BGE_M3_ONNX_BATCH_SIZE fix, memory footprint, RAM reduction options |
| [Model Variants](model-variants.md) | FP32 vs FP16 precision evaluation, quantized model table, production recommendation |
| [macOS Deployment](deployment.md) | install-bge-m3-apple.sh, LaunchAgent configuration, service management |
