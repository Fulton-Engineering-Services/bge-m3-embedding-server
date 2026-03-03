# Documentation

Design documentation for `bge-m3-embedding-server`.

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture.md) | Component diagram, module layout, worker pool design, middleware stack |
| [Request Flow](request-flow.md) | End-to-end request lifecycle for dense and sparse endpoints |
| [Health State Machine](health-state-machine.md) | Health endpoint states, decision logic, Docker HEALTHCHECK integration |
| [Cold Start](cold-start.md) | Leader–follower startup pattern, failure modes, idle-reload comparison |
| [The BGE-M3 Model](bge-m3-model.md) | Model provenance, vocabulary, dense/sparse capabilities, hybrid scoring, vector storage compatibility |
| [Apple Silicon Build Target](apple-silicon.md) | Dependency chain, MLAS NEON kernels, CoreML EP status, release profile, launchd deployment |
