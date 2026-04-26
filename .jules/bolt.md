## 2025-05-15

**Optimization:** Pre-loading ensemble checkpoints once and reusing them across multiple prediction passes.
**File:** src/masked_stellar_autoencoder/training/eval_ensemble.py
**Learning:** Loading and deserializing large PyTorch checkpoints (pickle) repeatedly is a major I/O and CPU bottleneck. Caching state dicts in memory is a highly effective trade-off for scripts performing multiple passes over the same ensemble.
