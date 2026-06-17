## 2025-04-26 - HDF5 I/O in tight loops
**Learning:** Opening HDF5 files (`h5py.File`) inside tight loops is a significant I/O bottleneck in Python, particularly for scripts operating on many files or large datasets. Opening and closing these files repeatedly adds high overhead compared to opening the file once outside the loop and writing datasets as needed.
**Action:** When doing file I/O within a loop over datasets or chunks, verify if the target file (like an `.h5` file) can be opened globally using a context manager `with h5py.File(..., 'a') as f:` before entering the loop. This single-open approach avoids the constant file locking and opening/closing costs.

## 2025-04-26 - PyTorch Masking Overhead
**Learning:** In PyTorch, using `torch.where(mask, a, b)` for masking operations allocates a new tensor for the result and involves reading multiple tensors, adding unnecessary overhead in tight training loops or loss functions.
**Action:** When applying a mask to zero-out elements (or fill with a constant), prefer using the in-place operation `.masked_fill_(~mask, 0.0)`. This operates directly on an existing temporary tensor, reducing memory allocation overhead and significantly improving execution speed.

## 2025-05-19 - PyTorch Diagonal Masking Overhead
**Learning:** In PyTorch custom loss functions (like `RnCLoss`), repeatedly calling `.masked_select()` with a dynamically constructed diagonal mask (e.g., `(1 - torch.eye(n)).bool()`) causes significant memory allocations and processing overhead.
**Action:** When filtering diagonals or specific elements from multiple tensors sharing the same dimension, precompute the boolean mask once (e.g., `mask = ~torch.eye(n, dtype=torch.bool)`) and apply it using direct boolean indexing (`tensor[mask]`) instead of `.masked_select()`. This reduces intermediate tensor creations and improves performance.

## 2026-05-01 - Optimizing HDF5 Dataset Creation
**Learning:** Instantiating `pandas.DataFrame` purely as an intermediate step to construct structured arrays for HDF5 `create_dataset` incurs significant and unnecessary Pandas overhead.
**Action:** When assembling tabular data strictly for writing to HDF5 datasets, always use native NumPy structured arrays (`np.empty(len(data), dtype=[...])`) to drastically improve script execution speed and reduce memory consumption.

## 2026-05-24 - PyTorch Loss Function GPU Sync Bottlenecks
**Learning:** In PyTorch, using dynamic-shape boolean array indexing (e.g., `input[mask]`) on GPU tensors triggers costly Device-to-Host (GPU to CPU) synchronizations. This is because the CPU needs to determine the resulting tensor's dynamic shape to allocate memory. We measured this and found it slows down loss calculations by 30-50%.
**Action:** When calculating masked reductions (`mean` or `sum`) on tensors, use `.masked_fill(~mask, 0.0)` on the full-shape tensors, compute the error metric, and then apply a sum reduction (e.g., `.sum() / mask.sum().clamp_min(1)`). Crucially, you must sanitize `NaN` values *before* math operations (e.g. `(safe_pred - safe_target)**2`), otherwise `NaN` gradients will propagate during the backward pass even if masked out later. Preserve the original boolean indexing pattern specifically for `reduction="none"` to maintain API shape contracts.

## 2026-05-12 - Vectorizing PyTorch Custom Losses
**Learning:** In PyTorch custom loss functions (like RnCLoss), using Python `for` loops for row-wise contrastive metrics results in slow O(n) execution times and loop overhead. Replacing loops with broadcasting using `.unsqueeze()` transforms the operation into a single vectorized O(1) step, dramatically increasing performance.
**Action:** When implementing contrastive or pairwise loss functions in PyTorch, always evaluate pairs using multi-dimensional broadcasting (e.g., `tensor.unsqueeze(1) - tensor.unsqueeze(2)`) instead of explicit loops over dimension sizes.

## 2026-05-25 - Avoid data-dependent branching in custom losses
**Learning:** In PyTorch, conditional branches that evaluate tensor data (e.g., `if mask.sum() == 0:`) force costly Device-to-Host (GPU to CPU) synchronizations, stalling the execution pipeline. This is particularly problematic in tight loops or custom loss functions.
**Action:** Remove data-dependent conditionals by using unconditionally safe mathematical operations. For example, instead of branching on a zero sum, use `.clamp_min(1)` (or equivalent) in denominators to ensure safe division without blocking CPU-GPU execution.

## 2026-05-13 - Avoid 3D tensor broadcasting in row-wise contrastive losses
**Learning:** In PyTorch custom loss functions (e.g., `RnCLoss`), vectorizing row-wise contrastive metrics using 3D tensor broadcasting (e.g., `a.unsqueeze(1) >= b.unsqueeze(2)`) creates an O(N^3) memory footprint. While this eliminates Python loops and is faster for small batches, it causes severe Out-Of-Memory errors for larger batches.
**Action:** Prioritize memory scalability by using efficient `for` loops instead of high-dimensional multi-tensor broadcasting when evaluating pair-wise or row-wise loss components on batches, especially for components like contrastive loss that evaluate every item against every other item.

## 2026-05-14 - Avoid dynamic boolean indexing in quantile loss
**Learning:** In PyTorch, using dynamic-shape boolean indexing like `loss[mask].mean()` forces device-to-host synchronization, causing massive slowdowns in tight loops or custom loss functions.
**Action:** Replace dynamic indexing with full-shape tensor operations like `loss.masked_fill(~mask, 0.0).sum() / mask.sum().clamp_min(1)`. Ensure the mask replacement is out-of-place (e.g. `masked_fill` instead of `masked_fill_`) if in-place modifications trigger autograd errors or undefined behavior in edge cases.
