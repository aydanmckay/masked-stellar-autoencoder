## 2025-04-26 - HDF5 I/O in tight loops
**Learning:** Opening HDF5 files (`h5py.File`) inside tight loops is a significant I/O bottleneck in Python, particularly for scripts operating on many files or large datasets. Opening and closing these files repeatedly adds high overhead compared to opening the file once outside the loop and writing datasets as needed.
**Action:** When doing file I/O within a loop over datasets or chunks, verify if the target file (like an `.h5` file) can be opened globally using a context manager `with h5py.File(..., 'a') as f:` before entering the loop. This single-open approach avoids the constant file locking and opening/closing costs.

## 2025-04-26 - PyTorch Masking Overhead
**Learning:** In PyTorch, using `torch.where(mask, a, b)` for masking operations allocates a new tensor for the result and involves reading multiple tensors, adding unnecessary overhead in tight training loops or loss functions.
**Action:** When applying a mask to zero-out elements (or fill with a constant), prefer using the in-place operation `.masked_fill_(~mask, 0.0)`. This operates directly on an existing temporary tensor, reducing memory allocation overhead and significantly improving execution speed.
