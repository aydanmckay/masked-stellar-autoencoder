## 2026-04-26 - [Replace insecure os.system calls]
**Vulnerability:** Insecure use of `os.system` for filesystem operations and package management. `os.system` runs a shell command, which is susceptible to shell injection if any part of the command string is user-influenced. It also lacks granular error handling and environment control compared to specialized alternatives.

**Learning:** `shutil.rmtree(path, ignore_errors=True)` is the preferred way to delete directory trees in Python as it is cross-platform, avoids shell overhead, and handles missing paths gracefully. For external commands like `pip`, `subprocess.check_call` with a list of arguments is much safer than `os.system` as it avoids the shell entirely and simplifies correct argument passing.

**Prevention:** Avoid `os.system` entirely. Use the `os` or `shutil` modules for filesystem operations. Use the `subprocess` module for external commands, passing arguments as a list to bypass the shell.
