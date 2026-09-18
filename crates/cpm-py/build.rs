//! Only exists for `cargo test`. The real extension module is always
//! dlopen'd into an already-running, already-initialized Python process, so
//! it never needs any of this.
//!
//! `cargo test`'s binary is not a `python3` process, so when a test needs to
//! touch a Python API (e.g. constructing a `PyErr`) and PyO3's
//! `auto-initialize` starts an embedded interpreter on first use, CPython's
//! own virtualenv redirect (following a venv's `pyvenv.cfg` back to the base
//! install that actually has the standard library) never fires — that
//! redirect is keyed off `argv[0]` naming a `python3` launcher, and here
//! `argv[0]` is the test binary itself. Left alone, the embedded interpreter
//! fails outright (`ModuleNotFoundError: No module named 'encodings'`)
//! whenever the active Python is a virtualenv whose own `lib/pythonX.Y`
//! holds only `site-packages` (as `uv`-managed venvs do) rather than the
//! full standard library.
//!
//! Fix: ask whichever `python3` `pyo3-build-config` would itself resolve
//! (respecting `PYO3_PYTHON` exactly like it does) for its `sys.base_prefix`
//! — the actual install that has the standard library — and bake that path
//! into the test binary so it can set `PYTHONHOME` before first touching a
//! Python API. No hardcoded machine-specific path anywhere in source; this
//! is recomputed on every build against whatever Python is actually active.
use std::process::Command;

fn main() {
    let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let output = Command::new(&python)
        .args(["-c", "import sys; print(sys.base_prefix)"])
        .output()
        .unwrap_or_else(|e| panic!("failed to run `{python} -c ...` to find its stdlib home: {e}"));
    if !output.status.success() {
        panic!(
            "`{python} -c 'import sys; print(sys.base_prefix)'` exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let base_prefix = String::from_utf8(output.stdout)
        .expect("python printed non-UTF-8 output")
        .trim()
        .to_string();
    println!("cargo:rustc-env=PYO3_TEST_PYTHONHOME={base_prefix}");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
}
