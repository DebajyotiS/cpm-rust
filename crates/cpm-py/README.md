# cpm-py

The PyO3 binding layer connecting [`cpm-core`](../cpm-core) to the Python package in [`python/cpm/`](../../python/cpm). This is the only crate in the workspace with a PyO3 dependency. It handles spatial dimension dispatch using an enum wrapper (`AnyCPM { D2(CPM<2>), D3(CPM<3>) }` in `src/lib.rs`), which infers the dimension directly from the length of the `grid` tuple passed from Python. Passing a grid tuple with an unsupported length returns an error.

For general project documentation, see the [main README](../../README.md).

## Module Overview

| Module | Responsibility |
| --- | --- |
| `lib.rs` | Handles `_cpm` module registration, `AnyCPM` dimension dispatch, and `DispatchError` definitions. |
| `builder.rs` | Exposes the `Simulation` struct (registered in Python as `CPM`). Methods mutate a `UserConfig<D>` struct on the Python side until `run()` triggers Rust execution. |
| `convert.rs` | Converts `cpm-core` output types (`RunResult<D>`, `Sample<D>`) into NumPy-backed Python classes (`PyRunResult`, `PySample`, `PyRunMetadata`, `PyStatus`). |
| `errors.rs` | Converts `cpm-core` configuration `Result` errors into Python exceptions with descriptive error messages. |

The public `CPM` class in `python/cpm/model.py` wraps `builder::Simulation`. Users should import `cpm` through Python rather than interacting with `cpm-py` directly.

## FFI Boundary Design

Execution crosses the FFI boundary only once per simulation run. A single Python call executes the entire trajectory in Rust and returns summary arrays. There are no per-step callbacks or FFI round-trips during Monte Carlo updates.

`Simulation::run` releases the Python GIL during trajectory execution:

```python
# Standard coarse execution
result = sim.run(burn_in_mcs=5000, readout_mcs=5000, sampling_interval_mcs=100)

```

Per-step step iteration from Python is deliberately unsupported to avoid FFI overhead.

## Building & Testing

```bash
# Build extension and install the cpm package into the active virtual environment
maturin develop

# Run Rust-side binding tests for AnyCPM dispatch
cargo test -p cpm-py

```

`pyproject.toml` at the workspace root configures `maturin` to compile this crate (`crates/cpm-py/Cargo.toml`) into `python/cpm/_cpm`, where the Python package wraps it.