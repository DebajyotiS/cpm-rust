# cpm-core

The core Cellular Potts Model simulation engine. Written in pure Rust with zero dependencies on PyO3 or Python bindings. It uses `const D: usize` (`Lattice<D>`, `ResolvedConfig<D>`, `State<D>`, `CPM<D>`) to handle 2D and 3D implementations through a single generic codebase. 

## Module Overview

| Module | Responsibility |
| --- | --- |
| `config.rs` | Handles user configuration parsing (`UserConfig<D>`), resolved configurations (`ResolvedConfig<D>`), and stencil assertions. |
| `lattice.rs` | Provides halo-padded storage arrays, row-major strides, and periodic/fixed boundary conditions. |
| `cell.rs` | Defines `CellType` and maintains separation between cell identity and cell type. |
| `state.rs` | Tracks per-cell volumes, interface counts, and unwrapped centroid position sums. |
| `model.rs` | Exposes the primary simulation engine struct `CPM<D>`. |
| `neighborhood.rs` | Generates lookup stencils for von Neumann, Moore, 18-neighbor, and 26-neighbor setups. |
| `energy.rs` | Implements conservative terms (volume, interface, adhesion) and non-conservative deltas (Act migration). |
| `dynamics.rs` | Applies accepted copies and updates cell state tracking incrementally. |
| `monte_carlo.rs` | Runs attempt loops under Metropolis or Metropolis-Hastings acceptance modes. |
| `edge_list.rs` | Implements boundary-only proposal lists using binomial thinning while preserving step definitions. |
| `connectivity.rs` | Evaluates simple-point connectivity for losing cells, exempting medium to allow lumen formation. |
| `simple_point.rs` | Contains precomputed 2D and 3D topological simple-point lookup tables. |
| `initialization.rs` | Handles explicit grid placement and default scatter-and-grow seeding. |
| `theta.rs` | Manages parameter vector extraction and injection for model orchestration. |
| `labelling.rs` | Performs connected-component labeling for fragmentation and medium lumen detection. |
| `checker.rs` | Contains the brute-force state consistency checker enabled via the `checker` feature flag. |
| `rng.rs` | Provides deterministic PRNG seeding using Xoshiro256++. |
| `output.rs` | Defines `RunResult<D>` and `Sample<D>` structs for centroids, contact graphs, and metadata. |

Related directories:

* `tests/`: Integration tests for exact Boltzmann distributions, 2D reference trajectories, and proposal equivalence.
* `benches/`: Benchmark suites using Criterion.
* `examples/`: Single-cell target calibration tools (`calibrate_targets`) and Boltzmann diagnostics (`boltzmann_diagnose`).

## Building & Testing

```bash
# Build core crate
cargo build -p cpm-core

# Run unit and integration tests
cargo test -p cpm-core

# Run brute-force consistency checker
# Recomputes volume, interface, and global energy from scratch after every accepted move
cargo test -p cpm-core --features checker -- --ignored

# Run benchmarks
cargo bench -p cpm-core

```

## Internal Invariants

* **Double Precision:** Energy values use `f64` throughout computation. The consistency checker verifies exact equality against integer-derived quantities, requiring precise floating-point operations.
* **Local Deltas:** The global Hamiltonian is never recomputed inside the Monte Carlo loop. State transitions calculate a local $\Delta H$ only. Global recomputations occur strictly inside `checker.rs`.
* **Fixed Temperature:** Temperature is fixed at $T = 1$ by design. It is not an adjustable parameter or struct field in any execution path.
* **Act Model Separation:** Non-conservative deltas from the Act model do not contribute to `State`'s running conservative energy sum. `ActTerm::energy` returns `None`, and Act energy deltas are evaluated solely during acceptance checks.