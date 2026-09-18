# cpm

```
 ██████╗██████╗ ███╗   ███╗      ██████╗ ██╗   ██╗███████╗████████╗
██╔════╝██╔══██╗████╗ ████║      ██╔══██╗██║   ██║██╔════╝╚══██╔══╝
██║     ██████╔╝██╔████╔██║█████╗██████╔╝██║   ██║███████╗   ██║
██║     ██╔═══╝ ██║╚██╔╝██║╚════╝██╔══██╗██║   ██║╚════██║   ██║
╚██████╗██║     ██║ ╚═╝ ██║      ██║  ██║╚██████╔╝███████║   ██║
 ╚═════╝╚═╝     ╚═╝     ╚═╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝
```

![Rust](https://img.shields.io/badge/rust-1.80%2B-orange?logo=rust&logoColor=white)
![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A Cellular Potts Model (CPM) simulator for organoid tissue. The computational core (`cpm-core`) is pure Rust and generic over 2D and 3D spatial dimensions from a single implementation. Python bindings (`cpm-py`) expose the engine through a user-facing Python package (`python/cpm`).

The simulator serves as a forward model for simulation-based inference (TMNRE) on organoid imaging data:

$$\theta \longrightarrow \text{stochastic CPM dynamics} \longrightarrow y$$

Here, $\theta$ is a vector of biophysical parameters (adhesion energies, volume/interface stiffnesses, etc), and $y$ is a vector of summary statistics compared against experimental images. Correctness and auditability take priority over speed or biological complexity. 


## Math & Mechanics

The lattice array $\sigma$ stores cell IDs (`0` for medium, positive integers for individual cells) padded with a one-site halo to allow branch-free neighbor lookups. Cell identity (the lattice value) and cell type $\tau(c)$ are distinct: two distinct cells can share the same type.

The conservative Hamiltonian is:

$$H = H_V + H_I + H_J$$

* **Volume:** Applied to non-medium cells ($c \ge 1$):

$$H_V = \sum_{c \ge 1} \lambda_V[\tau(c)] \, (V_c - V^*[\tau(c)])^2$$


* **Interface:** Applied to non-medium cells. $I_c$ measures raw lattice interface counts under the configured energy neighborhood, not Euclidean perimeter or surface area:

$$H_I = \sum_{c \ge 1} \lambda_I[\tau(c)] \, (I_c - I^*[\tau(c)])^2$$


* **Adhesion:** Summed over neighboring sites with different cell IDs (two distinct cells of the same type still pay contact energy $J[\alpha][\alpha]$):

$$H_J = \sum_{\langle i,j \rangle,\ \sigma_i \ne \sigma_j} J[\tau(\sigma_i)][\tau(\sigma_j)]$$



Target constants $V^*$ and $I^*$ are calibrated by relaxing a single cell under the simulator's neighborhood rules (see `crates/cpm-core/examples/calibrate_targets.rs`).

### Monte Carlo Step Loop

For each step attempt:

1. Select a target site $s$ uniformly at random from the lattice interior.
2. Select a source site $s'$ uniformly from `copy_neighborhood(s)`.
3. If $\sigma[s] == \sigma[s']$, return immediately.
4. If the move breaks simple-point connectivity for the losing cell, reject.
5. If the move reduces the losing cell volume below 1, reject.
6. Calculate $\Delta H$ locally.
7. Accept if $\Delta H \le 0$; otherwise accept with probability $P(\Delta H)$.

Temperature $T$ is fixed at $1$ to remove parameter degeneracy with $J$ and $\lambda$ scales. The Metropolis acceptance probability is:

$$P(\Delta H) = \begin{cases} 1 & \text{if } \Delta H \le 0 \\ \exp(-\Delta H) & \text{if } \Delta H > 0 \end{cases}$$

Proposal probabilities are asymmetric across forward and reverse moves, so standard CPM runs sample phenomenological tissue dynamics rather than an equilibrium distribution $\exp(-H)$.

### Act Migration Model

The Act model adds a non-conservative energy delta:

$$\Delta H_{\text{Act}} = -\frac{\lambda_{\text{Act}}}{\text{Max}_{\text{Act}}} \left(\text{GM}_{\text{src}} - \text{GM}_{\text{tgt}}\right)$$

Where $\text{GM}$ is the geometric mean of per-site activity over the copy neighborhood restricted to the respective cell. Because Act is non-conservative, it has no global $H$ and is excluded from brute-force energy checks.

### Connectivity Checks

Proposed copies are checked against simple-point connectivity for the losing cell using precomputed 2D and 3D lookup tables (Bertrand–Malandain topological numbers). Medium (cell ID `0`) is exempt from this check to allow lumen formation.

## Repository Structure

```
crates/
├── cpm-core/              Pure Rust simulator engine
│   └── src/
│       ├── config.rs        UserConfig, ResolvedConfig, and stencil assertions
│       ├── lattice.rs        Halo-padded lattice grid and boundary logic
│       ├── cell.rs           CellType definitions and identity separation
│       ├── state.rs          Incremental cell volume, interface, and centroid tracking
│       ├── model.rs          CPM<D> core engine struct
│       ├── neighborhood.rs   Stencils (von Neumann, Moore, 18, 26)
│       ├── energy.rs         VolumeTerm, InterfaceTerm, AdhesionTerm, ActTerm
│       ├── monte_carlo.rs    Attempt and accept update loops
│       ├── connectivity.rs   Simple-point topology checks for losing cells
│       ├── theta.rs          Canonical parameter vector serialization
│       └── output.rs         RunResult and Sample output types
└── cpm-py/                PyO3 extension bindings layer
    └── src/
        ├── lib.rs            Module entry point and AnyCPM dimension dispatch
        ├── builder.rs        Python-facing CPM builder class
        └── convert.rs        Rust structs to NumPy array conversions

python/cpm/                User-facing Python package
├── model.py                 CPM class wrapper around _cpm.CPM
├── parameters.py             Cell density helpers (cells_for_density)
└── warnings.py               Custom initialization warnings

```

Dimensionality is inferred from the shape of the `grid` argument passed during initialization.

## Installation & Setup

Requires Rust (edition 2021, MSRV 1.80) and Python >= 3.9. There is no published
wheel yet, so `cpm` is only available by building from source — pre-built wheel
distribution is on the roadmap.

### Working in this repo

```bash
# Create virtual environment and install dependencies
uv sync

# Build Rust bindings into the local environment
maturin develop

```

Rebuild with `maturin develop` when modifying Rust code in `crates/`. Changes to Python code under `python/cpm/` will apply immediately.

### Using `cpm` from another project

`cpm-py` depends on `cpm-core` via a local path within this repo rather than a
published crate, so building the extension needs this whole repo, not just the
Python package directory. From another project's environment:

```bash
pip install /path/to/cpm            # local checkout
# or, once this repo has a remote:
pip install git+<repo-url>
```

Either form requires a Rust toolchain on the machine running the install — `pip`
invokes `maturin`, which compiles `cpm-py` (and its `cpm-core` dependency) from
source via the `[tool.maturin]` config in `pyproject.toml`. There is no way to
depend on just the compiled extension without either building it yourself or
installing a wheel someone else built.

## Quickstart

```python
import cpm

# Initialize simulation grid
sim = cpm.CPM(
    grid=(30, 30),
    boundary="periodic",
    seed=1,
    copy_neighborhood="von_neumann",
    connectivity_neighborhood="von_neumann",
)

# Configure cell types and parameters
sim.add_cell_type(
    name="epithelial",
    target_volume=50,
    target_interface=75,
    lambda_volume=10.0,
    lambda_interface=2.0,
)
sim.add_cells(cell_type="epithelial", n=5)

# Set symmetric contact energy matrix (index 0 is medium)
sim.set_adhesion([[0.0, 5.0], [5.0, 2.0]])

# Run simulation
result = sim.run(burn_in_mcs=5000, readout_mcs=5000, sampling_interval_mcs=100)

print(f"Density: {result.metadata.derived_phi}")
print(f"Status: {result.status.kind}")

# Access summary metrics from the last sample
sample = result.samples[-1]
print(sample.volume)
print(sample.centroid)

```

Use `cpm.cells_for_density` to calculate cell count for a target packing density $\phi$:

```python
n_cells = cpm.cells_for_density(grid=(64, 64, 64), target_volume=1048, phi=0.8)

```

## Testing & Benchmarks

```bash
# Run unit and integration tests
cargo test -p cpm-core

# Run brute-force consistency checker (recomputes state every move)
cargo test -p cpm-core --features checker -- --ignored

# Run benchmarks
cargo bench

# Run Python test suite
maturin develop && pytest tests/python

```

The `--features checker` flag recomputes global energy, volumes, and interface counts from scratch after every accepted move to catch incremental state drift.

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request: `cargo fmt --check`,
`clippy -D warnings`, an MSRV check pinned to the `rust-version` in `Cargo.toml`,
`cargo test -p cpm-core`/`-p cpm-py` (release, non-ignored), a compile-only check of
the Criterion benches, and the `pytest` suite built via `maturin develop`. All of
that finishes in well under a minute.

The brute-force checker and the exact-Boltzmann validation tests are excluded from
that workflow — measured end-to-end, `cargo test -p cpm-core --release --features
checker -- --ignored` takes ~12 minutes, almost all of it the exact-Boltzmann
enumeration, which is far too slow to gate every push. `.github/workflows/slow-tests.yml`
runs them instead: nightly on a schedule, on every push to `main`, on demand via
`workflow_dispatch`, and — because these are the tests that catch silent
incremental-bookkeeping drift, the project's own highest-priority failure mode — on
any pull request that touches a correctness-critical module (`energy.rs`,
`dynamics.rs`, `connectivity.rs`, `simple_point.rs`, `monte_carlo.rs`, `state.rs`,
`theta.rs`, `checker.rs`, `edge_list.rs`). A PR that only touches the Python layer or
docs never pays the 12-minute cost; one that touches the Hamiltonian or the Monte
Carlo loop pays it before merge, not just at the next nightly run.

## Frequently Asked Questions

**Why is there no temperature parameter?**
Temperature $T$ is mathematically degenerate with $J$ and $\lambda$. Fixing $T = 1$ eliminates redundant parameter combinations that would make parameter estimation non-identifiable during inference.

**Why is packing density ($\phi$) derived rather than set directly?**
Packing density depends on grid size, cell counts, and target volumes. Direct setting would allow inconsistent configurations or alter cell counts dynamically, changing the dimension of the summary statistic vector $y$ across training sets.

**Why does `run()` issue a `CPMInitializationWarning`?**
This warning triggers when `sim.initialize()` is omitted, defaulting to scatter-and-grow placement. Initial placement affects required burn-in length. Pass explicit placement parameters to `sim.initialize()` or set `warn_on_default_init=False` to suppress the warning.

**Why does the 3D energy neighborhood default to 18-neighbor instead of 26-neighbor?**
Unweighted 26-neighbor stencils in 3D introduce significant grid anisotropy, causing artificial cubic faceting on cell surfaces. The 18-neighbor stencil offers a better balance between isotropy and performance.

**Why do cells of the same type pay non-zero contact energy?**
Adhesion energies are calculated per cell ID rather than per cell type. Two separate cells of the same type still pay contact energy $J[\alpha][\alpha]$ along their shared boundary.

## Guidelines

* Keep `cpm-core` completely free of PyO3 dependencies; binding logic belongs exclusively in `cpm-py`.
* Ensure energy values remain `f64` throughout computations.
* Avoid Python callbacks or per-step FFI calls inside the Monte Carlo execution loop.