//! `cpm-core`: the Cellular Potts Model engine. This crate carries no PyO3
//! dependency `cpm-py` is the only binding layer, and the only place
//! `AnyCPM` dispatch lives. One engine, generic over `D in {2, 3}`, covers
//! both 2D and 3D simulation from the same code.
//!
//! Covers the resolved configuration schema, dimension-generic foundations
//! (halo padding, offset tables), parameter and state layout, the canonical
//! `theta` layout, the Monte Carlo loop (energy terms, incremental
//! bookkeeping, simple-point connectivity, deterministic RNG,
//! initialisation), the brute-force consistency checker, the
//! Metropolis-Hastings validation mode, and full readouts with run metadata
//! (`output.rs`).

pub mod cell;
pub mod checker;
pub mod config;
pub mod connectivity;
pub mod dynamics;
pub mod edge_list;
pub mod energy;
pub mod initialization;
pub mod labelling;
pub mod lattice;
pub mod model;
pub mod monte_carlo;
pub mod neighborhood;
pub mod output;
pub mod rng;
pub mod simple_point;
pub mod state;
pub mod theta;
