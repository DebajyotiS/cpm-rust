//! Deterministic RNG. `Xoshiro256++` — fast, never a cryptographic RNG
//! (`StdRng` is ChaCha12) in the hot loop.

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub type Rng = Xoshiro256PlusPlus;

/// A single run's RNG, seeded directly from `config.seed`. A standalone run
/// is treated as simulation index 0; batch execution across many
/// simulations uses [`derive_seed`] instead.
pub fn rng_from_seed(seed: u64) -> Rng {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// Derive a simulation's seed deterministically from `(master_seed,
/// simulation_index)` via a fixed hash, so results are independent of
/// thread scheduling. Used by the batch execution layer to give each
/// simulation in a training set its own independent, reproducible stream.
pub fn derive_seed(master_seed: u64, simulation_index: u64) -> [u8; 32] {
    let mut input = [0u8; 16];
    input[..8].copy_from_slice(&master_seed.to_le_bytes());
    input[8..].copy_from_slice(&simulation_index.to_le_bytes());
    *blake3::hash(&input).as_bytes()
}

/// An RNG seeded from a derived `(master_seed, simulation_index)` seed.
pub fn rng_from_derived_seed(master_seed: u64, simulation_index: u64) -> Rng {
    Xoshiro256PlusPlus::from_seed(derive_seed(master_seed, simulation_index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::RngCore;

    #[test]
    fn same_seed_gives_identical_streams() {
        let mut a = rng_from_seed(42);
        let mut b = rng_from_seed(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_give_different_streams() {
        let mut a = rng_from_seed(1);
        let mut b = rng_from_seed(2);
        let stream_a: Vec<u64> = (0..20).map(|_| a.next_u64()).collect();
        let stream_b: Vec<u64> = (0..20).map(|_| b.next_u64()).collect();
        assert_ne!(stream_a, stream_b);
    }

    #[test]
    fn derive_seed_is_deterministic_and_index_sensitive() {
        assert_eq!(derive_seed(7, 3), derive_seed(7, 3));
        assert_ne!(derive_seed(7, 3), derive_seed(7, 4));
        assert_ne!(derive_seed(7, 3), derive_seed(8, 3));
    }

    #[test]
    fn derived_rng_streams_are_reproducible() {
        let mut a = rng_from_derived_seed(7, 3);
        let mut b = rng_from_derived_seed(7, 3);
        for _ in 0..50 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
