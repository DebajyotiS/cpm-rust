//! Explicit offset lists for the three independent neighbourhood roles:
//! `copy`, `energy`, and `connectivity`.
//!
//! Stencils are always explicit `Vec<[i32; D]>` offsets, never derived from one
//! another, so a new custom stencil is a matter of populating the `Custom`
//! variant of [`NeighborhoodKind`] rather than adding new code paths.

use serde::{Deserialize, Serialize};
use std::fmt;

/// An explicit neighbour offset list, with an optional per-offset weight used
/// by the energy neighbourhood. Connectivity and copy stencils ignore the
/// weights.
#[derive(Debug, Clone, PartialEq)]
pub struct Stencil<const D: usize> {
    pub offsets: Vec<[i32; D]>,
    pub weights: Vec<f64>,
}

impl<const D: usize> Stencil<D> {
    pub fn unweighted(offsets: Vec<[i32; D]>) -> Self {
        let weights = vec![1.0; offsets.len()];
        Self { offsets, weights }
    }

    pub fn weighted(offsets: Vec<[i32; D]>, weights: Vec<f64>) -> Self {
        assert_eq!(
            offsets.len(),
            weights.len(),
            "offsets and weights must have the same length"
        );
        Self { offsets, weights }
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// von Neumann (face) neighbours: 2*D offsets, unweighted. Generic over
    /// any D; the project only instantiates D = 2 and D = 3.
    pub fn von_neumann() -> Self {
        let mut offsets = Vec::with_capacity(2 * D);
        for axis in 0..D {
            let mut plus = [0i32; D];
            let mut minus = [0i32; D];
            plus[axis] = 1;
            minus[axis] = -1;
            offsets.push(plus);
            offsets.push(minus);
        }
        Self::unweighted(offsets)
    }

    /// `copy_neighborhood` must be a subset of `connectivity_neighborhood`.
    /// Weights are irrelevant to this check.
    pub fn is_subset_of(&self, other: &Stencil<D>) -> bool {
        self.offsets.iter().all(|o| other.offsets.contains(o))
    }

    /// The full `3^D - 1`-neighbour shell, unweighted: Moore in 2D, the
    /// 26-shell in 3D, generalised to any dimension. This is the fixed
    /// canonical basis `simple_point.rs`'s tables are built over, independent
    /// of whatever `energy_neighborhood`/`connectivity_neighborhood` a config
    /// actually chooses.
    pub fn full() -> Self {
        Self::unweighted(Self::full_moore())
    }

    /// Every offset in `{-1, 0, 1}^D` except the origin: the full Moore
    /// neighbourhood generalised to D dimensions (`3^D - 1` offsets).
    fn full_moore() -> Vec<[i32; D]> {
        let total = 3usize.pow(D as u32);
        let mut offsets = Vec::with_capacity(total - 1);
        for code in 0..total {
            let mut c = code;
            let mut offset = [0i32; D];
            let mut nonzero_count = 0usize;
            for slot in offset.iter_mut() {
                let digit = (c % 3) as i32 - 1;
                c /= 3;
                *slot = digit;
                if digit != 0 {
                    nonzero_count += 1;
                }
            }
            if nonzero_count > 0 {
                offsets.push(offset);
            }
        }
        offsets
    }
}

impl Stencil<2> {
    /// Default 2D energy neighbourhood: full Moore, 8 neighbours, unweighted.
    pub fn moore_2d() -> Self {
        Self::unweighted(Self::full_moore())
    }
}

impl Stencil<3> {
    /// Recommended 3D energy neighbourhood: face + edge neighbours, 18
    /// total, unweighted. Excludes the 8 corner (all-axes) contacts that make
    /// unweighted 26-counting anisotropic.
    pub fn eighteen_3d() -> Self {
        let offsets: Vec<[i32; 3]> = Self::full_moore()
            .into_iter()
            .filter(|o| o.iter().filter(|c| **c != 0).count() <= 2)
            .collect();
        Self::unweighted(offsets)
    }

    /// Default 3D connectivity neighbourhood for medium: full 26-neighbour
    /// shell, unweighted. Also usable as the unweighted-26 energy option,
    /// which is cheap but anisotropic.
    pub fn twenty_six_3d() -> Self {
        Self::unweighted(Self::full_moore())
    }

    /// Alternative 3D energy neighbourhood: full 26-neighbour shell, weighted
    /// `1, 1/sqrt(2), 1/sqrt(3)` for face / edge / corner contacts.
    pub fn twenty_six_3d_weighted() -> Self {
        let offsets = Self::full_moore();
        let weights = offsets
            .iter()
            .map(|o| {
                let nonzero = o.iter().filter(|c| **c != 0).count() as f64;
                1.0 / nonzero.sqrt()
            })
            .collect();
        Self::weighted(offsets, weights)
    }
}

/// Named, hashable choice of stencil for one of the three neighbourhood
/// roles. Deliberately not generic over `D`: a `ResolvedConfig<D>` picks one
/// of these per role and [`NeighborhoodKind::to_stencil`] validates it
/// against the concrete `D` (and, for `Custom`, against the offset length),
/// so an incompatible choice (e.g. `Moore` at D = 3) is a config error rather
/// than a type error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum NeighborhoodKind {
    /// 4 neighbours in 2D, 6 in 3D.
    VonNeumann,
    /// 8 neighbours. 2D only.
    Moore,
    /// 18 neighbours (face + edge). 3D only.
    Eighteen,
    /// 26 neighbours, unweighted. 3D only.
    TwentySix,
    /// 26 neighbours, weighted `1, 1/sqrt(2), 1/sqrt(3)`. 3D only.
    TwentySixWeighted,
    /// An explicit, user-supplied offset list. Each inner `Vec<i32>` must
    /// have length `D`. `weights` must either be absent (unweighted) or
    /// match `offsets` in length.
    Custom {
        offsets: Vec<Vec<i32>>,
        weights: Option<Vec<f64>>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeighborhoodKindError(pub String);

impl fmt::Display for NeighborhoodKindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for NeighborhoodKindError {}

impl NeighborhoodKind {
    pub fn to_stencil<const D: usize>(&self) -> Result<Stencil<D>, NeighborhoodKindError> {
        match self {
            NeighborhoodKind::VonNeumann => Ok(Stencil::<D>::von_neumann()),
            NeighborhoodKind::Moore => {
                if D == 2 {
                    // Rust cannot return a `Stencil<2>` as `Stencil<D>` without `D == 2`
                    // proven at the type level, so the offsets are built generically
                    // instead of constructing via `Stencil<2>` directly.
                    let full = Stencil::<D>::full_moore_pub();
                    Ok(Stencil::unweighted(full))
                } else {
                    Err(NeighborhoodKindError(format!(
                        "Moore neighbourhood is 2D only, got D = {D}"
                    )))
                }
            }
            NeighborhoodKind::Eighteen => {
                if D == 3 {
                    let full = Stencil::<D>::full_moore_pub();
                    let offsets: Vec<[i32; D]> = full
                        .into_iter()
                        .filter(|o| o.iter().filter(|c| **c != 0).count() <= 2)
                        .collect();
                    Ok(Stencil::unweighted(offsets))
                } else {
                    Err(NeighborhoodKindError(format!(
                        "Eighteen-neighbour is 3D only, got D = {D}"
                    )))
                }
            }
            NeighborhoodKind::TwentySix => {
                if D == 3 {
                    Ok(Stencil::unweighted(Stencil::<D>::full_moore_pub()))
                } else {
                    Err(NeighborhoodKindError(format!(
                        "Twenty-six-neighbour is 3D only, got D = {D}"
                    )))
                }
            }
            NeighborhoodKind::TwentySixWeighted => {
                if D == 3 {
                    let offsets = Stencil::<D>::full_moore_pub();
                    let weights = offsets
                        .iter()
                        .map(|o| {
                            let nonzero = o.iter().filter(|c| **c != 0).count() as f64;
                            1.0 / nonzero.sqrt()
                        })
                        .collect();
                    Ok(Stencil::weighted(offsets, weights))
                } else {
                    Err(NeighborhoodKindError(format!(
                        "Weighted twenty-six-neighbour is 3D only, got D = {D}"
                    )))
                }
            }
            NeighborhoodKind::Custom { offsets, weights } => {
                let mut fixed = Vec::with_capacity(offsets.len());
                for o in offsets {
                    if o.len() != D {
                        return Err(NeighborhoodKindError(format!(
                            "custom offset {o:?} has length {}, expected {D}",
                            o.len()
                        )));
                    }
                    let mut arr = [0i32; D];
                    arr.copy_from_slice(o);
                    fixed.push(arr);
                }
                match weights {
                    Some(w) => {
                        if w.len() != fixed.len() {
                            return Err(NeighborhoodKindError(format!(
                                "custom weights length {} does not match offsets length {}",
                                w.len(),
                                fixed.len()
                            )));
                        }
                        Ok(Stencil::weighted(fixed, w.clone()))
                    }
                    None => Ok(Stencil::unweighted(fixed)),
                }
            }
        }
    }
}

impl<const D: usize> Stencil<D> {
    /// Public wrapper over `full_moore` for `NeighborhoodKind::to_stencil`,
    /// which needs it outside this module without exposing it as API surface
    /// on every `Stencil<D>`.
    fn full_moore_pub() -> Vec<[i32; D]> {
        Self::full_moore()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn von_neumann_counts() {
        assert_eq!(Stencil::<2>::von_neumann().len(), 4);
        assert_eq!(Stencil::<3>::von_neumann().len(), 6);
    }

    #[test]
    fn moore_2d_count() {
        assert_eq!(Stencil::moore_2d().len(), 8);
    }

    #[test]
    fn eighteen_3d_count() {
        assert_eq!(Stencil::eighteen_3d().len(), 18);
    }

    #[test]
    fn twenty_six_3d_counts() {
        assert_eq!(Stencil::twenty_six_3d().len(), 26);
        assert_eq!(Stencil::twenty_six_3d_weighted().len(), 26);
    }

    #[test]
    fn weighted_26_weights_by_contact_type() {
        let s = Stencil::twenty_six_3d_weighted();
        for (offset, weight) in s.offsets.iter().zip(s.weights.iter()) {
            let nonzero = offset.iter().filter(|c| **c != 0).count();
            let expected = 1.0 / (nonzero as f64).sqrt();
            assert!((weight - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn no_duplicate_or_zero_offsets() {
        for s in [
            Stencil::<2>::von_neumann().offsets,
            Stencil::moore_2d().offsets,
        ] {
            assert!(!s.contains(&[0, 0]));
            let mut sorted = s.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), s.len());
        }
        for s in [
            Stencil::<3>::von_neumann().offsets,
            Stencil::eighteen_3d().offsets,
            Stencil::twenty_six_3d().offsets,
        ] {
            assert!(!s.contains(&[0, 0, 0]));
            let mut sorted = s.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), s.len());
        }
    }

    #[test]
    fn von_neumann_is_subset_of_moore_2d() {
        let vn = Stencil::<2>::von_neumann();
        let moore = Stencil::moore_2d();
        assert!(vn.is_subset_of(&moore));
        assert!(!moore.is_subset_of(&vn));
    }

    #[test]
    fn von_neumann_is_subset_of_twenty_six_3d() {
        let vn = Stencil::<3>::von_neumann();
        let big = Stencil::twenty_six_3d();
        assert!(vn.is_subset_of(&big));
    }

    #[test]
    fn kind_dispatch_matches_direct_constructors() {
        let via_kind: Stencil<2> = NeighborhoodKind::VonNeumann.to_stencil().unwrap();
        assert_eq!(via_kind, Stencil::<2>::von_neumann());

        let via_kind: Stencil<3> = NeighborhoodKind::Eighteen.to_stencil().unwrap();
        assert_eq!(via_kind, Stencil::<3>::eighteen_3d());
    }

    #[test]
    fn kind_rejects_wrong_dimension() {
        assert!(NeighborhoodKind::Moore.to_stencil::<3>().is_err());
        assert!(NeighborhoodKind::Eighteen.to_stencil::<2>().is_err());
        assert!(NeighborhoodKind::TwentySix.to_stencil::<2>().is_err());
        assert!(NeighborhoodKind::TwentySixWeighted
            .to_stencil::<2>()
            .is_err());
    }

    #[test]
    fn custom_kind_validates_offset_length() {
        let ok = NeighborhoodKind::Custom {
            offsets: vec![vec![1, 0], vec![0, 1]],
            weights: None,
        };
        assert!(ok.to_stencil::<2>().is_ok());

        let bad = NeighborhoodKind::Custom {
            offsets: vec![vec![1, 0, 0]],
            weights: None,
        };
        assert!(bad.to_stencil::<2>().is_err());
    }

    #[test]
    fn custom_kind_validates_weight_length() {
        let bad = NeighborhoodKind::Custom {
            offsets: vec![vec![1, 0], vec![0, 1]],
            weights: Some(vec![1.0]),
        };
        assert!(bad.to_stencil::<2>().is_err());
    }
}
