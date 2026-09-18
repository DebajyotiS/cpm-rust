//! Manages cell type definitions and per-type configuration parameters.
//! Holds pure data structures; calculation logic is separated into `energy`
//! and `dynamics`.

use crate::lattice::CellId;
use serde::{Deserialize, Serialize};

/// Per-type parameters. `target_volume`/`target_interface` are calibration
/// constants, never members of `theta`. Stiffnesses and Act parameters are
/// theta-eligible and are per type deliberately, since differential
/// stiffness between populations is something we intend to infer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellType {
    pub name: String,
    /// `V*`, from calibration. Not inferred.
    pub target_volume: u32,
    /// `I*`, from calibration, under the configured energy neighbourhood
    /// and weighting. Not inferred, not a Euclidean quantity.
    pub target_interface: u32,
    pub lambda_volume: f64,
    pub lambda_interface: f64,
    /// Act parameters. Carried in the schema now so `theta`'s layout is
    /// stable ahead of the Act energy term's implementation.
    pub lambda_act: f64,
    pub max_act: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CellTypeError(pub String);

impl std::fmt::Display for CellTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CellTypeError {}

impl CellType {
    pub fn validate(&self) -> Result<(), CellTypeError> {
        if self.name.is_empty() {
            return Err(CellTypeError("cell type name must not be empty".into()));
        }
        for (label, value) in [
            ("lambda_volume", self.lambda_volume),
            ("lambda_interface", self.lambda_interface),
            ("lambda_act", self.lambda_act),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(CellTypeError(format!(
                    "{label} must be finite and non-negative, got {value}"
                )));
            }
        }
        // The Act energy term divides by `Max_Act`; a nonzero `lambda_act`
        // with `max_act == 0` would silently produce `inf`/`NaN` the first
        // time `active_terms.act` is turned on, rather than failing loudly
        // here.
        if self.lambda_act > 0.0 && self.max_act == 0 {
            return Err(CellTypeError(
                "max_act must be positive when lambda_act is positive".into(),
            ));
        }
        Ok(())
    }
}

/// One individual cell. Identity (`id`) is assigned at initialisation and
/// stable for the run — cells are never created or destroyed mid-run.
/// `type_index` indexes into the resolved config's `cell_types`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub id: CellId,
    pub type_index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_type() -> CellType {
        CellType {
            name: "epithelial".into(),
            target_volume: 50,
            target_interface: 116,
            lambda_volume: 10.0,
            lambda_interface: 2.0,
            lambda_act: 0.0,
            max_act: 0,
        }
    }

    #[test]
    fn valid_cell_type_passes() {
        assert!(valid_type().validate().is_ok());
    }

    #[test]
    fn empty_name_rejected() {
        let mut t = valid_type();
        t.name = String::new();
        assert!(t.validate().is_err());
    }

    #[test]
    fn negative_or_nonfinite_stiffness_rejected() {
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let mut t = valid_type();
            t.lambda_volume = bad;
            assert!(t.validate().is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn positive_lambda_act_requires_positive_max_act() {
        let mut t = valid_type();
        t.lambda_act = 1.0;
        t.max_act = 0;
        assert!(t.validate().is_err());

        t.max_act = 1;
        assert!(t.validate().is_ok());
    }

    #[test]
    fn zero_lambda_act_allows_zero_max_act() {
        // The common case: Act inactive entirely (every existing test
        // fixture in the crate), lambda_act = 0 and max_act = 0 together.
        let t = valid_type();
        assert_eq!(t.lambda_act, 0.0);
        assert_eq!(t.max_act, 0);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn identity_and_type_are_independent() {
        // Two distinct cells can share a type — nothing in `Cell` ties
        // identity to type beyond the index.
        let a = Cell {
            id: 1,
            type_index: 0,
        };
        let b = Cell {
            id: 2,
            type_index: 0,
        };
        assert_ne!(a.id, b.id);
        assert_eq!(a.type_index, b.type_index);
    }
}
