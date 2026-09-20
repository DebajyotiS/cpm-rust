//! The canonical `theta` layout: a stable, explicit mapping between the
//! inferrable model parameters and a flat `Vec<f64>`, plus the parallel
//! human-readable names Python needs to label posterior dimensions without
//! reimplementing this layout.
//!
//! `theta` covers the model parameters eligible for inference: per-type
//! `lambda_V`, per-type `lambda_I`, the `J` matrix, per-type `lambda_Act`
//! and `Max_Act`. Everything else on [`crate::config::ResolvedConfig`] —
//! grid, neighbourhoods, boundary, `V*`/`I*`, burn-in, seed, and so on — is
//! simulation control or calibration and must never end up in this vector.

use crate::config::ResolvedConfig;

#[derive(Debug, Clone, PartialEq)]
pub struct ThetaError(pub String);

impl std::fmt::Display for ThetaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ThetaError {}

/// Bumped whenever the layout itself changes (field order, inclusion, or
/// encoding) — independent of `n_types`, which changes the vector's length
/// but not its meaning. Belongs in run metadata alongside `convention_hash`;
/// `output.rs` assembles that bundle.
pub const THETA_LAYOUT_VERSION: u32 = 1;

const MEDIUM_LABEL: &str = "medium";

fn type_label<const D: usize>(config: &ResolvedConfig<D>, index: usize) -> String {
    if index == 0 {
        MEDIUM_LABEL.to_string()
    } else {
        config.cell_types[index - 1].name.clone()
    }
}

/// Length of the `theta` vector for `n_types` cell types (medium is index 0
/// of the adhesion matrix, not a counted "type").
pub fn theta_len(n_types: usize) -> usize {
    let adhesion_entries = (n_types + 1) * (n_types + 2) / 2; // upper triangular incl. medium row
    n_types /* lambda_V */
        + n_types /* lambda_I */
        + adhesion_entries /* J */
        + n_types /* lambda_Act */
        + n_types /* Max_Act */
}

/// Extract `theta` from a resolved configuration, in canonical order:
/// `[lambda_V[0..n], lambda_I[0..n], J[0][0..=n], J[1][1..=n], ...,
/// J[n][n], lambda_Act[0..n], Max_Act[0..n]]`.
pub fn extract<const D: usize>(config: &ResolvedConfig<D>) -> Vec<f64> {
    let n = config.cell_types.len();
    let mut theta = Vec::with_capacity(theta_len(n));

    for t in &config.cell_types {
        theta.push(t.lambda_volume);
    }
    for t in &config.cell_types {
        theta.push(t.lambda_interface);
    }
    for i in 0..=n {
        for j in i..=n {
            theta.push(config.adhesion[i][j]);
        }
    }
    for t in &config.cell_types {
        theta.push(t.lambda_act);
    }
    for t in &config.cell_types {
        theta.push(t.max_act as f64);
    }

    theta
}

/// Inject `theta` into an otherwise-fixed configuration: updates only the
/// fields `theta` covers, leaving grid, neighbourhoods, `V*`/`I*`,
/// `min_cell_volume` and everything else untouched — and therefore leaving
/// `convention_hash` untouched too, since none of those fields feed it.
///
/// `Max_Act` is stored as `u32` on [`crate::cell::CellType`] but travels
/// through `theta` as `f64`; injection rounds to the nearest integer. Rust's
/// `f64 -> u32` cast *saturates* rather than erroring, so an out-of-range
/// value (negative, NaN, or above `u32::MAX`) would otherwise be silently
/// remapped to a different, valid `Max_Act` instead of being rejected —
/// which corrupts the theta-to-simulation mapping without a visible error.
/// `Max_Act` values are validated before anything is mutated, so a rejected
/// `theta` leaves `config` unchanged.
pub fn inject<const D: usize>(
    config: &mut ResolvedConfig<D>,
    theta: &[f64],
) -> Result<(), ThetaError> {
    let n = config.cell_types.len();
    assert_eq!(
        theta.len(),
        theta_len(n),
        "theta has {} entries, expected {} for {n} cell types",
        theta.len(),
        theta_len(n)
    );

    let max_act_start = theta.len() - n;
    for (i, t) in config.cell_types.iter().enumerate() {
        let value = theta[max_act_start + i];
        let rounded = value.round();
        if !rounded.is_finite() || rounded < 0.0 || rounded > u32::MAX as f64 {
            return Err(ThetaError(format!(
                "max_act[{}] = {value} does not round to a value representable as u32 \
                 (must be finite and within [0, {}])",
                t.name,
                u32::MAX
            )));
        }
    }

    let mut cursor = 0usize;
    for t in &mut config.cell_types {
        t.lambda_volume = theta[cursor];
        cursor += 1;
    }
    for t in &mut config.cell_types {
        t.lambda_interface = theta[cursor];
        cursor += 1;
    }
    for i in 0..=n {
        for j in i..=n {
            let value = theta[cursor];
            cursor += 1;
            config.adhesion[i][j] = value;
            config.adhesion[j][i] = value; // adhesion is symmetric
        }
    }
    for t in &mut config.cell_types {
        t.lambda_act = theta[cursor];
        cursor += 1;
    }
    for t in &mut config.cell_types {
        t.max_act = theta[cursor].round() as u32;
        cursor += 1;
    }
    debug_assert_eq!(cursor, theta.len());
    Ok(())
}

/// Human-readable names in the same order as [`extract`], using declared
/// type names (and `"medium"` for adhesion row/column 0).
pub fn names<const D: usize>(config: &ResolvedConfig<D>) -> Vec<String> {
    let n = config.cell_types.len();
    let mut names = Vec::with_capacity(theta_len(n));

    for t in &config.cell_types {
        names.push(format!("lambda_volume[{}]", t.name));
    }
    for t in &config.cell_types {
        names.push(format!("lambda_interface[{}]", t.name));
    }
    for i in 0..=n {
        for j in i..=n {
            names.push(format!(
                "J[{}][{}]",
                type_label(config, i),
                type_label(config, j)
            ));
        }
    }
    for t in &config.cell_types {
        names.push(format!("lambda_act[{}]", t.name));
    }
    for t in &config.cell_types {
        names.push(format!("max_act[{}]", t.name));
    }

    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_support::two_type_config;

    #[test]
    fn length_matches_extracted_vector() {
        let config = two_type_config();
        let theta = extract(&config);
        assert_eq!(theta.len(), theta_len(config.cell_types.len()));
        assert_eq!(theta.len(), names(&config).len());
    }

    /// Build a config, extract theta, perturb every entry, inject, extract
    /// again, and assert the values match. Perturb by a whole number so the
    /// lossy `f64 -> u32` round trip on `Max_Act` is still exact.
    #[test]
    fn round_trip_after_perturbation() {
        let mut config = two_type_config();
        let original_hash = crate::config::convention_hash(&config);

        let theta = extract(&config);
        let perturbed: Vec<f64> = theta.iter().map(|v| v + 1.0).collect();
        inject(&mut config, &perturbed).unwrap();
        let recovered = extract(&config);

        assert_eq!(perturbed, recovered);

        // Injection must change only the intended fields — the convention
        // hash never depends on theta-eligible values.
        assert_eq!(original_hash, crate::config::convention_hash(&config));
    }

    #[test]
    fn injection_keeps_adhesion_symmetric() {
        let mut config = two_type_config();
        let mut theta = extract(&config);
        // J[0][1] is somewhere in the adhesion block; bump the first
        // adhesion entry and confirm both triangle halves move together.
        let adhesion_start = config.cell_types.len() * 2;
        theta[adhesion_start] += 5.0;
        inject(&mut config, &theta).unwrap();
        for i in 0..config.adhesion.len() {
            for j in 0..config.adhesion.len() {
                assert_eq!(config.adhesion[i][j], config.adhesion[j][i]);
            }
        }
    }

    /// A negative `Max_Act` entry must be rejected rather than silently
    /// saturating to `0` on the `f64 -> u32` cast — two different thetas
    /// (e.g. -1.0 and 0.0) must not map to the same simulation.
    #[test]
    fn negative_max_act_is_rejected_not_saturated() {
        let mut config = two_type_config();
        let mut theta = extract(&config);
        let max_act_start = theta.len() - config.cell_types.len();
        theta[max_act_start] = -1.0;
        let err = inject(&mut config, &theta).unwrap_err();
        assert!(err.0.contains("max_act"));
    }

    #[test]
    fn non_finite_max_act_is_rejected() {
        let mut config = two_type_config();
        let mut theta = extract(&config);
        let max_act_start = theta.len() - config.cell_types.len();
        theta[max_act_start] = f64::NAN;
        assert!(inject(&mut config, &theta).is_err());
    }

    /// A rejected `theta` must leave `config` untouched — injection is
    /// atomic, not partially applied up to the point of failure.
    #[test]
    fn rejected_injection_leaves_config_unchanged() {
        let mut config = two_type_config();
        let before = config.clone();
        let mut theta = extract(&config);
        for v in theta.iter_mut() {
            *v += 1.0;
        }
        let max_act_start = theta.len() - config.cell_types.len();
        theta[max_act_start] = -1.0;
        assert!(inject(&mut config, &theta).is_err());
        assert_eq!(config.cell_types, before.cell_types);
        assert_eq!(config.adhesion, before.adhesion);
    }

    #[test]
    fn names_use_declared_type_labels() {
        let config = two_type_config();
        let names = names(&config);
        assert!(names.contains(&"lambda_volume[epithelial]".to_string()));
        assert!(names.contains(&"J[medium][epithelial]".to_string()));
        assert!(names.contains(&"J[epithelial][stem]".to_string()));
    }
}
