//// Free Energy Principle (FEP) calculations.
////
//// Based on Karl Friston's work (2010, 2019).
//// Free Energy approximates surprise/entropy in the brain's predictions.
////
//// F ≈ Prediction_Error² + Complexity
////
//// In VIVA, this is used for interoception - sensing internal state
//// and minimizing "surprise" through prediction.
////
//// References:
//// - Friston (2010) "The free-energy principle: a unified brain theory?"
//// - Parr & Friston (2019) "Generalised free energy and active inference"

import gleam/list
import viva_math/vector.{type Vec3}

/// Free Energy state for a system.
pub type FreeEnergyState {
  FreeEnergyState(
    /// The free energy value (lower is better)
    free_energy: Float,
    /// Prediction error component
    prediction_error: Float,
    /// Complexity/prior divergence component
    complexity: Float,
    /// Qualitative feeling based on free energy level
    feeling: Feeling,
  )
}

/// Qualitative feeling based on free energy level.
pub type Feeling {
  /// Low free energy - predictions match reality
  Homeostatic
  /// Moderate free energy - slight mismatch
  Surprised
  /// High free energy - significant mismatch
  Alarmed
  /// Very high free energy - system overwhelmed
  Overwhelmed
}

/// Compute prediction error between expected and actual state.
///
/// Uses squared Euclidean distance for continuous states.
pub fn prediction_error(expected: Vec3, actual: Vec3) -> Float {
  vector.distance_squared(expected, actual)
}

/// Compute complexity term (prior divergence approximation).
///
/// In full FEP, this is KL divergence between approximate and true posterior.
/// Here we use a simplified version based on deviation from a baseline.
pub fn complexity(current: Vec3, baseline: Vec3, weight: Float) -> Float {
  weight *. vector.distance_squared(current, baseline)
}

/// Compute free energy: F = prediction_error + complexity
///
/// ## Parameters
/// - expected: predicted/expected state
/// - actual: observed/actual state
/// - baseline: prior baseline state (e.g., personality)
/// - complexity_weight: how much to weight complexity (default ~0.1)
pub fn free_energy(
  expected: Vec3,
  actual: Vec3,
  baseline: Vec3,
  complexity_weight: Float,
) -> Float {
  let pe = prediction_error(expected, actual)
  let cx = complexity(actual, baseline, complexity_weight)
  pe +. cx
}

/// Compute free energy and return full state with feeling.
pub fn compute_state(
  expected: Vec3,
  actual: Vec3,
  baseline: Vec3,
  complexity_weight: Float,
) -> FreeEnergyState {
  let pe = prediction_error(expected, actual)
  let cx = complexity(actual, baseline, complexity_weight)
  let fe = pe +. cx

  FreeEnergyState(
    free_energy: fe,
    prediction_error: pe,
    complexity: cx,
    feeling: classify_feeling(fe),
  )
}

/// Classify feeling based on free energy level.
///
/// Thresholds calibrated for PAD space (max distance ~3.46).
pub fn classify_feeling(free_energy: Float) -> Feeling {
  case free_energy {
    fe if fe <. 0.1 -> Homeostatic
    fe if fe <. 0.5 -> Surprised
    fe if fe <. 1.5 -> Alarmed
    _ -> Overwhelmed
  }
}

/// Compute surprise for a single dimension.
///
/// Surprise = -log(p(observation | model))
/// Using Gaussian approximation: surprise ∝ (x - μ)² / (2σ²)
pub fn surprise(expected: Float, observed: Float, sigma: Float) -> Float {
  let diff = observed -. expected
  let sigma_sq = sigma *. sigma
  case sigma_sq == 0.0 {
    True -> 0.0
    False -> { diff *. diff } /. { 2.0 *. sigma_sq }
  }
}

/// Active Inference: compute action that minimizes expected free energy.
///
/// This returns the delta to apply to current state to move toward target.
/// Rate controls how quickly to move (0 = no movement, 1 = instant).
pub fn active_inference_delta(
  current: Vec3,
  target: Vec3,
  rate: Float,
) -> Vec3 {
  let diff = vector.sub(target, current)
  vector.scale(diff, rate)
}

/// Precision-weighted prediction error.
///
/// Precision = 1/variance. Higher precision = more weight on that dimension.
/// Returns weighted sum of squared errors.
pub fn precision_weighted_error(
  expected: Vec3,
  actual: Vec3,
  precisions: Vec3,
) -> Float {
  let diff = vector.sub(expected, actual)
  let diff_sq = vector.multiply(diff, diff)
  let weighted = vector.multiply(diff_sq, precisions)
  vector.sum(weighted)
}

/// Estimate precision from recent prediction errors.
///
/// Precision = 1 / variance of errors
/// Higher precision means more reliable predictions.
pub fn estimate_precision(errors: List(Float)) -> Float {
  case list.length(errors) {
    0 -> 1.0
    // Default precision
    1 -> 1.0
    n -> {
      // Compute variance
      let n_float = int_to_float(n)
      let mean = list.fold(errors, 0.0, fn(acc, e) { acc +. e }) /. n_float
      let variance =
        list.fold(errors, 0.0, fn(acc, e) {
          let diff = e -. mean
          acc +. diff *. diff
        })
        /. n_float

      // Precision = 1/variance (with floor to avoid division by zero)
      case variance <. 0.001 {
        True -> 100.0
        // Very precise
        False -> 1.0 /. variance
      }
    }
  }
}

/// Bayesian belief update: combine prior with likelihood.
///
/// posterior ∝ likelihood × prior
/// Using precision-weighted combination:
/// new_belief = (precision_prior × prior + precision_likelihood × observation) /
///              (precision_prior + precision_likelihood)
pub fn belief_update(
  prior: Float,
  observation: Float,
  precision_prior: Float,
  precision_likelihood: Float,
) -> Float {
  let total_precision = precision_prior +. precision_likelihood
  case total_precision == 0.0 {
    True -> prior
    False ->
      { precision_prior *. prior +. precision_likelihood *. observation }
      /. total_precision
  }
}

/// Generalized Free Energy (expected free energy for planning).
///
/// G = ambiguity + risk
/// - ambiguity: expected surprise under model
/// - risk: KL divergence from preferred outcomes
///
/// Used for action selection in active inference.
pub fn generalized_free_energy(
  expected_state: Vec3,
  preferred_state: Vec3,
  uncertainty: Float,
) -> Float {
  // Ambiguity term (entropy of predictions)
  let ambiguity = uncertainty

  // Risk term (distance from preferences)
  let risk = vector.distance_squared(expected_state, preferred_state)

  ambiguity +. risk
}

// Helper: convert int to float
fn int_to_float(n: Int) -> Float {
  case n {
    0 -> 0.0
    1 -> 1.0
    2 -> 2.0
    3 -> 3.0
    4 -> 4.0
    5 -> 5.0
    _ -> {
      // Recursive for larger numbers
      let half = n / 2
      let remainder = n - half * 2
      int_to_float(half) *. 2.0 +. int_to_float(remainder)
    }
  }
}
