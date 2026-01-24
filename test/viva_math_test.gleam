import gleam/float
import gleam/list
import gleeunit
import gleeunit/should
import viva_math/attractor
import viva_math/common
import viva_math/cusp
import viva_math/entropy
import viva_math/free_energy
import viva_math/vector.{Vec3}

pub fn main() {
  gleeunit.main()
}

// ============================================================================
// common.gleam tests
// ============================================================================

pub fn clamp_test() {
  common.clamp(5.0, 0.0, 10.0)
  |> should.equal(5.0)

  common.clamp(-1.0, 0.0, 10.0)
  |> should.equal(0.0)

  common.clamp(15.0, 0.0, 10.0)
  |> should.equal(10.0)
}

pub fn clamp_unit_test() {
  common.clamp_unit(0.5)
  |> should.equal(0.5)

  common.clamp_unit(-0.5)
  |> should.equal(0.0)

  common.clamp_unit(1.5)
  |> should.equal(1.0)
}

pub fn clamp_bipolar_test() {
  common.clamp_bipolar(0.5)
  |> should.equal(0.5)

  common.clamp_bipolar(-1.5)
  |> should.equal(-1.0)

  common.clamp_bipolar(1.5)
  |> should.equal(1.0)
}

pub fn lerp_test() {
  common.lerp(0.0, 10.0, 0.5)
  |> should.equal(5.0)

  common.lerp(0.0, 10.0, 0.0)
  |> should.equal(0.0)

  common.lerp(0.0, 10.0, 1.0)
  |> should.equal(10.0)
}

pub fn sigmoid_center_test() {
  // sigmoid(0) should be 0.5
  let result = common.sigmoid(0.0, 1.0)
  should.be_true(is_close(result, 0.5, 0.001))
}

pub fn sigmoid_extremes_test() {
  // sigmoid(-100) should be ~0
  let low = common.sigmoid(-100.0, 1.0)
  should.be_true(low <. 0.001)

  // sigmoid(100) should be ~1
  let high = common.sigmoid(100.0, 1.0)
  should.be_true(high >. 0.999)
}

pub fn softmax_test() {
  // Equal inputs should give equal outputs
  let assert Ok(result) = common.softmax([1.0, 1.0])
  case result {
    [a, b] -> {
      should.be_true(is_close(a, 0.5, 0.001))
      should.be_true(is_close(b, 0.5, 0.001))
    }
    _ -> should.fail()
  }
}

pub fn softmax_sum_to_one_test() {
  let assert Ok(result) = common.softmax([1.0, 2.0, 3.0])
  let sum = list.fold(result, 0.0, fn(acc, x) { acc +. x })
  should.be_true(is_close(sum, 1.0, 0.001))
}

pub fn safe_div_test() {
  common.safe_div(10.0, 2.0, 0.0)
  |> should.equal(5.0)

  common.safe_div(10.0, 0.0, -1.0)
  |> should.equal(-1.0)
}

// ============================================================================
// vector.gleam tests
// ============================================================================

pub fn vec3_zero_test() {
  vector.zero()
  |> should.equal(Vec3(0.0, 0.0, 0.0))
}

pub fn vec3_add_test() {
  let a = Vec3(1.0, 2.0, 3.0)
  let b = Vec3(4.0, 5.0, 6.0)
  vector.add(a, b)
  |> should.equal(Vec3(5.0, 7.0, 9.0))
}

pub fn vec3_sub_test() {
  let a = Vec3(5.0, 7.0, 9.0)
  let b = Vec3(1.0, 2.0, 3.0)
  vector.sub(a, b)
  |> should.equal(Vec3(4.0, 5.0, 6.0))
}

pub fn vec3_scale_test() {
  let v = Vec3(1.0, 2.0, 3.0)
  vector.scale(v, 2.0)
  |> should.equal(Vec3(2.0, 4.0, 6.0))
}

pub fn vec3_dot_test() {
  let a = Vec3(1.0, 2.0, 3.0)
  let b = Vec3(4.0, 5.0, 6.0)
  vector.dot(a, b)
  |> should.equal(32.0)
  // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

pub fn vec3_length_test() {
  let v = Vec3(3.0, 4.0, 0.0)
  let len = vector.length(v)
  should.be_true(is_close(len, 5.0, 0.001))
}

pub fn vec3_distance_test() {
  let a = Vec3(0.0, 0.0, 0.0)
  let b = Vec3(1.0, 1.0, 1.0)
  let dist = vector.distance(a, b)
  // sqrt(3) ≈ 1.732
  should.be_true(is_close(dist, 1.732, 0.01))
}

pub fn vec3_normalize_test() {
  let v = Vec3(3.0, 0.0, 0.0)
  vector.normalize(v)
  |> should.equal(Vec3(1.0, 0.0, 0.0))
}

pub fn vec3_clamp_pad_test() {
  let v = Vec3(2.0, -2.0, 0.5)
  vector.clamp_pad(v)
  |> should.equal(Vec3(1.0, -1.0, 0.5))
}

pub fn vec3_pad_test() {
  let v = vector.pad(0.5, -0.3, 0.8)
  should.equal(v.x, 0.5)
  should.equal(v.y, -0.3)
  should.equal(v.z, 0.8)
}

// ============================================================================
// cusp.gleam tests
// ============================================================================

pub fn cusp_potential_at_zero_test() {
  // V(0) = 0^4/4 + α*0^2/2 + β*0 = 0
  let params = cusp.CuspParams(-1.0, 0.0)
  cusp.potential(0.0, params)
  |> should.equal(0.0)
}

pub fn cusp_gradient_at_zero_test() {
  // dV/dx(0) = 0^3 + α*0 + β = β
  let params = cusp.CuspParams(-1.0, 0.5)
  cusp.gradient(0.0, params)
  |> should.equal(0.5)
}

pub fn cusp_is_bistable_test() {
  // α = -1 < 0, Δ = -4*(-1)^3 - 27*0^2 = 4 > 0 → bistable
  let params = cusp.CuspParams(-1.0, 0.0)
  cusp.is_bistable(params)
  |> should.be_true()
}

pub fn cusp_is_monostable_test() {
  // α = 1 ≥ 0 → monostable regardless of β
  let params = cusp.CuspParams(1.0, 0.0)
  cusp.is_bistable(params)
  |> should.be_false()
}

pub fn cusp_from_arousal_test() {
  // High arousal (0.8) should give negative alpha → potential bistability
  let params = cusp.from_arousal_dominance(0.8, 0.0)
  should.be_true(params.alpha <. 0.0)
}

pub fn cusp_equilibria_monostable_test() {
  // With α > 0, should be monostable
  let params = cusp.CuspParams(1.0, 0.0)
  case cusp.equilibria(params) {
    cusp.Monostable(_) -> should.be_true(True)
    cusp.Bistable(_, _, _) -> should.fail()
  }
}

pub fn cusp_equilibria_bistable_test() {
  // With α = -1, β = 0, should be bistable
  let params = cusp.CuspParams(-1.0, 0.0)
  case cusp.equilibria(params) {
    cusp.Bistable(_, _, _) -> should.be_true(True)
    cusp.Monostable(_) -> should.fail()
  }
}

// ============================================================================
// free_energy.gleam tests
// ============================================================================

pub fn prediction_error_zero_test() {
  // Same state → zero error
  let state = Vec3(0.5, 0.3, -0.2)
  free_energy.prediction_error(state, state)
  |> should.equal(0.0)
}

pub fn prediction_error_nonzero_test() {
  let expected = Vec3(0.0, 0.0, 0.0)
  let actual = Vec3(1.0, 0.0, 0.0)
  // Squared distance = 1.0
  free_energy.prediction_error(expected, actual)
  |> should.equal(1.0)
}

pub fn free_energy_homeostatic_test() {
  // Low free energy should be homeostatic
  let state = Vec3(0.0, 0.0, 0.0)
  let result = free_energy.compute_state_simple(state, state, state, 0.1)
  should.equal(result.feeling, free_energy.Homeostatic)
}

pub fn free_energy_alarmed_test() {
  // High prediction error should NOT be homeostatic
  let expected = Vec3(0.0, 0.0, 0.0)
  let actual = Vec3(1.0, 1.0, 0.0)
  let baseline = Vec3(0.0, 0.0, 0.0)
  let result = free_energy.compute_state_simple(expected, actual, baseline, 0.1)
  // Distance squared = 2.0 + complexity ~= 2.2, should be Alarmed or Overwhelmed
  should.be_true(
    result.feeling == free_energy.Alarmed
    || result.feeling == free_energy.Overwhelmed,
  )
}

pub fn free_energy_precision_weighted_test() {
  // Higher precision should amplify prediction error
  let expected = Vec3(0.0, 0.0, 0.0)
  let actual = Vec3(1.0, 0.0, 0.0)

  let low_precision = free_energy.precision_weighted_prediction_error(expected, actual, 0.5)
  let high_precision = free_energy.precision_weighted_prediction_error(expected, actual, 2.0)

  should.be_true(high_precision >. low_precision)
  should.be_true(is_close(low_precision, 0.5, 0.001))
  should.be_true(is_close(high_precision, 2.0, 0.001))
}

pub fn free_energy_gaussian_kl_test() {
  // KL divergence between same distributions is 0
  let state = Vec3(0.5, 0.3, -0.2)
  let kl = free_energy.gaussian_kl_divergence(state, state, 1.0)
  should.be_true(is_close(kl, 0.0, 0.001))
}

pub fn free_energy_normalized_thresholds_test() {
  // Test normalized threshold classification
  let thresholds = free_energy.FeelingThresholds(mean: 1.0, std_dev: 0.5)

  // F < μ - σ = 0.5 → Homeostatic
  should.equal(free_energy.classify_feeling_normalized(0.3, thresholds), free_energy.Homeostatic)

  // μ - σ ≤ F < μ → Surprised
  should.equal(free_energy.classify_feeling_normalized(0.7, thresholds), free_energy.Surprised)

  // μ ≤ F < μ + σ → Alarmed
  should.equal(free_energy.classify_feeling_normalized(1.2, thresholds), free_energy.Alarmed)

  // F ≥ μ + σ → Overwhelmed
  should.equal(free_energy.classify_feeling_normalized(2.0, thresholds), free_energy.Overwhelmed)
}

// ============================================================================
// attractor.gleam tests
// ============================================================================

pub fn attractor_classify_joy_test() {
  // Point near joy attractor
  let state = Vec3(0.7, 0.5, 0.3)
  attractor.classify_emotion(state)
  |> should.equal("joy")
}

pub fn attractor_classify_sadness_test() {
  // Point near sadness attractor
  let state = Vec3(-0.6, -0.3, -0.3)
  attractor.classify_emotion(state)
  |> should.equal("sadness")
}

pub fn attractor_classify_fear_test() {
  // Point near fear attractor
  let state = Vec3(-0.6, 0.6, -0.4)
  attractor.classify_emotion(state)
  |> should.equal("fear")
}

pub fn attractor_nearest_test() {
  let attractors = attractor.emotional_attractors()
  let point = Vec3(0.76, 0.48, 0.35)
  // Exactly at joy
  let assert Ok(nearest) = attractor.nearest(point, attractors)
  should.equal(nearest.name, "joy")
}

pub fn attractor_basin_weights_sum_test() {
  let attractors = attractor.emotional_attractors()
  let point = Vec3(0.0, 0.0, 0.0)
  let weights = attractor.basin_weights(point, attractors, 1.0)
  let sum =
    list.fold(weights, 0.0, fn(acc, pair) {
      let #(_, w) = pair
      acc +. w
    })
  // Weights should sum to ~1.0
  should.be_true(is_close(sum, 1.0, 0.01))
}

// ============================================================================
// entropy.gleam tests
// ============================================================================

pub fn entropy_uniform_test() {
  // Uniform distribution [0.5, 0.5] has entropy = 1 bit
  let h = entropy.shannon([0.5, 0.5])
  should.be_true(is_close(h, 1.0, 0.001))
}

pub fn entropy_certain_test() {
  // Certain outcome [1.0, 0.0] has entropy = 0
  let h = entropy.shannon([1.0, 0.0])
  should.be_true(is_close(h, 0.0, 0.001))
}

pub fn entropy_four_uniform_test() {
  // Uniform [0.25, 0.25, 0.25, 0.25] has entropy = 2 bits
  let h = entropy.shannon([0.25, 0.25, 0.25, 0.25])
  should.be_true(is_close(h, 2.0, 0.001))
}

pub fn kl_divergence_same_test() {
  // KL divergence of identical distributions is 0
  let p = [0.5, 0.5]
  let assert Ok(kl) = entropy.kl_divergence(p, p)
  should.be_true(is_close(kl, 0.0, 0.001))
}

pub fn kl_divergence_different_test() {
  // KL divergence of different distributions is positive
  let p = [0.9, 0.1]
  let q = [0.5, 0.5]
  let assert Ok(kl) = entropy.kl_divergence(p, q)
  should.be_true(kl >. 0.0)
}

pub fn jensen_shannon_symmetric_test() {
  // JS divergence should be symmetric
  let p = [0.9, 0.1]
  let q = [0.5, 0.5]
  let assert Ok(js_pq) = entropy.jensen_shannon(p, q)
  let assert Ok(js_qp) = entropy.jensen_shannon(q, p)
  should.be_true(is_close(js_pq, js_qp, 0.001))
}

// ============================================================================
// NEW: Stochastic cusp tests (DeepSeek R1 proposals)
// ============================================================================

pub fn stochastic_cusp_deterministic_test() {
  // Same seed should produce same noise
  let noise1 = common.deterministic_noise(0, 42)
  let noise2 = common.deterministic_noise(0, 42)
  should.be_true(is_close(noise1, noise2, 0.0001))
}

pub fn stochastic_cusp_different_steps_test() {
  // Different steps should produce different noise
  let noise1 = common.deterministic_noise(0, 42)
  let noise2 = common.deterministic_noise(1, 42)
  should.be_false(is_close(noise1, noise2, 0.0001))
}

pub fn stochastic_cusp_range_test() {
  // Noise should be in [-1, 1]
  let noise = common.deterministic_noise(100, 999)
  should.be_true(noise >=. -1.0 && noise <=. 1.0)
}

pub fn stochastic_simulation_length_test() {
  // Simulation should return correct number of steps
  let params = cusp.StochasticCuspParams(alpha: -1.0, beta: 0.0, sigma: 0.1, seed: 42)
  let trajectory = cusp.simulate_stochastic(0.0, params, 0.01, 10)
  should.equal(list.length(trajectory), 11)  // initial + 10 steps
}

// ============================================================================
// NEW: Basin weights with exp(-γd) tests
// ============================================================================

pub fn basin_weights_exp_sum_to_one_test() {
  // Basin weights should sum to 1.0
  let attractors = attractor.emotional_attractors()
  let point = vector.Vec3(0.0, 0.0, 0.0)
  let weights = attractor.basin_weights(point, attractors, 1.0)
  let sum = list.fold(weights, 0.0, fn(acc, pair) { acc +. pair.1 })
  should.be_true(is_close(sum, 1.0, 0.01))
}

pub fn basin_weights_temperature_effect_test() {
  // Lower temperature should make weights sharper (max weight higher)
  let attractors = attractor.emotional_attractors()
  let point = vector.Vec3(0.7, 0.4, 0.3)  // Near joy

  let weights_warm = attractor.basin_weights(point, attractors, 2.0)
  let weights_cold = attractor.basin_weights(point, attractors, 0.5)

  let max_warm = list.fold(weights_warm, 0.0, fn(acc, p) { float.max(acc, p.1) })
  let max_cold = list.fold(weights_cold, 0.0, fn(acc, p) { float.max(acc, p.1) })

  // Cold (low temp) should have higher max weight
  should.be_true(max_cold >. max_warm)
}

// ============================================================================
// NEW: Hybrid entropy tests
// ============================================================================

pub fn hybrid_entropy_blend_test() {
  // Blend of two distributions
  let p1 = [0.5, 0.5]  // H = 1.0
  let p2 = [1.0, 0.0]  // H = 0.0

  let h_blend = entropy.hybrid_shannon(p1, p2, 0.5)
  // Should be average: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
  should.be_true(is_close(h_blend, 0.5, 0.01))
}

pub fn hybrid_entropy_alpha_zero_test() {
  // Alpha = 0 should give H(p2)
  let p1 = [0.5, 0.5]  // H = 1.0
  let p2 = [1.0, 0.0]  // H = 0.0

  let h = entropy.hybrid_shannon(p1, p2, 0.0)
  should.be_true(is_close(h, 0.0, 0.01))
}

pub fn hybrid_entropy_alpha_one_test() {
  // Alpha = 1 should give H(p1)
  let p1 = [0.5, 0.5]  // H = 1.0
  let p2 = [1.0, 0.0]  // H = 0.0

  let h = entropy.hybrid_shannon(p1, p2, 1.0)
  should.be_true(is_close(h, 1.0, 0.01))
}

// ============================================================================
// NEW: KL with sensitivity tests
// ============================================================================

pub fn kl_sensitivity_standard_test() {
  // Standard sensitivity should match regular KL
  let p = [0.5, 0.5]
  let q = [0.6, 0.4]

  let assert Ok(kl_standard) = entropy.kl_divergence(p, q)
  let assert Ok(kl_sens) = entropy.kl_divergence_with_sensitivity(p, q, entropy.Standard)

  should.be_true(is_close(kl_standard, kl_sens, 0.001))
}

pub fn kl_sensitivity_arousal_increases_test() {
  // Higher arousal should increase KL (more sensitive)
  let p = [0.9, 0.1]
  let q = [0.5, 0.5]

  let assert Ok(kl_low) = entropy.kl_divergence_with_sensitivity(p, q, entropy.ArousalWeighted(0.2))
  let assert Ok(kl_high) = entropy.kl_divergence_with_sensitivity(p, q, entropy.ArousalWeighted(0.8))

  should.be_true(kl_high >. kl_low)
}

// ============================================================================
// NEW: Renyi entropy tests
// ============================================================================

pub fn renyi_order_one_is_shannon_test() {
  // Renyi entropy with α=1 should equal Shannon entropy
  let p = [0.5, 0.5]
  let assert Ok(h_renyi) = entropy.renyi(p, 1.0)
  let h_shannon = entropy.shannon(p)
  should.be_true(is_close(h_renyi, h_shannon, 0.001))
}

pub fn renyi_order_two_collision_test() {
  // Renyi entropy with α=2 (collision entropy)
  // For uniform [0.5, 0.5]: H_2 = -log2(0.5² + 0.5²) = -log2(0.5) = 1.0
  let p = [0.5, 0.5]
  let assert Ok(h2) = entropy.renyi(p, 2.0)
  should.be_true(is_close(h2, 1.0, 0.01))
}

// ============================================================================
// NEW: Full KL divergence tests
// ============================================================================

pub fn gaussian_kl_full_equal_variance_test() {
  // When variances are equal, full KL should reduce to simple form
  let mean1 = vector.Vec3(0.5, 0.0, 0.0)
  let mean2 = vector.Vec3(0.0, 0.0, 0.0)

  let kl_simple = free_energy.gaussian_kl_divergence(mean1, mean2, 1.0)
  let kl_full = free_energy.gaussian_kl_divergence_full(mean1, mean2, 1.0, 1.0)

  // Full KL with equal variances = log(1) + (σ² + d²)/(2σ²) - 0.5
  // = 0 + (1 + 0.25)/2 - 0.5 = 0.625 - 0.5 = 0.125
  // Simple KL = d²/(2σ²) = 0.25/2 = 0.125
  should.be_true(is_close(kl_simple, kl_full, 0.01))
}

// ============================================================================
// Helper functions
// ============================================================================

fn is_close(a: Float, b: Float, tolerance: Float) -> Bool {
  float.absolute_value(a -. b) <=. tolerance
}
