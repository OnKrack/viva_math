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
  let result = free_energy.compute_state(state, state, state, 0.1)
  should.equal(result.feeling, free_energy.Homeostatic)
}

pub fn free_energy_alarmed_test() {
  // High prediction error should NOT be homeostatic
  let expected = Vec3(0.0, 0.0, 0.0)
  let actual = Vec3(1.0, 1.0, 0.0)
  let baseline = Vec3(0.0, 0.0, 0.0)
  let result = free_energy.compute_state(expected, actual, baseline, 0.1)
  // Distance squared = 2.0 + complexity ~= 2.2, should be Alarmed or Overwhelmed
  should.be_true(
    result.feeling == free_energy.Alarmed
    || result.feeling == free_energy.Overwhelmed,
  )
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
// Helper functions
// ============================================================================

fn is_close(a: Float, b: Float, tolerance: Float) -> Bool {
  float.absolute_value(a -. b) <=. tolerance
}
