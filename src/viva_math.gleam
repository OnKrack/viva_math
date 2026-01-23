//// viva_math - Core mathematical functions for VIVA.
////
//// A specialized math library for sentient digital life,
//// built on top of gleam_community_maths.
////
//// ## Modules
////
//// - `viva_math/common` - Utilities: clamp, sigmoid, softmax, lerp
//// - `viva_math/vector` - Vec3 type for PAD emotional space
//// - `viva_math/cusp` - Cusp catastrophe theory
//// - `viva_math/free_energy` - Free Energy Principle
//// - `viva_math/attractor` - Emotional attractor dynamics
//// - `viva_math/entropy` - Information theory
////
//// ## Dependencies
////
//// Re-exports gleam_community_maths for convenience.
//// Use `import gleam_community/maths` for:
//// - Trigonometry (sin, cos, tan, etc.)
//// - Statistics (mean, variance, etc.)
//// - Distances (euclidean, manhattan, etc.)
//// - Constants (pi, e, tau)
////
//// ## Example
////
//// ```gleam
//// import viva_math/vector
//// import viva_math/attractor
//// import viva_math/cusp
////
//// // Create PAD state
//// let state = vector.pad(-0.3, 0.7, -0.2)
////
//// // Classify emotion
//// let emotion = attractor.classify_emotion(state)
//// // -> "fear"
////
//// // Check for cusp bistability
//// let params = cusp.from_arousal_dominance(0.7, -0.2)
//// let volatile = cusp.is_bistable(params)
//// // -> True (high arousal creates bistability)
//// ```

// Re-export submodules for easy access
import viva_math/attractor
import viva_math/common
import viva_math/cusp
import viva_math/entropy
import viva_math/free_energy
import viva_math/vector

/// Library version
pub const version = "0.1.0"

/// Create a PAD vector with clamping.
/// Shorthand for vector.pad/3.
pub fn pad(pleasure: Float, arousal: Float, dominance: Float) -> vector.Vec3 {
  vector.pad(pleasure, arousal, dominance)
}

/// Classify emotional state to nearest attractor name.
/// Shorthand for attractor.classify_emotion/1.
pub fn classify(state: vector.Vec3) -> String {
  attractor.classify_emotion(state)
}

/// Check if emotional state is volatile (cusp bistability).
pub fn is_volatile(arousal: Float, dominance: Float) -> Bool {
  cusp.from_arousal_dominance(arousal, dominance)
  |> cusp.is_bistable
}

/// Compute free energy from expected and actual states.
pub fn free_energy(
  expected: vector.Vec3,
  actual: vector.Vec3,
) -> free_energy.FreeEnergyState {
  // Use neutral baseline and default complexity weight
  let baseline = vector.zero()
  let complexity_weight = 0.1
  free_energy.compute_state(expected, actual, baseline, complexity_weight)
}

/// Compute Shannon entropy of a probability distribution.
pub fn entropy(probabilities: List(Float)) -> Float {
  entropy.shannon(probabilities)
}

/// Standard sigmoid function.
pub fn sigmoid(x: Float) -> Float {
  common.sigmoid_standard(x)
}

/// Clamp value to [-1, 1] range.
pub fn clamp_bipolar(x: Float) -> Float {
  common.clamp_bipolar(x)
}
