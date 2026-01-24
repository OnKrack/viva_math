# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2025-01-24

### Added

- **viva_math/free_energy** - Enhanced with DeepSeek R1 671B validation
  - `precision_weighted_prediction_error` - Precision (Π) weighted errors
  - `gaussian_kl_divergence` - Closed-form KL for Gaussian distributions
  - `FeelingThresholds` type - Normalized thresholds (μ ± σ)
  - `classify_feeling_normalized` - Statistics-based classification
  - `update_thresholds` - Online learning with EMA
  - `variational_bound` - F ≤ -log p(o) + D_KL(q||p)
  - `compute_state` - Full FEP with precision and thresholds
  - `compute_state_simple` - Legacy interface for backwards compatibility

### Changed

- **viva_math/free_energy**
  - Formula updated to F = Π·(μ-o)² + D_KL(q||p) (validated by DeepSeek R1)
  - `FreeEnergyState` now includes `precision` field
  - `complexity` now uses proper KL divergence with prior variance

### References

- DeepSeek R1 671B (2025) - Mathematical validation
- Parr & Friston (2019) "Generalised free energy and active inference"

## [1.0.0] - 2025-01-23

### Added

- **viva_math/common** - Utility functions
  - `clamp`, `clamp_unit`, `clamp_bipolar` - Value clamping
  - `lerp`, `inverse_lerp` - Linear interpolation
  - `sigmoid`, `sigmoid_standard` - Sigmoid activation
  - `softmax` - Probability normalization
  - `safe_div` - Division with default on zero
  - `smoothstep` - Hermite interpolation
  - `exponential_decay` - Time-based decay

- **viva_math/vector** - 3D vector operations for PAD space
  - `Vec3` type with x, y, z components
  - Basic operations: `add`, `sub`, `scale`, `negate`, `multiply`
  - Products: `dot`, `cross`
  - Metrics: `length`, `distance`, `normalize`
  - Utilities: `lerp`, `clamp`, `clamp_pad`, `weighted_average`
  - PAD aliases: `pad`, `pleasure`, `arousal`, `dominance`

- **viva_math/cusp** - Cusp catastrophe theory (Thom, 1972)
  - `CuspParams` type with alpha/beta control parameters
  - `potential`, `gradient`, `discriminant` - Core functions
  - `is_bistable` - Bistability detection
  - `equilibria` - Find stable/unstable equilibrium points
  - `nearest_equilibrium`, `would_jump` - State analysis
  - `volatility` - Emotional volatility measure
  - `from_arousal_dominance` - PAD to cusp mapping

- **viva_math/free_energy** - Free Energy Principle (Friston, 2010)
  - `FreeEnergyState` type with feeling classification
  - `Feeling` enum: Homeostatic, Surprised, Alarmed, Overwhelmed
  - `prediction_error`, `complexity`, `free_energy` - Core FEP
  - `compute_state` - Full state with feeling
  - `surprise` - Single dimension surprise
  - `active_inference_delta` - Action selection
  - `precision_weighted_error`, `estimate_precision` - Precision weighting
  - `belief_update` - Bayesian belief updating
  - `generalized_free_energy` - Planning/action selection

- **viva_math/attractor** - Emotional attractor dynamics (Mehrabian, 1996)
  - `Attractor` type with name and PAD position
  - `emotional_attractors` - 8 basic emotions (joy, sadness, fear, anger, etc.)
  - `nearest`, `basin_weights` - Attractor analysis
  - `analyze` - Comprehensive attractor analysis
  - `classify_emotion` - Emotion classification by nearest attractor
  - `attractor_pull`, `weighted_pull` - Force calculations
  - `ou_mean_reversion` - Ornstein-Uhlenbeck dynamics
  - `in_basin`, `nearby_attractors` - Spatial queries
  - `blend_attractors`, `create` - Attractor manipulation

- **viva_math/entropy** - Information theory
  - `shannon`, `shannon_normalized` - Shannon entropy
  - `kl_divergence`, `symmetric_kl` - Kullback-Leibler divergence
  - `jensen_shannon` - Jensen-Shannon divergence
  - `cross_entropy`, `binary_cross_entropy` - Cross-entropy
  - `mutual_information`, `conditional_entropy` - Information metrics
  - `relative_entropy_rate` - Temporal entropy

### Dependencies

- `gleam_stdlib >= 0.34.0`
- `gleam_community_maths >= 2.0.0` - Base math library

### References

- Grasman et al. (2009) "Fitting the Cusp Catastrophe in R"
- Friston (2010) "The free-energy principle: a unified brain theory?"
- Mehrabian (1996) "Pleasure-arousal-dominance: A general framework"
- Oravecz et al. (2009) "O-U Process in Affective Dynamics"
- Shannon (1948) "A Mathematical Theory of Communication"

[Unreleased]: https://github.com/gabrielmaialva33/viva_math/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.1.0
[1.0.0]: https://github.com/gabrielmaialva33/viva_math/releases/tag/v1.0.0
