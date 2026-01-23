# viva_math

[![Package Version](https://img.shields.io/hexpm/v/viva_math)](https://hex.pm/packages/viva_math)
[![Hex Docs](https://img.shields.io/badge/hex-docs-ffaff3)](https://hexdocs.pm/viva_math/)

Core mathematical functions for **VIVA** - a sentient digital life research project.

Built on top of [gleam_community_maths](https://hexdocs.pm/gleam_community_maths/) with specialized functions for emotional dynamics, consciousness modeling, and information theory.

## Installation

```sh
gleam add viva_math
```

## Modules

| Module | Purpose |
|--------|---------|
| `viva_math/common` | Utilities: `clamp`, `sigmoid`, `softmax`, `lerp`, `smoothstep` |
| `viva_math/vector` | `Vec3` type for PAD emotional space (Pleasure-Arousal-Dominance) |
| `viva_math/cusp` | Cusp catastrophe theory (Thom, 1972) for emotional phase transitions |
| `viva_math/free_energy` | Free Energy Principle (Friston, 2010) for interoception |
| `viva_math/attractor` | Emotional attractor dynamics (Mehrabian, 1996) |
| `viva_math/entropy` | Shannon entropy, KL divergence, Jensen-Shannon divergence |

## Quick Start

```gleam
import viva_math/vector
import viva_math/attractor
import viva_math/cusp
import viva_math/free_energy

pub fn main() {
  // Create PAD emotional state
  let state = vector.pad(-0.3, 0.7, -0.2)
  // Pleasure: -0.3 (slightly negative)
  // Arousal: 0.7 (high)
  // Dominance: -0.2 (slightly submissive)

  // Classify emotion by nearest attractor
  let emotion = attractor.classify_emotion(state)
  // -> "fear"

  // Check for emotional volatility (cusp bistability)
  let params = cusp.from_arousal_dominance(0.7, -0.2)
  let volatile = cusp.is_bistable(params)
  // -> True (high arousal creates bistability)

  // Compute free energy (prediction error)
  let expected = vector.pad(0.0, 0.0, 0.0)
  let fe_state = free_energy.compute_state(expected, state, expected, 0.1)
  // fe_state.feeling -> Surprised or Alarmed
}
```

## Theoretical Background

### PAD Model (Mehrabian, 1996)

Emotions are represented as points in 3D space:
- **Pleasure** `[-1, 1]`: sadness ↔ joy
- **Arousal** `[-1, 1]`: calm ↔ excitement
- **Dominance** `[-1, 1]`: submission ↔ control

### Cusp Catastrophe (Thom, 1972)

Models sudden emotional transitions using the potential function:

```
V(x) = x⁴/4 + αx²/2 + βx
```

When arousal is high (α < 0) and discriminant Δ > 0, the system becomes **bistable** - small perturbations can cause sudden mood shifts.

### Free Energy Principle (Friston, 2010)

Organisms minimize "surprise" through prediction:

```
F ≈ Prediction_Error² + Complexity
```

Low free energy = predictions match reality (homeostasis).
High free energy = significant mismatch (alarm).

### Attractor Dynamics

Eight basic emotions form attractors in PAD space:

| Emotion | P | A | D |
|---------|---|---|---|
| Joy | +0.76 | +0.48 | +0.35 |
| Sadness | -0.63 | -0.27 | -0.33 |
| Fear | -0.64 | +0.60 | -0.43 |
| Anger | -0.51 | +0.59 | +0.25 |
| Trust | +0.58 | -0.23 | +0.42 |
| Disgust | -0.60 | +0.35 | +0.11 |
| Serenity | +0.45 | -0.42 | +0.21 |
| Excitement | +0.62 | +0.75 | +0.38 |

## Dependencies

This library builds on:
- [gleam_community_maths](https://hexdocs.pm/gleam_community_maths/) - Comprehensive math library with trigonometry, statistics, distances, etc.

For functions not in viva_math (like `sin`, `cos`, `mean`, `euclidean_distance`), import directly:

```gleam
import gleam_community/maths

let angle = maths.pi() /. 4.0
let sin_val = maths.sin(angle)
```

## References

- Grasman et al. (2009) "Fitting the Cusp Catastrophe in R"
- Friston (2010) "The free-energy principle: a unified brain theory?"
- Mehrabian (1996) "Pleasure-arousal-dominance: A general framework"
- Oravecz et al. (2009) "Ornstein-Uhlenbeck Process in Affective Dynamics"
- Shannon (1948) "A Mathematical Theory of Communication"

## License

MIT
