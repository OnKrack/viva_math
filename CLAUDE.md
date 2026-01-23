# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**viva_math** is a mathematical library in Gleam for modeling emotional dynamics in sentient digital entities (VIVA project). It implements four scientific frameworks:

1. **PAD Model** - Pleasure-Arousal-Dominance 3D emotional space [-1, 1]³
2. **Cusp Catastrophe Theory** - Abrupt emotional transitions (Thom, 1972)
3. **Free Energy Principle** - Prediction error minimization (Friston, 2010)
4. **Attractor Dynamics** - Emotions as basins of attraction (Mehrabian, 1996)

## Commands

```bash
gleam build           # Compile
gleam test            # Run all tests
gleam format          # Format code
gleam deps download   # Download dependencies
```

## Architecture

```
src/viva_math/
├── common.gleam      # Utilities: clamp, lerp, sigmoid, softmax, safe_div
├── vector.gleam      # Vec3 type + operations for PAD space
├── cusp.gleam        # Catastrophe potential, gradients, bistability detection
├── free_energy.gleam # FEP: prediction_error, surprise, active inference
├── attractor.gleam   # 8 emotional attractors, basin weights, OU mean-reversion
├── entropy.gleam     # Shannon, KL divergence, Jensen-Shannon, cross-entropy
└── viva_math.gleam   # Root module with re-exports and shorthand helpers
```

**Key types:**
- `Vec3` - 3D vector for PAD coordinates (all emotions map to this)
- `CuspParams` - alpha (arousal-derived) and beta (dominance-derived)
- `FreeEnergyState` - Homeostatic | Surprised | Alarmed | Overwhelmed
- `Emotion` - 8 variants: Joy, Excitement, Trust, Serenity, Sadness, Fear, Anger, Disgust

**Data flow:** PAD vector → cusp params → catastrophe dynamics → emotion classification

## Testing

Tests in `test/viva_math_test.gleam` use gleeunit. Helper `is_close()` handles floating-point comparison with tolerance.

Test sections mirror module structure: common, vector, cusp, free_energy, attractor, entropy.

## Code Patterns

- Pure functional, no side effects
- `Result` for fallible operations (division by zero, invalid inputs)
- Doc comments use `////` with examples
- Numeric stability: clamp in trigonometric ops, max-subtraction in softmax

## Dependencies

- `gleam_stdlib` - Standard library
- `gleam_community_maths` - Trigonometry, logarithms, constants
- `gleeunit` (dev) - Test framework

## Commit Convention

```
feat: new functionality
fix: bug fix
docs: documentation
test: tests
refactor: code restructuring
chore: maintenance
```
