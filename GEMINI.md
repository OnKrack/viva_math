# viva_math

**Core mathematical engine for VIVA (Virtual Intelligent Vida Autônoma).**

This library provides the foundational mathematical models required to simulate sentient digital life, specifically focusing on emotional dynamics, homeostasis, and decision-making stability. It bridges the gap between abstract psychological theories and executable functional code.

## 🧠 Architecture & Modules

The library is organized into specific mathematical domains, each corresponding to a key aspect of consciousness simulation:

### 1. Emotional Space (`viva_math/vector`)
- **PAD Model:** Implements the Pleasure-Arousal-Dominance 3D vector space (Mehrabian, 1996).
- **Core Type:** `Vec3` (Float, Float, Float).
- **Operations:** Vector arithmetic, normalization, distance calculations, and specialized clamping for emotional bounds `[-1, 1]`.

### 2. Phase Transitions (`viva_math/cusp`)
- **Catastrophe Theory:** Implements René Thom's Cusp Catastrophe model.
- **Purpose:** Models non-linear emotional shifts (e.g., sudden "snaps" from calm to anger).
- **Key Logic:**
  - `alpha` (normal factor) derived from negative **Arousal**.
  - `beta` (splitting factor) derived from **Dominance**.
  - High arousal creates a **bistable system** where the emotional state can "jump" between attractors.

### 3. Homeostasis (`viva_math/free_energy`)
- **FEP (Free Energy Principle):** Implements Karl Friston's theory of minimizing surprise.
- **Usage:** Agents seek to minimize the difference between their *expected* state and *actual* state (prediction error).
- **States:** `Homeostatic`, `Surprised`, `Alarmed`, `Overwhelmed`.

### 4. Emotional Stability (`viva_math/attractor`)
- **Dynamics:** Defines the 8 basic emotional attractors (Joy, Distress, Fear, Anger, etc.) in PAD space.
- **Basins of Attraction:** Calculates which emotion currently "pulls" the agent's state.

### 5. Information Theory (`viva_math/entropy`)
- **Metrics:** Shannon entropy, KL Divergence, Jensen-Shannon divergence.
- **Purpose:** Measures the complexity and novelty of inputs to drive curiosity.

## 🛠️ Development Workflow

This project uses the standard **Gleam** toolchain.

### Build
Compile the project to Erlang (or JavaScript) targets.
```sh
gleam build
```

### Test
Run the comprehensive test suite (powered by `gleeunit`).
```sh
gleam test
```

### Format
Ensure code adheres to the official Gleam style guide.
```sh
gleam format
```

## 📝 Coding Conventions

- **Functional Purity:** All functions are pure. State changes are modeled as transformations of data structures (`Vec3` -> `Vec3`).
- **Result Types:** Operations that can fail (e.g., division by zero, roots of negative numbers) return `Result(T, Nil)` or handle edge cases explicitly to avoid crashes.
- **Documentation:** Public functions include `///` comments explaining both the *what* (code behavior) and the *why* (theoretical background).
- **Maths Library:** Relies on `gleam_community_maths` for underlying primitives, keeping this library focused on the *application* of math to VIVA's domain.

## 🔗 Key References
- **PAD Model:** Mehrabian (1996)
- **Cusp Catastrophe:** Thom (1972), Grasman et al. (2009)
- **Free Energy Principle:** Friston (2010)
