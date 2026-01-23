//// Entropy and information theory functions.
////
//// Based on Shannon (1948) and Kullback-Leibler (1951).
//// Used for memory consolidation scoring and uncertainty quantification.
////
//// References:
//// - Shannon (1948) "A Mathematical Theory of Communication"
//// - Cover & Thomas (2006) "Elements of Information Theory"

import gleam/float
import gleam/list
import gleam_community/maths

/// Shannon entropy: H(X) = -Σ p(x) log₂ p(x)
///
/// Measures uncertainty/information content of a distribution.
/// Higher entropy = more uncertainty.
///
/// ## Examples
///
/// ```gleam
/// shannon([0.5, 0.5])     // -> 1.0 (maximum for 2 outcomes)
/// shannon([1.0, 0.0])     // -> 0.0 (no uncertainty)
/// shannon([0.25, 0.25, 0.25, 0.25]) // -> 2.0
/// ```
pub fn shannon(probabilities: List(Float)) -> Float {
  list.fold(probabilities, 0.0, fn(acc, p) {
    case p <=. 0.0 {
      True -> acc
      // 0 * log(0) = 0 by convention
      False ->
        case maths.logarithm_2(p) {
          Ok(log_p) -> acc -. { p *. log_p }
          Error(_) -> acc
        }
    }
  })
}

/// Normalized Shannon entropy (0 to 1 range).
///
/// Divides by log₂(n) where n is number of outcomes.
pub fn shannon_normalized(probabilities: List(Float)) -> Float {
  let n = list.length(probabilities)
  case n <= 1 {
    True -> 0.0
    False -> {
      let h = shannon(probabilities)
      case maths.logarithm_2(int_to_float(n)) {
        Ok(max_h) ->
          case max_h == 0.0 {
            True -> 0.0
            False -> h /. max_h
          }
        Error(_) -> 0.0
      }
    }
  }
}

/// KL Divergence: D_KL(P || Q) = Σ p(x) log(p(x) / q(x))
///
/// Measures how much P diverges from Q (not symmetric!).
/// P is the "true" distribution, Q is the approximation.
///
/// Returns Error if distributions have different lengths or Q has zeros where P is non-zero.
pub fn kl_divergence(p: List(Float), q: List(Float)) -> Result(Float, Nil) {
  case list.length(p) == list.length(q) {
    False -> Error(Nil)
    True -> {
      let pairs = list.zip(p, q)
      let result =
        list.fold(pairs, Ok(0.0), fn(acc, pair) {
          case acc {
            Error(Nil) -> Error(Nil)
            Ok(sum) -> {
              let #(pi, qi) = pair
              case pi <=. 0.0 {
                True -> Ok(sum)
                // 0 * log(0/q) = 0
                False ->
                  case qi <=. 0.0 {
                    True -> Error(Nil)
                    // Can't have q=0 when p>0
                    False ->
                      case maths.natural_logarithm(pi /. qi) {
                        Ok(log_ratio) -> Ok(sum +. { pi *. log_ratio })
                        Error(_) -> Error(Nil)
                      }
                  }
              }
            }
          }
        })
      result
    }
  }
}

/// Symmetric KL Divergence (Jensen-Shannon divergence without the 1/2).
///
/// D_sym(P, Q) = D_KL(P || Q) + D_KL(Q || P)
pub fn symmetric_kl(p: List(Float), q: List(Float)) -> Result(Float, Nil) {
  case kl_divergence(p, q), kl_divergence(q, p) {
    Ok(d1), Ok(d2) -> Ok(d1 +. d2)
    _, _ -> Error(Nil)
  }
}

/// Jensen-Shannon Divergence: JS(P, Q) = (D_KL(P || M) + D_KL(Q || M)) / 2
/// where M = (P + Q) / 2
///
/// This is symmetric and bounded [0, 1] when using log₂.
pub fn jensen_shannon(p: List(Float), q: List(Float)) -> Result(Float, Nil) {
  case list.length(p) == list.length(q) {
    False -> Error(Nil)
    True -> {
      // Compute M = (P + Q) / 2
      let m =
        list.zip(p, q)
        |> list.map(fn(pair) {
          let #(pi, qi) = pair
          { pi +. qi } /. 2.0
        })

      case kl_divergence(p, m), kl_divergence(q, m) {
        Ok(d_pm), Ok(d_qm) -> Ok({ d_pm +. d_qm } /. 2.0)
        _, _ -> Error(Nil)
      }
    }
  }
}

/// Cross-entropy: H(P, Q) = -Σ p(x) log q(x)
///
/// Used in machine learning loss functions.
/// H(P, Q) = H(P) + D_KL(P || Q)
pub fn cross_entropy(p: List(Float), q: List(Float)) -> Result(Float, Nil) {
  case list.length(p) == list.length(q) {
    False -> Error(Nil)
    True -> {
      let pairs = list.zip(p, q)
      let result =
        list.fold(pairs, Ok(0.0), fn(acc, pair) {
          case acc {
            Error(Nil) -> Error(Nil)
            Ok(sum) -> {
              let #(pi, qi) = pair
              case pi <=. 0.0 {
                True -> Ok(sum)
                False ->
                  case qi <=. 0.0 {
                    True -> Error(Nil)
                    False ->
                      case maths.natural_logarithm(qi) {
                        Ok(log_q) -> Ok(sum -. { pi *. log_q })
                        Error(_) -> Error(Nil)
                      }
                  }
              }
            }
          }
        })
      result
    }
  }
}

/// Binary cross-entropy for single probability.
///
/// H(p, q) = -[p log(q) + (1-p) log(1-q)]
pub fn binary_cross_entropy(p: Float, q: Float) -> Result(Float, Nil) {
  // Clamp q to avoid log(0)
  let q_clamped = float.max(float.min(q, 0.999999), 0.000001)
  let q_inv = 1.0 -. q_clamped
  let p_inv = 1.0 -. p

  case maths.natural_logarithm(q_clamped), maths.natural_logarithm(q_inv) {
    Ok(log_q), Ok(log_q_inv) -> {
      let result = p *. log_q +. p_inv *. log_q_inv
      Ok(0.0 -. result)
    }
    _, _ -> Error(Nil)
  }
}

/// Mutual Information: I(X; Y) = H(X) + H(Y) - H(X, Y)
///
/// Measures shared information between two variables.
/// Takes marginal distributions and joint distribution as input.
pub fn mutual_information(
  px: List(Float),
  py: List(Float),
  pxy: List(List(Float)),
) -> Float {
  let hx = shannon(px)
  let hy = shannon(py)

  // Flatten joint distribution and compute joint entropy
  let pxy_flat = list.flatten(pxy)
  let hxy = shannon(pxy_flat)

  hx +. hy -. hxy
}

/// Conditional entropy: H(X|Y) = H(X, Y) - H(Y)
///
/// Uncertainty in X given knowledge of Y.
pub fn conditional_entropy(px: List(Float), pxy: List(List(Float))) -> Float {
  let hx = shannon(px)
  let pxy_flat = list.flatten(pxy)
  let hxy = shannon(pxy_flat)
  hxy -. hx
}

/// Relative entropy rate for sequences.
///
/// Used for measuring "surprise" in temporal data.
pub fn relative_entropy_rate(
  observed: List(Float),
  expected: List(Float),
) -> Result(Float, Nil) {
  case list.length(observed) == list.length(expected) {
    False -> Error(Nil)
    True -> {
      let n = list.length(observed)
      case n == 0 {
        True -> Ok(0.0)
        False ->
          case kl_divergence(observed, expected) {
            Ok(kl) -> Ok(kl /. int_to_float(n))
            Error(Nil) -> Error(Nil)
          }
      }
    }
  }
}

// Helper: convert int to float
fn int_to_float(n: Int) -> Float {
  case n {
    0 -> 0.0
    1 -> 1.0
    2 -> 2.0
    _ -> {
      let half = n / 2
      let remainder = n - half * 2
      int_to_float(half) *. 2.0 +. int_to_float(remainder)
    }
  }
}
