//! Deterministic std-only F64 one-sided Jacobi SVD and exact modular bounds.
use crate::error::ToolError;
use crate::ops::Cx;

pub(super) fn invalid(detail: impl Into<String>) -> ToolError {
    ToolError::InvalidArgument {
        detail: detail.into(),
    }
}

pub(super) fn bf16(bits: u16) -> f64 {
    f32::from_bits(u32::from(bits) << 16) as f64
}

/// Error-free TwoSum residual detects a dyadic difference rounded by F64.
pub(super) fn exact_difference(new: u16, old: u16) -> Result<f64, ToolError> {
    let x = bf16(new);
    let y = -bf16(old);
    if !x.is_finite() || !y.is_finite() {
        return Err(invalid(
            "derive-lora requires finite BF16 base and patched values",
        ));
    }
    let sum = x + y;
    let virtual_y = sum - x;
    let residual = (x - (sum - virtual_y)) + (y - virtual_y);
    if residual != 0.0 {
        return Err(invalid("exact BF16 difference is not representable in F64"));
    }
    Ok(sum)
}

pub(super) fn bake(value: f64) -> Result<u16, ToolError> {
    let value = value as f32;
    if !value.is_finite() {
        return Err(invalid("frozen bake overflowed F32"));
    }
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    let out = (rounded >> 16) as u16;
    if out & 0x7f80 == 0x7f80 {
        return Err(invalid("frozen bake overflowed BF16"));
    }
    Ok(out)
}

pub(super) fn ordered(bits: u16) -> u16 {
    if bits & 0x8000 != 0 {
        !bits
    } else {
        bits | 0x8000
    }
}

pub(super) struct Svd {
    pub singular: Vec<f64>,
    // Row major: U [rows, thin], V [cols, thin].
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub sweeps: usize,
}

/// Thin SVD, transposing wide input so the Jacobi columns are on the tall side.
pub(super) fn svd(input: &[f64], rows: usize, cols: usize, cx: &Cx) -> Result<Svd, ToolError> {
    if rows < cols {
        let transposed: Vec<f64> = (0..cols)
            .flat_map(|j| (0..rows).map(move |i| input[i * cols + j]))
            .collect();
        let result = tall_svd(&transposed, cols, rows, cx)?;
        let mut out = Svd {
            singular: result.singular,
            u: result.v,
            v: result.u,
            sweeps: result.sweeps,
        };
        canonical_signs(&mut out, rows, cols);
        Ok(out)
    } else {
        tall_svd(input, rows, cols, cx)
    }
}

fn canonical_signs(out: &mut Svd, rows: usize, cols: usize) {
    let n = out.singular.len();
    for j in 0..n {
        let mut pivot = 0;
        for i in 1..rows {
            if out.u[i * n + j].abs() > out.u[pivot * n + j].abs() {
                pivot = i;
            }
        }
        if out.u[pivot * n + j].is_sign_negative() {
            for i in 0..rows {
                out.u[i * n + j] = -out.u[i * n + j];
            }
            for i in 0..cols {
                out.v[i * n + j] = -out.v[i * n + j];
            }
        }
    }
}

fn tall_svd(input: &[f64], rows: usize, cols: usize, cx: &Cx) -> Result<Svd, ToolError> {
    let mut work = input.to_vec();
    let mut v = vec![0.0; cols * cols];
    for j in 0..cols {
        v[j * cols + j] = 1.0;
    }
    // Cyclic fixed pair ordering; refuse rather than export an unconverged SVD.
    const TOL: f64 = 8.0 * f64::EPSILON;
    const MAX_SWEEPS: usize = 256;
    for sweep in 1..=MAX_SWEEPS {
        cx.check()?;
        let mut rotations = 0;
        for p in 0..cols {
            for q in p + 1..cols {
                cx.check()?;
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for i in 0..rows {
                    let a = work[i * cols + p];
                    let b = work[i * cols + q];
                    alpha += a * a;
                    beta += b * b;
                    gamma += a * b;
                }
                if gamma.abs() <= TOL * alpha.sqrt() * beta.sqrt() {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + zeta.hypot(1.0));
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for data in [&mut work, &mut v] {
                    for row in data.chunks_exact_mut(cols) {
                        let a = row[p];
                        let b = row[q];
                        row[p] = c * a - s * b;
                        row[q] = s * a + c * b;
                    }
                }
                rotations += 1;
            }
        }
        if rotations == 0 {
            let norms: Vec<f64> = (0..cols)
                .map(|j| {
                    (0..rows)
                        .map(|i| work[i * cols + j].powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            let mut order: Vec<usize> = (0..cols).collect();
            order.sort_by(|&a, &b| norms[b].total_cmp(&norms[a]).then(a.cmp(&b)));
            let mut out = Svd {
                singular: order.iter().map(|&j| norms[j]).collect(),
                u: vec![0.0; rows * cols],
                v: vec![0.0; cols * cols],
                sweeps: sweep,
            };
            for (j, &old) in order.iter().enumerate() {
                for i in 0..rows {
                    out.u[i * cols + j] = if norms[old] == 0.0 {
                        0.0
                    } else {
                        work[i * cols + old] / norms[old]
                    };
                }
                for i in 0..cols {
                    out.v[i * cols + j] = v[i * cols + old];
                }
            }
            // Complete exact-zero U columns with a deterministic orthonormal
            // complement. Their singular values stay zero; no tail is dropped.
            for j in 0..cols {
                if out.singular[j] != 0.0 {
                    continue;
                }
                let mut found = false;
                for basis in 0..rows {
                    let mut candidate = vec![0.0; rows];
                    candidate[basis] = 1.0;
                    for _ in 0..2 {
                        for previous in 0..j {
                            let dot: f64 = (0..rows)
                                .map(|i| candidate[i] * out.u[i * cols + previous])
                                .sum();
                            for (i, value) in candidate.iter_mut().enumerate() {
                                *value -= dot * out.u[i * cols + previous];
                            }
                        }
                    }
                    let norm = candidate.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if norm > f64::EPSILON.sqrt() {
                        for (i, value) in candidate.into_iter().enumerate() {
                            out.u[i * cols + j] = value / norm;
                        }
                        found = true;
                        break;
                    }
                }
                if !found {
                    return Err(invalid("cannot complete zero-singular-value U basis"));
                }
            }
            canonical_signs(&mut out, rows, cols);
            return Ok(out);
        }
    }
    Err(invalid(
        "Jacobi SVD did not converge in 256 sweeps; nothing exported",
    ))
}

pub(super) const PRIMES: [u64; 2] = [1_000_000_007, 1_000_000_009];
fn pow_mod(mut a: u64, mut n: u64, prime: u64) -> u64 {
    let mut result = 1;
    while n > 0 {
        if n & 1 != 0 {
            result = result * a % prime;
        }
        a = a * a % prime;
        n >>= 1;
    }
    result
}

// Exact BF16 * 2^133 mapped directly into the field, no F64 conversion.
fn integer_mod(bits: u16, prime: u64) -> u64 {
    let exp = (bits >> 7) & 0xff;
    let frac = u64::from(bits & 0x7f);
    let value = if exp == 0 {
        frac
    } else {
        (128 + frac) * pow_mod(2, u64::from(exp - 1), prime) % prime
    };
    if bits & 0x8000 != 0 && value != 0 {
        prime - value
    } else {
        value
    }
}

/// Modular ranks are lower bounds; equality to min(rows,k) proves exact rank.
pub(super) fn modular_ranks(
    base: &[u16],
    patched: &[u16],
    width: usize,
    active: &[usize],
    cx: &Cx,
) -> Result<[usize; 2], ToolError> {
    let rows = base.len() / width;
    let cols = active.len();
    let mut result = [0; 2];
    for (slot, prime) in result.iter_mut().zip(PRIMES) {
        let mut a: Vec<u64> = (0..rows)
            .flat_map(|i| {
                active.iter().map(move |&j| {
                    (integer_mod(patched[i * width + j], prime) + prime
                        - integer_mod(base[i * width + j], prime))
                        % prime
                })
            })
            .collect();
        let mut rank = 0;
        for col in 0..cols {
            cx.check()?;
            let Some(pivot) = (rank..rows).find(|&row| a[row * cols + col] != 0) else {
                continue;
            };
            for j in col..cols {
                a.swap(rank * cols + j, pivot * cols + j);
            }
            let inverse = pow_mod(a[rank * cols + col], prime - 2, prime);
            for j in col..cols {
                a[rank * cols + j] = a[rank * cols + j] * inverse % prime;
            }
            for i in rank + 1..rows {
                let factor = a[i * cols + col];
                for j in col..cols {
                    a[i * cols + j] =
                        (a[i * cols + j] + prime - factor * a[rank * cols + j] % prime) % prime;
                }
            }
            rank += 1;
            if rank == rows {
                break;
            }
        }
        *slot = rank;
    }
    Ok(result)
}
