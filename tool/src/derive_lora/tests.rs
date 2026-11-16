use super::*;
use std::sync::atomic::AtomicBool;

fn with_context(f: impl FnOnce(&mut Cx)) {
    let cancelled = AtomicBool::new(false);
    let mut progress = |_| {};
    f(&mut Cx::new(&cancelled, &mut progress));
}

// Proves analytic non-diagonal spectra and reconstruction for these tall rank-2/3
// fixtures (absolute singular-value tolerance 1e-12); not a general SVD proof.
#[test]
fn known_tall_spectra_and_signs() {
    with_context(|cx| {
        let u = [
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
        ];
        let v = [[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]];
        for spectrum in [[8.0, 3.0, 1.0], [8.0, 3.0, 0.0]] {
            let mut input = vec![0.0; 12];
            for i in 0..4 {
                for j in 0..3 {
                    input[i * 3 + j] = (0..3).map(|k| u[i][k] * spectrum[k] * v[j][k]).sum();
                }
            }
            let got = math::svd(&input, 4, 3, cx).unwrap();
            for (&a, &b) in got.singular.iter().zip(&spectrum) {
                assert!((a - b).abs() < 1e-12, "{a} vs {b}");
            }
            for i in 0..4 {
                for j in 0..3 {
                    let value: f64 = (0..3)
                        .map(|k| got.u[i * 3 + k] * got.singular[k] * got.v[j * 3 + k])
                        .sum();
                    assert!((value - input[i * 3 + j]).abs() < 1e-12);
                }
            }
            for j in 0..3 {
                let mut pivot = 0;
                for i in 1..4 {
                    if got.u[i * 3 + j].abs() > got.u[pivot * 3 + j].abs() {
                        pivot = i;
                    }
                }
                assert!(!got.u[pivot * 3 + j].is_sign_negative());
            }
        }
    });
}

// Proves the wide-matrix transpose path recovers a known spectrum and product.
#[test]
fn wide_spectrum() {
    with_context(|cx| {
        let matrix = [3.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let got = math::svd(&matrix, 2, 4, cx).unwrap();
        assert!((got.singular[0] - 5.0).abs() < 1e-12);
        assert!((got.singular[1] - 2.0).abs() < 1e-12);
        for i in 0..2 {
            for j in 0..4 {
                let x: f64 = (0..2)
                    .map(|k| got.u[i * 2 + k] * got.singular[k] * got.v[j * 2 + k])
                    .sum();
                assert!((x - matrix[i * 4 + j]).abs() < 1e-12);
            }
        }
    });
}

// Proves this selector's uncast F64 product is exact and its stored F32 factors
// bake with zero mismatches; unlike SVD, no numerical factorization is involved.
#[test]
fn selector_exactness() {
    with_context(|cx| {
        let old = vec![0x3f80; 12];
        let new = vec![
            0x4000, 0x3f80, 0x3fc0, 0x3f80, 0x4040, 0x3f80, 0x3f00, 0x3f80, 0x4080, 0x3f80, 0x4000,
            0x3f80,
        ];
        let delta: Vec<f64> = new
            .iter()
            .zip(&old)
            .map(|(&n, &o)| exact_difference(n, o).unwrap())
            .collect();
        let compact: Vec<f64> = (0..3)
            .flat_map(|i| [delta[i * 4], delta[i * 4 + 2]])
            .collect();
        let svd = math::svd(&compact, 3, 2, cx).unwrap();
        let request = DeriveLora {
            model: PathBuf::new(),
            patch: PathBuf::new(),
            rank: 2,
            method: Method::Selector,
            out: PathBuf::new(),
            expect_tensor_sha256: BTreeMap::new(),
        };
        let (a, b) = factors_for(&request, 3, 4, &[0, 2], &compact, &svd);
        for i in 0..3 {
            for j in 0..4 {
                let value: f64 = (0..2).map(|k| b[i * 2 + k] * a[k * 4 + j]).sum();
                assert_eq!(value, delta[i * 4 + j]);
            }
        }
        let report = measure(
            &old,
            &new,
            &delta,
            &[true, false, true, false],
            &stored_f32(a).unwrap(),
            &stored_f32(b).unwrap(),
            2,
            cx,
        )
        .unwrap();
        assert_eq!(report["delta_frobenius_error"], 0.0);
        assert_eq!(report["bf16_mismatch_count"], 0);
        assert_eq!(report["baked_frobenius_error"], 0.0);
    });
}

// Proves paired valid/refused exact-dyadic differences, including an exponent
// gap that F64 would silently round; NaN and both infinity signs are refused.
#[test]
fn exact_difference_controls() {
    assert_eq!(exact_difference(0x4000, 0x3f80).unwrap(), 1.0);
    assert_eq!(exact_difference(1, 0).unwrap(), 2.0f64.powi(-133));
    assert!(
        exact_difference(0x3f80, 1)
            .unwrap_err()
            .to_string()
            .contains("not representable in F64")
    );
    for bits in [0x7f80, 0xff80, 0x7fc1] {
        assert!(
            exact_difference(bits, 0)
                .unwrap_err()
                .to_string()
                .contains("finite BF16")
        );
        assert!(
            exact_difference(0, bits)
                .unwrap_err()
                .to_string()
                .contains("finite BF16")
        );
    }
}

// Proves ties-to-even at two neighboring BF16 midpoints and signed-zero ULP
// distinction, and refuses a nonfinite/overflowed bake and stored factor.
#[test]
fn bake_and_factor_controls() {
    assert_eq!(math::bake(1.00390625).unwrap(), 0x3f80);
    assert_eq!(math::bake(1.01171875).unwrap(), 0x3f82);
    assert_eq!(math::bake(-0.0).unwrap(), 0x8000);
    assert_eq!(math::ordered(0).abs_diff(math::ordered(0x8000)), 1);
    assert!(math::bake(f64::INFINITY).is_err());
    assert!(math::bake(1e100).is_err());
    assert_eq!(stored_f32(vec![1.0]).unwrap(), vec![1.0]);
    assert!(stored_f32(vec![1e100]).is_err());
}

// Proves full/deficient modular lower bounds on small integer fixtures and
// exact handling of signed subnormal BF16 integers (scale 2^133).
#[test]
fn modular_rank_controls() {
    with_context(|cx| {
        let old = [0, 0, 0, 0];
        assert_eq!(
            math::modular_ranks(&old, &[1, 0, 0, 0x8001], 2, &[0, 1], cx).unwrap(),
            [2, 2]
        );
        assert_eq!(
            math::modular_ranks(&old, &[0x3f80, 0x4000, 0x4000, 0x4080], 2, &[0, 1], cx).unwrap(),
            [1, 1]
        );
    });
}

// Proves fixed-order Jacobi handles a repeated nonzero singular subspace and
// zero matrix without a convergence failure; portable factor identity unclaimed.
#[test]
fn repeated_and_zero_spectra() {
    with_context(|cx| {
        for matrix in [[2.0, 2.0, 2.0, -2.0], [0.0; 4]] {
            let got = math::svd(&matrix, 2, 2, cx).unwrap();
            let expected = matrix[0].abs() * 2.0f64.sqrt();
            for j in 0..2 {
                for k in 0..2 {
                    let dot: f64 = (0..2).map(|i| got.u[i * 2 + j] * got.u[i * 2 + k]).sum();
                    assert!((dot - if j == k { 1.0 } else { 0.0 }).abs() < 1e-12);
                }
            }
            for value in got.singular {
                assert!((value - expected).abs() < 1e-12);
            }
        }
    });
}

// Proves the metric checker detects a deliberately lossy stored F32 selector
// difference while its frozen BF16 bake still reproduces this output.
#[test]
fn lossy_storage_metric_control() {
    with_context(|cx| {
        let old = [0x3080]; // 2^-30
        let new = [0x3f80]; // 1
        let delta = [exact_difference(new[0], old[0]).unwrap()];
        let report = measure(&old, &new, &delta, &[true], &[1.0], &[1.0], 1, cx).unwrap();
        assert_eq!(report["delta_frobenius_error"], 2.0f64.powi(-30));
        assert_eq!(report["bf16_mismatch_count"], 0);
    });
}
