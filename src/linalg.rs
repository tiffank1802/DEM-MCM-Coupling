//! Dense linear algebra, matrix operations, and numerical decompositions.
//!
//! All matrices are stored in row-major order. Includes:
//! - Matrix multiplication, vector-matrix multiplication (`phi @ P`), transpose
//! - Sample statistics (`mean`, `std`, `var`, `median`, `quantile`, `cov`)
//! - Cholesky factorization + triangular solve (for GMM log-likelihoods)
//! - Symmetric eigendecomposition `jacobi_eigh` (ascending eigenvalues)
//! - Thin SVD (`one_sided_jacobi_svd`, descending singular values)
//! - Hessenberg + Francis Double-Shift QR for general real matrix eigenvalues
//! - Stationary distribution of a row-stochastic transition matrix
//! - Vandermonde QR polynomial least-squares regression (`polyfit`, `polyval`)

use std::cmp::Ordering;

use crate::error::{DemMcmError, Result};

/// Dense row-major `f64` matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Allocate a zero-initialised `(rows, cols)` matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Allocate a `(rows, cols)` matrix filled with `val`.
    pub fn full(rows: usize, cols: usize, val: f64) -> Self {
        Self {
            rows,
            cols,
            data: vec![val; rows * cols],
        }
    }

    /// Allocate an `n x n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Construct from flat row-major data.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if rows * cols != data.len() {
            return Err(DemMcmError::ShapeMismatch(format!(
                "Matrix::from_vec expected {} elements for ({rows}, {cols}), got {}",
                rows * cols,
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// Construct from row slices.
    pub fn from_rows(rows_data: &[Vec<f64>]) -> Result<Self> {
        if rows_data.is_empty() {
            return Ok(Self::zeros(0, 0));
        }
        let cols = rows_data[0].len();
        let mut data = Vec::with_capacity(rows_data.len() * cols);
        for (i, r) in rows_data.iter().enumerate() {
            if r.len() != cols {
                return Err(DemMcmError::ShapeMismatch(format!(
                    "Row {i} has length {}, expected {cols}",
                    r.len()
                )));
            }
            data.extend_from_slice(r);
        }
        Ok(Self {
            rows: rows_data.len(),
            cols,
            data,
        })
    }

    /// Construct from `[f64; 3]` coordinate rows.
    pub fn from_vec3(points: &[[f64; 3]]) -> Self {
        let mut data = Vec::with_capacity(points.len() * 3);
        for p in points {
            data.extend_from_slice(p);
        }
        Self {
            rows: points.len(),
            cols: 3,
            data,
        }
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f64] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn set_row(&mut self, r: usize, values: &[f64]) {
        self.row_mut(r).copy_from_slice(values);
    }

    pub fn select_rows(&self, indices: &[usize]) -> Self {
        let mut out = Self::zeros(indices.len(), self.cols);
        for (dst_r, &src_r) in indices.iter().enumerate() {
            out.set_row(dst_r, self.row(src_r));
        }
        out
    }

    pub fn col(&self, c: usize) -> Vec<f64> {
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }

    /// Return the transpose `self^T`.
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// Sum over each row (`axis = 1`).
    pub fn row_sums(&self) -> Vec<f64> {
        (0..self.rows).map(|r| self.row(r).iter().sum()).collect()
    }

    /// Sum over each column (`axis = 0`).
    pub fn col_sums(&self) -> Vec<f64> {
        let mut sums = vec![0.0; self.cols];
        for r in 0..self.rows {
            let row = self.row(r);
            for c in 0..self.cols {
                sums[c] += row[c];
            }
        }
        sums
    }

    /// Diagonal elements (length `min(rows, cols)`).
    pub fn diag(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }

    /// Mean of the diagonal elements.
    pub fn diag_mean(&self) -> f64 {
        let d = self.diag();
        if d.is_empty() {
            0.0
        } else {
            d.iter().sum::<f64>() / d.len() as f64
        }
    }

    /// Standard deviation (ddof=0) of the diagonal elements.
    pub fn diag_std(&self) -> f64 {
        std_dev(&self.diag())
    }

    /// Total sum of all entries.
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Frobenius norm `sqrt(sum(a_ij^2))` (treating NaN as 0 if `ignore_nan`).
    pub fn frobenius_norm(&self, ignore_nan: bool) -> f64 {
        let s: f64 = self
            .data
            .iter()
            .map(|&v| if ignore_nan && v.is_nan() { 0.0 } else { v * v })
            .sum();
        s.sqrt()
    }

    /// Right vector-matrix multiplication `phi @ self` (`len(phi) == self.rows`).
    pub fn vec_mul(&self, phi: &[f64]) -> Result<Vec<f64>> {
        if phi.len() != self.rows {
            return Err(DemMcmError::ShapeMismatch(format!(
                "vec_mul: phi length {} != matrix rows {}",
                phi.len(),
                self.rows
            )));
        }
        let mut out = vec![0.0; self.cols];
        for (i, &p) in phi.iter().enumerate() {
            if p == 0.0 {
                continue;
            }
            let row = self.row(i);
            for (j, &m_ij) in row.iter().enumerate() {
                out[j] += p * m_ij;
            }
        }
        Ok(out)
    }

    /// Matrix-matrix multiplication `self @ other`.
    pub fn mat_mul(&self, other: &Self) -> Result<Self> {
        if self.cols != other.rows {
            return Err(DemMcmError::ShapeMismatch(format!(
                "mat_mul: ({}, {}) x ({}, {})",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut out = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            let a_row = self.row(i);
            let out_row = out.row_mut(i);
            for (k, &a_ik) in a_row.iter().enumerate() {
                if a_ik == 0.0 {
                    continue;
                }
                let b_row = other.row(k);
                for j in 0..other.cols {
                    out_row[j] += a_ik * b_row[j];
                }
            }
        }
        Ok(out)
    }

    /// Replace every NaN entry with `replacement` and return a new matrix.
    pub fn nan_to_num(&self, replacement: f64) -> Self {
        let data = self
            .data
            .iter()
            .map(|&v| if v.is_nan() { replacement } else { v })
            .collect();
        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Count NaN entries.
    pub fn nan_count(&self) -> usize {
        self.data.iter().filter(|v| v.is_nan()).count()
    }

    /// Column-wise means (`axis = 0`).
    pub fn col_means(&self) -> Vec<f64> {
        if self.rows == 0 {
            return vec![0.0; self.cols];
        }
        let sums = self.col_sums();
        sums.into_iter().map(|s| s / self.rows as f64).collect()
    }

    /// Column-wise population standard deviations (`axis = 0`, `ddof = 0`).
    pub fn col_stds(&self) -> Vec<f64> {
        if self.rows == 0 {
            return vec![0.0; self.cols];
        }
        let means = self.col_means();
        let mut vars = vec![0.0; self.cols];
        for r in 0..self.rows {
            let row = self.row(r);
            for c in 0..self.cols {
                let d = row[c] - means[c];
                vars[c] += d * d;
            }
        }
        vars.into_iter()
            .map(|v| (v / self.rows as f64).sqrt())
            .collect()
    }
}

/// Arithmetic mean of a slice (0.0 when empty).
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Population variance (`ddof = 0`).
pub fn variance(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64
}

/// Population standard deviation (`ddof = 0`).
pub fn std_dev(xs: &[f64]) -> f64 {
    variance(xs).sqrt()
}

/// Linear-interpolated quantile matching `numpy.quantile(xs, q, method="linear")`.
pub fn quantile(xs: &[f64], q: f64) -> f64 {
    assert!(!xs.is_empty(), "quantile called on empty slice");
    let q = q.clamp(0.0, 1.0);
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Sample median matching `numpy.median`.
pub fn median(xs: &[f64]) -> f64 {
    quantile(xs, 0.5)
}

/// Sample covariance matrix `(cols, cols)` of `data` (`rows` observations, `ddof = 1`).
pub fn covariance_matrix(data: &Matrix) -> Result<Matrix> {
    if data.rows < 2 {
        return Ok(Matrix::zeros(data.cols, data.cols));
    }
    let means = data.col_means();
    let d = data.cols;
    let mut cov = Matrix::zeros(d, d);
    for r in 0..data.rows {
        let row = data.row(r);
        for i in 0..d {
            let di = row[i] - means[i];
            for j in i..d {
                let dj = row[j] - means[j];
                let idx = i * d + j;
                cov.data[idx] += di * dj;
            }
        }
    }
    let denom = (data.rows - 1) as f64;
    for i in 0..d {
        for j in i..d {
            let v = cov.get(i, j) / denom;
            cov.set(i, j, v);
            cov.set(j, i, v);
        }
    }
    Ok(cov)
}

/// Lower-triangular Cholesky decomposition `A = L L^T` for symmetric positive-definite `A`.
pub fn cholesky(a: &Matrix) -> Result<Matrix> {
    if a.rows != a.cols {
        return Err(DemMcmError::ShapeMismatch(
            "cholesky requires a square matrix".to_string(),
        ));
    }
    let n = a.rows;
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a.get(i, j);
            for k in 0..j {
                sum -= l.get(i, k) * l.get(j, k);
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return Err(DemMcmError::InvalidArgument(
                        "Matrix is not positive definite in cholesky".to_string(),
                    ));
                }
                l.set(i, j, sum.sqrt());
            } else {
                l.set(i, j, sum / l.get(j, j));
            }
        }
    }
    Ok(l)
}

/// Solve `L y = b` where `L` is lower triangular.
pub fn solve_lower_triangular(l: &Matrix, b: &[f64]) -> Vec<f64> {
    let n = l.rows;
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        let row = l.row(i);
        for j in 0..i {
            s -= row[j] * y[j];
        }
        y[i] = s / row[i];
    }
    y
}

/// Symmetric eigendecomposition via Cyclic Jacobi rotations.
///
/// Returns `(eigenvalues, eigenvectors)` sorted in **ascending** eigenvalue order,
/// where column `k` of `eigenvectors` is the unit eigenvector for `eigenvalues[k]`.
pub fn jacobi_eigh(a: &Matrix) -> Result<(Vec<f64>, Matrix)> {
    if a.rows != a.cols {
        return Err(DemMcmError::ShapeMismatch(
            "jacobi_eigh requires a square matrix".to_string(),
        ));
    }
    let n = a.rows;
    if n == 0 {
        return Ok((Vec::new(), Matrix::zeros(0, 0)));
    }
    if n == 1 {
        return Ok((vec![a.get(0, 0)], Matrix::identity(1)));
    }

    // Symmetrise input slightly against floating-point asymmetry.
    let mut m = Matrix::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let v = 0.5 * (a.get(i, j) + a.get(j, i));
            m.set(i, j, v);
            m.set(j, i, v);
        }
    }
    let mut v = Matrix::identity(n);
    let max_sweeps = 100;

    for _sweep in 0..max_sweeps {
        let mut off_norm = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                let val = m.get(p, q);
                off_norm += val * val;
            }
        }
        if off_norm <= 1e-28 * (n as f64) {
            break;
        }

        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                let g = 100.0 * m.get(p, q).abs();
                let app = m.get(p, p);
                let aqq = m.get(q, q);
                if g == 0.0 {
                    continue;
                }
                let apq = m.get(p, q);
                if apq.abs() <= 1e-16 * (app.abs() + aqq.abs()) {
                    m.set(p, q, 0.0);
                    m.set(q, p, 0.0);
                    continue;
                }

                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let rot_tau = s / (1.0 + c);

                m.set(p, p, app - t * apq);
                m.set(q, q, aqq + t * apq);
                m.set(p, q, 0.0);
                m.set(q, p, 0.0);

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = m.get(r, p);
                    let arq = m.get(r, q);
                    let nrp = arp - s * (arq + rot_tau * arp);
                    let nrq = arq + s * (arp - rot_tau * arq);
                    m.set(r, p, nrp);
                    m.set(p, r, nrp);
                    m.set(r, q, nrq);
                    m.set(q, r, nrq);
                }

                for r in 0..n {
                    let vrp = v.get(r, p);
                    let vrq = v.get(r, q);
                    v.set(r, p, vrp - s * (vrq + rot_tau * vrp));
                    v.set(r, q, vrq + s * (vrp - rot_tau * vrq));
                }
            }
        }
    }

    let mut pairs: Vec<(f64, usize)> = (0..n).map(|i| (m.get(i, i), i)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let evals: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let mut evecs = Matrix::zeros(n, n);
    for (new_c, &(_, old_c)) in pairs.iter().enumerate() {
        for r in 0..n {
            evecs.set(r, new_c, v.get(r, old_c));
        }
    }
    Ok((evals, evecs))
}

/// Thin SVD `A ≈ U * diag(sigma) * Vt` where singular values are descending.
///
/// For `rows >= cols` computes `U: (rows, r)`, `sigma: (r,)`, `Vt: (r, cols)`
/// where `r = min(rows, cols)`.
/// Uses one-sided Jacobi rotations on the smaller dimension (`A` or `A^T`).
pub fn thin_svd(a: &Matrix) -> Result<(Matrix, Vec<f64>, Matrix)> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Ok((Matrix::zeros(m, 0), Vec::new(), Matrix::zeros(0, n)));
    }
    if m < n {
        // Compute SVD of A^T = V * Sigma * U^T, then transpose back.
        let (v, s, ut) = thin_svd(&a.transpose())?;
        return Ok((ut.transpose(), s, v.transpose()));
    }

    // Now m >= n. Apply one-sided Jacobi rotations to columns of W (m x n)
    // accumulating right orthogonal matrix V (n x n).
    // For tall skinny matrices (e.g. 4000 x 7), computing Gram matrix first is
    // equivalent and much faster if we still orthogonalise columns cleanly;
    // one-sided Jacobi directly on W is already O(sweeps * m * n^2) and handles
    // small singular values without squaring condition number.
    let mut w = a.clone();
    let mut v = Matrix::identity(n);

    for _sweep in 0..60 {
        let mut converged = true;
        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for r in 0..m {
                    let wp = w.get(r, p);
                    let wq = w.get(r, q);
                    alpha += wp * wp;
                    beta += wq * wq;
                    gamma += wp * wq;
                }
                let denom = (alpha * beta).sqrt();
                if denom > 0.0 && gamma.abs() > 1e-14 * denom {
                    converged = false;
                    let zeta = (beta - alpha) / (2.0 * gamma);
                    let t = if zeta >= 0.0 {
                        1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                    } else {
                        -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = c * t;

                    for r in 0..m {
                        let wp = w.get(r, p);
                        let wq = w.get(r, q);
                        w.set(r, p, c * wp - s * wq);
                        w.set(r, q, s * wp + c * wq);
                    }
                    for r in 0..n {
                        let vp = v.get(r, p);
                        let vq = v.get(r, q);
                        v.set(r, p, c * vp - s * vq);
                        v.set(r, q, s * vp + c * vq);
                    }
                }
            }
        }
        if converged {
            break;
        }
    }

    let mut sigmas = vec![0.0; n];
    for (c, sigma) in sigmas.iter_mut().enumerate() {
        let norm: f64 = (0..m).map(|r| w.get(r, c).powi(2)).sum::<f64>().sqrt();
        *sigma = norm;
        if norm > 1e-15 {
            for r in 0..m {
                w.set(r, c, w.get(r, c) / norm);
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| sigmas[j].partial_cmp(&sigmas[i]).unwrap_or(Ordering::Equal));

    let mut u_out = Matrix::zeros(m, n);
    let mut s_out = vec![0.0; n];
    let mut vt_out = Matrix::zeros(n, n);
    for (new_c, &old_c) in order.iter().enumerate() {
        s_out[new_c] = sigmas[old_c];
        for r in 0..m {
            u_out.set(r, new_c, w.get(r, old_c));
        }
        for r in 0..n {
            vt_out.set(new_c, r, v.get(r, old_c));
        }
    }

    Ok((u_out, s_out, vt_out))
}

/// Reduce real square matrix `a` to upper Hessenberg form in-place (Householder).
fn to_hessenberg(h: &mut Matrix) {
    let n = h.rows;
    if n <= 2 {
        return;
    }
    for k in 0..(n - 2) {
        let len = n - 1 - k;
        let mut norm2 = 0.0;
        for i in (k + 1)..n {
            let v = h.get(i, k);
            norm2 += v * v;
        }
        let norm = norm2.sqrt();
        if norm < 1e-30 {
            continue;
        }
        let x0 = h.get(k + 1, k);
        let alpha = if x0 >= 0.0 { -norm } else { norm };
        let mut u = vec![0.0; len];
        for (i, u_i) in u.iter_mut().enumerate() {
            *u_i = h.get(k + 1 + i, k);
        }
        u[0] -= alpha;
        let u_norm2: f64 = u.iter().map(|v| v * v).sum();
        if u_norm2 < 1e-30 {
            continue;
        }
        let beta = 2.0 / u_norm2;

        // Left multiply: H[k+1:, k:] -= beta * u * (u^T H[k+1:, k:])
        for j in k..n {
            let mut dot = 0.0;
            for (i, &u_i) in u.iter().enumerate() {
                dot += u_i * h.get(k + 1 + i, j);
            }
            let scale = beta * dot;
            for (i, &u_i) in u.iter().enumerate() {
                let cur = h.get(k + 1 + i, j);
                h.set(k + 1 + i, j, cur - scale * u_i);
            }
        }
        // Right multiply: H[:, k+1:] -= beta * (H[:, k+1:] u) * u^T
        for i in 0..n {
            let mut dot = 0.0;
            for (j, &u_j) in u.iter().enumerate() {
                dot += h.get(i, k + 1 + j) * u_j;
            }
            let scale = beta * dot;
            for (j, &u_j) in u.iter().enumerate() {
                let cur = h.get(i, k + 1 + j);
                h.set(i, k + 1 + j, cur - scale * u_j);
            }
        }
        // Clean subdiagonal below k+1
        for i in (k + 2)..n {
            h.set(i, k, 0.0);
        }
    }
}

/// Compute all eigenvalues `(re, im)` of a general real square matrix via
/// Hessenberg reduction + Francis double-shift QR algorithm.
pub fn real_eigvals(a: &Matrix) -> Result<Vec<(f64, f64)>> {
    if a.rows != a.cols {
        return Err(DemMcmError::ShapeMismatch(
            "real_eigvals requires a square matrix".to_string(),
        ));
    }
    let n = a.rows;
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![(a.get(0, 0), 0.0)]);
    }

    let mut h = a.clone();
    to_hessenberg(&mut h);

    // Overall scale of A so zero eigenvalues still deflate cleanly despite 1e-16 roundoff.
    let mut h_norm = 0.0f64;
    for r in 0..n {
        let row_sum: f64 = (0..n).map(|c| h.get(r, c).abs()).sum();
        if row_sum > h_norm {
            h_norm = row_sum;
        }
    }
    let eps = 2.220446049250313e-16_f64;
    let floor_tol = (eps * h_norm).max(1e-30);
    let is_negligible = |sub: f64, d1: f64, d2: f64| -> bool {
        let s = d1.abs() + d2.abs();
        let thr = if s > 0.0 {
            (eps * s).max(floor_tol)
        } else {
            floor_tol
        };
        sub.abs() <= thr
    };

    let mut eigs = Vec::with_capacity(n);
    let mut nn = n; // active submatrix [0..nn)
    let mut iter_count = 0usize;

    while nn > 0 {
        if nn == 1 {
            eigs.push((h.get(0, 0), 0.0));
            break;
        }
        if nn == 2 {
            let a11 = h.get(0, 0);
            let a12 = h.get(0, 1);
            let a21 = h.get(1, 0);
            let a22 = h.get(1, 1);
            let tr = 0.5 * (a11 + a22);
            let disc = (0.5 * (a11 - a22)).powi(2) + a12 * a21;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                eigs.push((tr + sq, 0.0));
                eigs.push((tr - sq, 0.0));
            } else {
                let sq = (-disc).sqrt();
                eigs.push((tr, sq));
                eigs.push((tr, -sq));
            }
            break;
        }

        // Check deflation along subdiagonal
        let m_idx = nn - 1;
        let l_idx = nn - 2;
        if is_negligible(
            h.get(m_idx, m_idx - 1),
            h.get(m_idx - 1, m_idx - 1),
            h.get(m_idx, m_idx),
        ) {
            eigs.push((h.get(m_idx, m_idx), 0.0));
            nn -= 1;
            iter_count = 0;
            continue;
        }
        if l_idx > 0
            && is_negligible(
                h.get(l_idx, l_idx - 1),
                h.get(l_idx - 1, l_idx - 1),
                h.get(l_idx, l_idx),
            )
        {
            let a11 = h.get(l_idx, l_idx);
            let a12 = h.get(l_idx, m_idx);
            let a21 = h.get(m_idx, l_idx);
            let a22 = h.get(m_idx, m_idx);
            let tr = 0.5 * (a11 + a22);
            let disc = (0.5 * (a11 - a22)).powi(2) + a12 * a21;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                eigs.push((tr + sq, 0.0));
                eigs.push((tr - sq, 0.0));
            } else {
                let sq = (-disc).sqrt();
                eigs.push((tr, sq));
                eigs.push((tr, -sq));
            }
            nn -= 2;
            iter_count = 0;
            continue;
        }

        // Find active subblock starting row `l`
        let mut l = m_idx - 1;
        while l > 0 {
            if is_negligible(h.get(l, l - 1), h.get(l - 1, l - 1), h.get(l, l)) {
                h.set(l, l - 1, 0.0);
                break;
            }
            l -= 1;
        }

        iter_count += 1;
        if iter_count > 300 * nn {
            // Exceptional recovery: extract remaining diagonals
            for i in 0..nn {
                eigs.push((h.get(i, i), 0.0));
            }
            break;
        }

        // Wilkinson / LAPACK DLAHQR exceptional shift
        let (s, t) = if iter_count % 10 == 0 {
            let w = h.get(m_idx, m_idx - 1).abs() + h.get(l_idx, l_idx - 1).abs();
            let h11 = 0.75 * w + h.get(m_idx, m_idx);
            let h12 = -0.4375 * w;
            let h21 = w;
            (h11 + h11, h11 * h11 - h12 * h21)
        } else {
            let p1 = h.get(l_idx, l_idx);
            let p2 = h.get(m_idx, m_idx);
            (p1 + p2, p1 * p2 - h.get(l_idx, m_idx) * h.get(m_idx, l_idx))
        };

        // First column of (H^2 - s H + t I) at row l
        let h00 = h.get(l, l);
        let h10 = h.get(l + 1, l);
        let h01 = h.get(l, l + 1);
        let h11 = h.get(l + 1, l + 1);
        let h21 = if l + 2 < nn { h.get(l + 2, l + 1) } else { 0.0 };

        let mut x = h00 * h00 + h01 * h10 - s * h00 + t;
        let mut y = h10 * (h00 + h11 - s);
        let mut z = h10 * h21;

        for k in l..(nn - 1) {
            let nr = if k + 2 < nn { 3 } else { 2 };
            let mut v = if nr == 3 { vec![x, y, z] } else { vec![x, y] };
            let v_norm: f64 = v.iter().map(|u| u * u).sum::<f64>().sqrt();
            if v_norm > 1e-30 {
                let alpha = if v[0] >= 0.0 { -v_norm } else { v_norm };
                v[0] -= alpha;
                let u_norm2: f64 = v.iter().map(|u| u * u).sum();
                if u_norm2 > 1e-30 {
                    let beta = 2.0 / u_norm2;
                    let col_start = k.saturating_sub(1).max(l);
                    for j in col_start..nn {
                        let mut dot = 0.0;
                        for (r, &v_r) in v.iter().enumerate() {
                            dot += v_r * h.get(k + r, j);
                        }
                        let sc = beta * dot;
                        for (r, &v_r) in v.iter().enumerate() {
                            let cur = h.get(k + r, j);
                            h.set(k + r, j, cur - sc * v_r);
                        }
                    }
                    let row_end = (k + nr + 1).min(nn);
                    for i in 0..row_end {
                        let mut dot = 0.0;
                        for (c, &v_c) in v.iter().enumerate() {
                            dot += h.get(i, k + c) * v_c;
                        }
                        let sc = beta * dot;
                        for (c, &v_c) in v.iter().enumerate() {
                            let cur = h.get(i, k + c);
                            h.set(i, k + c, cur - sc * v_c);
                        }
                    }
                    if k > l {
                        h.set(k + 1, k - 1, 0.0);
                        if nr == 3 {
                            h.set(k + 2, k - 1, 0.0);
                        }
                    }
                }
            }
            if k + 2 < nn {
                x = h.get(k + 1, k);
                y = h.get(k + 2, k);
                z = if k + 3 < nn { h.get(k + 3, k) } else { 0.0 };
            }
        }
    }

    Ok(eigs)
}

/// Eigenvalue magnitudes `|lambda|` of a real square matrix.
pub fn real_eigvals_abs(a: &Matrix) -> Result<Vec<f64>> {
    Ok(real_eigvals(a)?
        .into_iter()
        .map(|(re, im)| (re * re + im * im).sqrt())
        .collect())
}

/// Compute the stationary distribution `pi` satisfying `pi @ P = pi` and `sum(pi) = 1`.
///
/// Uses lazy random-walk power iteration `(I + P^T) / 2` (which removes periodic
/// cycle oscillations while preserving the exact `+1` stationary subspace)
/// followed by inverse/nullspace refinement via SVD if residual > 1e-9.
pub fn stationary_distribution_left(p: &Matrix) -> Result<Vec<f64>> {
    if p.rows != p.cols {
        return Err(DemMcmError::ShapeMismatch(
            "stationary_distribution requires a square matrix".to_string(),
        ));
    }
    let n = p.rows;
    if n == 0 {
        return Ok(Vec::new());
    }
    let pt = p.transpose();

    // Identify visited rows (active states).
    let row_sums = p.row_sums();
    let mut pi: Vec<f64> = row_sums
        .iter()
        .map(|&s| if s > 0.0 { 1.0 } else { 0.0 })
        .collect();
    let active_sum: f64 = pi.iter().sum();
    if active_sum == 0.0 {
        return Ok(vec![1.0 / n as f64; n]);
    }
    for v in &mut pi {
        *v /= active_sum;
    }

    // Lazy iteration: pi_next = 0.5 * (pi + P^T pi)
    let mut buf = vec![0.0; n];
    for _ in 0..4000 {
        for i in 0..n {
            let row = pt.row(i);
            let mut s = 0.0;
            for j in 0..n {
                s += row[j] * pi[j];
            }
            buf[i] = 0.5 * (pi[i] + s);
        }
        let sum: f64 = buf.iter().sum();
        if sum > 0.0 {
            for v in &mut buf {
                *v /= sum;
            }
        }
        let diff: f64 = buf.iter().zip(pi.iter()).map(|(a, b)| (a - b).abs()).sum();
        pi.copy_from_slice(&buf);
        if diff < 1e-13 {
            break;
        }
    }

    // Check residual ||pi @ P - pi||_1; if slow-mixing/reducible, refine via
    // nullspace of (P^T - I) when matrix is reasonably small.
    let next = p.vec_mul(&pi)?;
    let res: f64 = next.iter().zip(pi.iter()).map(|(a, b)| (a - b).abs()).sum();
    if res > 1e-8 && n <= 256 {
        let mut a_mat = pt.clone();
        for i in 0..n {
            let cur = a_mat.get(i, i);
            a_mat.set(i, i, cur - 1.0);
        }
        if let Ok((_, sigmas, vt)) = thin_svd(&a_mat) {
            // Find singular vector with smallest singular value that is non-negative (or best sign).
            if let Some((min_idx, &min_s)) = sigmas
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            {
                if min_s < 1e-6 {
                    let candidate = vt.row(min_idx);
                    // Only accept if predominantly single-sign (valid probability vector).
                    let pos_sum: f64 = candidate.iter().filter(|&&x| x > 0.0).sum();
                    let neg_sum: f64 = candidate.iter().filter(|&&x| x < 0.0).map(|x| -x).sum();
                    let total = pos_sum + neg_sum;
                    if total > 0.0 && (pos_sum > 0.999 * total || neg_sum > 0.999 * total) {
                        let sign = if pos_sum >= neg_sum { 1.0 } else { -1.0 };
                        let mut refined: Vec<f64> =
                            candidate.iter().map(|&x| (sign * x).max(0.0)).collect();
                        let s: f64 = refined.iter().sum();
                        if s > 0.0 {
                            for v in &mut refined {
                                *v /= s;
                            }
                            let next2 = p.vec_mul(&refined)?;
                            let res2: f64 = next2
                                .iter()
                                .zip(refined.iter())
                                .map(|(a, b)| (a - b).abs())
                                .sum();
                            if res2 < res {
                                pi = refined;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(pi)
}

/// Polynomial least-squares fit matching `numpy.polyfit(x, y, degree)`.
///
/// Returns coefficients `[c_deg, ..., c_1, c_0]` ordered from highest power to constant.
pub fn polyfit(x: &[f64], y: &[f64], degree: usize) -> Result<Vec<f64>> {
    if x.len() != y.len() {
        return Err(DemMcmError::ShapeMismatch(format!(
            "polyfit: len(x)={} != len(y)={}",
            x.len(),
            y.len()
        )));
    }
    let m = x.len();
    let k = degree + 1;
    if m < k {
        return Err(DemMcmError::InvalidArgument(format!(
            "polyfit: need at least {k} points for degree {degree}, got {m}"
        )));
    }

    // Build Vandermonde matrix V[i, j] = x[i]^(degree - j)
    let mut r_mat = Matrix::zeros(m, k);
    for (i, &xi) in x.iter().enumerate() {
        let mut p = 1.0;
        for j in (0..k).rev() {
            r_mat.set(i, j, p);
            p *= xi;
        }
    }
    let mut rhs = y.to_vec();

    // Householder QR in-place on r_mat and rhs
    for col in 0..k {
        let mut norm2 = 0.0;
        for row in col..m {
            let val = r_mat.get(row, col);
            norm2 += val * val;
        }
        let norm = norm2.sqrt();
        if norm < 1e-15 {
            continue;
        }
        let x0 = r_mat.get(col, col);
        let alpha = if x0 >= 0.0 { -norm } else { norm };
        let len = m - col;
        let mut u = vec![0.0; len];
        for (i, u_i) in u.iter_mut().enumerate() {
            *u_i = r_mat.get(col + i, col);
        }
        u[0] -= alpha;
        let u_norm2: f64 = u.iter().map(|v| v * v).sum();
        if u_norm2 < 1e-30 {
            continue;
        }
        let beta = 2.0 / u_norm2;

        for j in col..k {
            let mut dot = 0.0;
            for (i, &u_i) in u.iter().enumerate() {
                dot += u_i * r_mat.get(col + i, j);
            }
            let sc = beta * dot;
            for (i, &u_i) in u.iter().enumerate() {
                let cur = r_mat.get(col + i, j);
                r_mat.set(col + i, j, cur - sc * u_i);
            }
        }
        let mut dot_y = 0.0;
        for (i, &u_i) in u.iter().enumerate() {
            dot_y += u_i * rhs[col + i];
        }
        let sc_y = beta * dot_y;
        for (i, &u_i) in u.iter().enumerate() {
            rhs[col + i] -= sc_y * u_i;
        }
    }

    // Back-substitution on upper-triangular R[0..k, 0..k]
    let mut coeffs = vec![0.0; k];
    for i in (0..k).rev() {
        let mut s = rhs[i];
        for (j, &c_j) in coeffs.iter().enumerate().skip(i + 1) {
            s -= r_mat.get(i, j) * c_j;
        }
        let diag = r_mat.get(i, i);
        coeffs[i] = if diag.abs() > 1e-15 { s / diag } else { 0.0 };
    }
    Ok(coeffs)
}

/// Evaluate polynomial `coeffs[0]*x^deg + ... + coeffs[deg]` via Horner's method.
pub fn polyval(coeffs: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &c in coeffs {
        acc = acc * x + c;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_eigh_matches_analytical() {
        // Symmetric 3x3 matrix with known eigenvalues
        let a = Matrix::from_rows(&[
            vec![2.0, -1.0, 0.0],
            vec![-1.0, 2.0, -1.0],
            vec![0.0, -1.0, 2.0],
        ])
        .unwrap();
        let (evals, evecs) = jacobi_eigh(&a).unwrap();
        let s2 = std::f64::consts::SQRT_2;
        let expected = [2.0 - s2, 2.0, 2.0 + s2];
        for (got, exp) in evals.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-10, "got {got}, expected {exp}");
        }
        // Check A * V == V * diag(evals)
        let av = a.mat_mul(&evecs).unwrap();
        for (c, &ev) in evals.iter().enumerate() {
            for r in 0..3 {
                assert!((av.get(r, c) - ev * evecs.get(r, c)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn thin_svd_reconstructs_matrix() {
        let a = Matrix::from_rows(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0],
            vec![2.0, 1.0, 0.5],
        ])
        .unwrap();
        let (u, s, vt) = thin_svd(&a).unwrap();
        assert!(s[0] >= s[1] && s[1] >= s[2]);
        let mut us = Matrix::zeros(4, 3);
        for r in 0..4 {
            for (c, &sv) in s.iter().enumerate() {
                us.set(r, c, u.get(r, c) * sv);
            }
        }
        let rec = us.mat_mul(&vt).unwrap();
        for i in 0..a.data.len() {
            assert!((rec.data[i] - a.data[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn polyfit_recovers_quadratic() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&t| 2.5 * t * t - 3.0 * t + 7.0).collect();
        let c = polyfit(&x, &y, 2).unwrap();
        assert!((c[0] - 2.5).abs() < 1e-9);
        assert!((c[1] + 3.0).abs() < 1e-9);
        assert!((c[2] - 7.0).abs() < 1e-9);
    }

    #[test]
    fn real_eigvals_handles_complex_and_stochastic() {
        // Rotation matrix by 90 degrees + scaling 0.8 => eigenvalues ±0.8i
        let rot = Matrix::from_rows(&[vec![0.0, -0.8], vec![0.8, 0.0]]).unwrap();
        let mags = real_eigvals_abs(&rot).unwrap();
        assert!((mags[0] - 0.8).abs() < 1e-10);
        assert!((mags[1] - 0.8).abs() < 1e-10);

        // 4x4 row-stochastic transition matrix
        let p = Matrix::from_rows(&[
            vec![0.7, 0.2, 0.1, 0.0],
            vec![0.1, 0.6, 0.2, 0.1],
            vec![0.0, 0.3, 0.5, 0.2],
            vec![0.1, 0.0, 0.2, 0.7],
        ])
        .unwrap();
        let mags_p = real_eigvals_abs(&p.transpose()).unwrap();
        let max_mag = mags_p.into_iter().fold(0.0f64, f64::max);
        assert!((max_mag - 1.0).abs() < 1e-10);

        let pi = stationary_distribution_left(&p).unwrap();
        let pi_p = p.vec_mul(&pi).unwrap();
        for (a, b) in pi.iter().zip(pi_p.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn real_eigvals_handles_zero_row_stochastic_matrix() {
        // Sparse transition matrix with multiple unvisited zero rows (exact 0 eigenvalues):
        // eigenvalues must deflate cleanly so |lambda|_max == 1.0 (not fallback norm bound).
        let mut p = Matrix::zeros(8, 8);
        p.set(0, 0, 0.7);
        p.set(0, 2, 0.3);
        p.set(2, 0, 0.25);
        p.set(2, 2, 0.5);
        p.set(2, 5, 0.25);
        p.set(5, 2, 0.4);
        p.set(5, 5, 0.5);
        p.set(5, 7, 0.1);
        p.set(7, 5, 0.3);
        p.set(7, 7, 0.7);
        // rows 1, 3, 4, 6 remain all zeros
        let mags = real_eigvals_abs(&p.transpose()).unwrap();
        let max_mag = mags.iter().copied().fold(0.0f64, f64::max);
        assert!(
            (max_mag - 1.0).abs() < 1e-10,
            "expected |lambda|_max=1.0, got {max_mag}"
        );
        let n_zeros = mags.iter().filter(|&&m| m < 1e-10).count();
        assert_eq!(n_zeros, 4);
    }
}
