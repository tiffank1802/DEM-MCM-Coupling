//! Machine learning primitives implemented in pure Rust:
//! - [`KMeans`] (`k-means++`, multiple restarts, best inertia)
//! - [`MiniBatchKMeans`]
//! - [`Dbscan`] (backed by [`KdTree::query_radius`])
//! - [`GaussianMixture`] (full covariance EM, Cholesky log-likelihoods)
//! - [`SpectralClustering`] (sparse k-NN RBF Laplacian + Rayleigh-Ritz subspace iteration)
//! - [`SpectralBiclustering`] (Kluger log/bistochastic/scale SVD biclustering)
//! - [`LinearSvc`] (dual coordinate descent linear SVM for oblique octree cuts)

use std::collections::{HashMap, VecDeque};

use crate::error::{DemMcmError, Result};
use crate::linalg::{cholesky, jacobi_eigh, solve_lower_triangular, thin_svd, Matrix};
use crate::rng::Rng;
use crate::spatial::KdTree;

// ============================================================================
// K-MEANS & MINI-BATCH K-MEANS
// ============================================================================

/// Result of fitting K-means.
#[derive(Debug, Clone)]
pub struct KMeansModel {
    pub centroids: Matrix,
    pub labels: Vec<usize>,
    pub inertia: f64,
}

/// Standard Lloyd's K-means with `k-means++` initialization and `n_init` restarts.
#[derive(Debug, Clone)]
pub struct KMeans {
    pub n_clusters: usize,
    pub n_init: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub random_state: u64,
}

impl KMeans {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            n_init: 10,
            max_iter: 300,
            tol: 1e-4,
            random_state: 42,
        }
    }

    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init.max(1);
        self
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = seed;
        self
    }

    /// Fit K-means on `data` of shape `(n_samples, n_features)`.
    pub fn fit(&self, data: &Matrix) -> Result<KMeansModel> {
        let (n, d) = data.shape();
        if n == 0 || d == 0 {
            return Err(DemMcmError::InvalidArgument(
                "KMeans::fit called on empty matrix".to_string(),
            ));
        }
        if self.n_clusters == 0 {
            return Err(DemMcmError::InvalidArgument(
                "KMeans n_clusters must be >= 1".to_string(),
            ));
        }
        let k = self.n_clusters.min(n);

        // If fewer samples than requested clusters, pad centroids deterministically.
        let mut rng = Rng::seed_from_u64(self.random_state);
        let mut best_model: Option<KMeansModel> = None;

        for _init_idx in 0..self.n_init {
            let mut centroids = self.kmeans_plus_plus(data, k, &mut rng);
            let mut labels = vec![0usize; n];
            let mut prev_inertia = f64::INFINITY;
            let mut inertia = f64::INFINITY;

            for _iter in 0..self.max_iter {
                inertia = 0.0;
                // Assignment step
                for (i, label_i) in labels.iter_mut().enumerate() {
                    let x = data.row(i);
                    let mut best_c = 0usize;
                    let mut best_d2 = f64::INFINITY;
                    for c in 0..k {
                        let cen = centroids.row(c);
                        let mut d2 = 0.0;
                        for j in 0..d {
                            let diff = x[j] - cen[j];
                            d2 += diff * diff;
                        }
                        if d2 < best_d2 {
                            best_d2 = d2;
                            best_c = c;
                        }
                    }
                    *label_i = best_c;
                    inertia += best_d2;
                }

                // Update step
                let mut sums = Matrix::zeros(k, d);
                let mut counts = vec![0usize; k];
                for (i, &c) in labels.iter().enumerate() {
                    counts[c] += 1;
                    let x = data.row(i);
                    let s_row = sums.row_mut(c);
                    for j in 0..d {
                        s_row[j] += x[j];
                    }
                }

                // Recover any empty cluster by relocating it to a high-error sample.
                for (c, &count) in counts.iter().enumerate() {
                    if count == 0 {
                        let pick = rng.gen_range(n);
                        centroids.row_mut(c).copy_from_slice(data.row(pick));
                    } else {
                        let inv = 1.0 / count as f64;
                        let s_row = sums.row(c);
                        let c_row = centroids.row_mut(c);
                        for j in 0..d {
                            c_row[j] = s_row[j] * inv;
                        }
                    }
                }

                if (prev_inertia - inertia).abs() <= self.tol * prev_inertia.max(1.0) {
                    break;
                }
                prev_inertia = inertia;
            }

            if best_model.as_ref().map_or(true, |b| inertia < b.inertia) {
                best_model = Some(KMeansModel {
                    centroids,
                    labels,
                    inertia,
                });
            }
        }

        let mut model = best_model.unwrap();
        // If n < self.n_clusters, pad remaining centroids so shape == (n_clusters, d).
        if k < self.n_clusters {
            let mut full_centroids = Matrix::zeros(self.n_clusters, d);
            for c in 0..self.n_clusters {
                full_centroids
                    .row_mut(c)
                    .copy_from_slice(model.centroids.row(c % k));
            }
            model.centroids = full_centroids;
        }
        Ok(model)
    }

    fn kmeans_plus_plus(&self, data: &Matrix, k: usize, rng: &mut Rng) -> Matrix {
        let (n, d) = data.shape();
        let mut centroids = Matrix::zeros(k, d);

        let first = rng.gen_range(n);
        centroids.row_mut(0).copy_from_slice(data.row(first));

        let mut min_d2 = vec![f64::INFINITY; n];
        for c in 1..k {
            let prev_c = centroids.row(c - 1);
            let mut total = 0.0;
            for (i, md) in min_d2.iter_mut().enumerate() {
                let x = data.row(i);
                let mut d2 = 0.0;
                for j in 0..d {
                    let diff = x[j] - prev_c[j];
                    d2 += diff * diff;
                }
                if d2 < *md {
                    *md = d2;
                }
                total += *md;
            }

            let chosen = if total <= 0.0 {
                rng.gen_range(n)
            } else {
                let target = rng.next_f64() * total;
                let mut cum = 0.0;
                let mut idx = n - 1;
                for (i, &w) in min_d2.iter().enumerate() {
                    cum += w;
                    if cum >= target {
                        idx = i;
                        break;
                    }
                }
                idx
            };
            centroids.row_mut(c).copy_from_slice(data.row(chosen));
        }

        centroids
    }
}

/// Mini-batch K-means (matching `FullVectorVelocityKMeansPartitioner`).
#[derive(Debug, Clone)]
pub struct MiniBatchKMeans {
    pub n_clusters: usize,
    pub batch_size: usize,
    pub n_init: usize,
    pub max_iter: usize,
    pub random_state: u64,
}

impl MiniBatchKMeans {
    pub fn new(n_clusters: usize, batch_size: usize, random_state: u64) -> Self {
        Self {
            n_clusters,
            batch_size: batch_size.max(1),
            n_init: 10,
            max_iter: 100,
            random_state,
        }
    }

    pub fn fit(&self, data: &Matrix) -> Result<KMeansModel> {
        let (n, d) = data.shape();
        if n == 0 || d == 0 {
            return Err(DemMcmError::InvalidArgument(
                "MiniBatchKMeans::fit called on empty matrix".to_string(),
            ));
        }
        let k = self.n_clusters.min(n);
        let batch = self.batch_size.min(n);

        let mut rng = Rng::seed_from_u64(self.random_state);
        let base_km = KMeans::new(k)
            .with_n_init(1)
            .with_random_state(self.random_state);

        let mut best_model: Option<KMeansModel> = None;

        for init_i in 0..self.n_init.max(1) {
            let mut rng_init = Rng::seed_from_u64(self.random_state.wrapping_add(init_i as u64));
            let mut centroids = base_km.kmeans_plus_plus(data, k, &mut rng_init);
            let mut counts = vec![0usize; k];

            for _step in 0..self.max_iter {
                let batch_idx = rng.sample_indices(n, batch);
                let mut nearest = Vec::with_capacity(batch_idx.len());
                for &i in &batch_idx {
                    let x = data.row(i);
                    let mut best_c = 0usize;
                    let mut best_d2 = f64::INFINITY;
                    for c in 0..k {
                        let cen = centroids.row(c);
                        let mut d2 = 0.0;
                        for j in 0..d {
                            let diff = x[j] - cen[j];
                            d2 += diff * diff;
                        }
                        if d2 < best_d2 {
                            best_d2 = d2;
                            best_c = c;
                        }
                    }
                    nearest.push((i, best_c));
                }

                for (i, c) in nearest {
                    counts[c] += 1;
                    let eta = 1.0 / counts[c] as f64;
                    let x = data.row(i);
                    let cen = centroids.row_mut(c);
                    for j in 0..d {
                        cen[j] = (1.0 - eta) * cen[j] + eta * x[j];
                    }
                }
            }

            // Evaluate full inertia
            let mut labels = vec![0usize; n];
            let mut inertia = 0.0;
            for (i, label_i) in labels.iter_mut().enumerate() {
                let x = data.row(i);
                let mut best_c = 0usize;
                let mut best_d2 = f64::INFINITY;
                for c in 0..k {
                    let cen = centroids.row(c);
                    let mut d2 = 0.0;
                    for j in 0..d {
                        let diff = x[j] - cen[j];
                        d2 += diff * diff;
                    }
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_c = c;
                    }
                }
                *label_i = best_c;
                inertia += best_d2;
            }

            if best_model.as_ref().map_or(true, |b| inertia < b.inertia) {
                best_model = Some(KMeansModel {
                    centroids,
                    labels,
                    inertia,
                });
            }
        }

        let mut model = best_model.unwrap();
        if k < self.n_clusters {
            let mut full_centroids = Matrix::zeros(self.n_clusters, d);
            for c in 0..self.n_clusters {
                full_centroids
                    .row_mut(c)
                    .copy_from_slice(model.centroids.row(c % k));
            }
            model.centroids = full_centroids;
        }
        Ok(model)
    }
}

// ============================================================================
// DBSCAN
// ============================================================================

/// Density-based spatial clustering (DBSCAN) backed by a KD-tree.
#[derive(Debug, Clone)]
pub struct Dbscan {
    pub eps: f64,
    pub min_samples: usize,
}

impl Dbscan {
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self { eps, min_samples }
    }

    /// Cluster rows of `data`. Returns `(labels, n_clusters)` where `-1` indicates noise.
    pub fn fit_predict(&self, data: &Matrix) -> (Vec<i64>, usize) {
        let (n, d) = data.shape();
        if n == 0 {
            return (Vec::new(), 0);
        }
        let tree = KdTree::build(&data.data, d);
        let mut labels = vec![-2i64; n]; // -2 = unvisited, -1 = noise
        let mut cluster_id = 0i64;

        for i in 0..n {
            if labels[i] != -2 {
                continue;
            }
            let neighbors = tree.query_radius(data.row(i), self.eps);
            if neighbors.len() < self.min_samples {
                labels[i] = -1;
                continue;
            }

            labels[i] = cluster_id;
            let mut queue: VecDeque<usize> = neighbors.into_iter().collect();
            while let Some(q) = queue.pop_front() {
                if labels[q] == -1 {
                    labels[q] = cluster_id;
                }
                if labels[q] != -2 {
                    continue;
                }
                labels[q] = cluster_id;
                let q_neighbors = tree.query_radius(data.row(q), self.eps);
                if q_neighbors.len() >= self.min_samples {
                    for nb in q_neighbors {
                        if labels[nb] == -2 || labels[nb] == -1 {
                            queue.push_back(nb);
                        }
                    }
                }
            }
            cluster_id += 1;
        }

        (labels, cluster_id as usize)
    }
}

// ============================================================================
// GAUSSIAN MIXTURE MODEL (FULL COVARIANCE)
// ============================================================================

/// Fitted full-covariance Gaussian Mixture Model.
#[derive(Debug, Clone)]
pub struct GaussianMixtureModel {
    pub n_components: usize,
    pub n_features: usize,
    pub weights: Vec<f64>,
    pub means: Matrix,
    pub covariances: Vec<Matrix>,
    cholesky_l: Vec<Matrix>,
    log_det_half: Vec<f64>,
}

/// Full-covariance Gaussian Mixture Model trainer (EM initialized with K-means).
#[derive(Debug, Clone)]
pub struct GaussianMixture {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub reg_covar: f64,
    pub random_state: u64,
}

impl GaussianMixture {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 50,
            tol: 1e-3,
            reg_covar: 1e-6,
            random_state: 42,
        }
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = seed;
        self
    }

    pub fn fit(&self, data: &Matrix) -> Result<GaussianMixtureModel> {
        let (n, d) = data.shape();
        if n == 0 || d == 0 {
            return Err(DemMcmError::InvalidArgument(
                "GaussianMixture::fit called on empty data".to_string(),
            ));
        }
        let k = self.n_components;
        let km = KMeans::new(k)
            .with_n_init(1)
            .with_random_state(self.random_state)
            .fit(data)?;

        let mut means = km.centroids;
        let mut weights = vec![0.0; k];
        let mut covariances = vec![Matrix::zeros(d, d); k];

        // Initialise responsibilities hard from KMeans labels
        let mut resp = Matrix::zeros(n, k);
        for (i, &lbl) in km.labels.iter().enumerate() {
            resp.set(i, lbl, 1.0);
        }

        Self::m_step(
            data,
            &resp,
            &mut weights,
            &mut means,
            &mut covariances,
            self.reg_covar,
        );

        let mut prev_ll = f64::NEG_INFINITY;
        let mut cholesky_l = Vec::with_capacity(k);
        let mut log_det_half = Vec::with_capacity(k);

        for _iter in 0..self.max_iter {
            Self::compute_cholesky_factors(
                &covariances,
                self.reg_covar,
                &mut cholesky_l,
                &mut log_det_half,
            )?;

            let ll = Self::e_step(
                data,
                &weights,
                &means,
                &cholesky_l,
                &log_det_half,
                &mut resp,
            );
            Self::m_step(
                data,
                &resp,
                &mut weights,
                &mut means,
                &mut covariances,
                self.reg_covar,
            );

            if (ll - prev_ll).abs() < self.tol {
                break;
            }
            prev_ll = ll;
        }

        Self::compute_cholesky_factors(
            &covariances,
            self.reg_covar,
            &mut cholesky_l,
            &mut log_det_half,
        )?;

        Ok(GaussianMixtureModel {
            n_components: k,
            n_features: d,
            weights,
            means,
            covariances,
            cholesky_l,
            log_det_half,
        })
    }

    fn compute_cholesky_factors(
        covariances: &[Matrix],
        reg_covar: f64,
        cholesky_out: &mut Vec<Matrix>,
        log_det_half_out: &mut Vec<f64>,
    ) -> Result<()> {
        cholesky_out.clear();
        log_det_half_out.clear();
        for cov in covariances {
            let d = cov.rows;
            let mut reg = cov.clone();
            let mut l_res = cholesky(&reg);
            let mut jitter = reg_covar;
            while l_res.is_err() && jitter < 1.0 {
                for i in 0..d {
                    let cur = reg.get(i, i);
                    reg.set(i, i, cur + jitter);
                }
                l_res = cholesky(&reg);
                jitter *= 10.0;
            }
            let l = l_res?;
            let mut half_ld = 0.0;
            for i in 0..d {
                half_ld += l.get(i, i).ln();
            }
            cholesky_out.push(l);
            log_det_half_out.push(half_ld);
        }
        Ok(())
    }

    fn component_log_prob(
        x: &[f64],
        mean: &[f64],
        l: &Matrix,
        log_det_half: f64,
        log_weight: f64,
    ) -> f64 {
        let d = x.len();
        let mut diff = vec![0.0; d];
        for j in 0..d {
            diff[j] = x[j] - mean[j];
        }
        let y = solve_lower_triangular(l, &diff);
        let maha: f64 = y.iter().map(|v| v * v).sum();
        let const_term = -0.5 * (d as f64) * (2.0 * std::f64::consts::PI).ln();
        log_weight + const_term - log_det_half - 0.5 * maha
    }

    fn e_step(
        data: &Matrix,
        weights: &[f64],
        means: &Matrix,
        cholesky_l: &[Matrix],
        log_det_half: &[f64],
        resp: &mut Matrix,
    ) -> f64 {
        let (n, _d) = data.shape();
        let k = weights.len();
        let log_weights: Vec<f64> = weights
            .iter()
            .map(|&w| if w > 1e-15 { w.ln() } else { -100.0 })
            .collect();

        let mut total_ll = 0.0;
        for i in 0..n {
            let x = data.row(i);
            let r_row = resp.row_mut(i);
            let mut max_lp = f64::NEG_INFINITY;
            for c in 0..k {
                let lp = Self::component_log_prob(
                    x,
                    means.row(c),
                    &cholesky_l[c],
                    log_det_half[c],
                    log_weights[c],
                );
                r_row[c] = lp;
                if lp > max_lp {
                    max_lp = lp;
                }
            }
            let mut sum_exp = 0.0;
            for val in r_row.iter_mut() {
                let e = (*val - max_lp).exp();
                *val = e;
                sum_exp += e;
            }
            total_ll += max_lp + sum_exp.ln();
            if sum_exp > 0.0 {
                let inv = 1.0 / sum_exp;
                for val in r_row.iter_mut() {
                    *val *= inv;
                }
            } else {
                r_row.fill(1.0 / k as f64);
            }
        }
        total_ll / n as f64
    }

    fn m_step(
        data: &Matrix,
        resp: &Matrix,
        weights: &mut [f64],
        means: &mut Matrix,
        covariances: &mut [Matrix],
        reg_covar: f64,
    ) {
        let (n, d) = data.shape();
        let k = weights.len();
        let mut nk = vec![0.0; k];
        for i in 0..n {
            let r_row = resp.row(i);
            for (c, &r_ic) in r_row.iter().enumerate() {
                nk[c] += r_ic;
            }
        }

        let n_f = n as f64;
        for c in 0..k {
            weights[c] = (nk[c] / n_f).max(1e-12);
            let denom = nk[c] + 10.0 * f64::EPSILON;
            let m_row = means.row_mut(c);
            m_row.fill(0.0);
            for i in 0..n {
                let r = resp.get(i, c);
                let x = data.row(i);
                for j in 0..d {
                    m_row[j] += r * x[j];
                }
            }
            for val in m_row.iter_mut() {
                *val /= denom;
            }

            let cov = &mut covariances[c];
            cov.data.fill(0.0);
            for i in 0..n {
                let r = resp.get(i, c);
                let x = data.row(i);
                for a in 0..d {
                    let da = x[a] - m_row[a];
                    for b in a..d {
                        let db = x[b] - m_row[b];
                        let cur = cov.get(a, b);
                        cov.set(a, b, cur + r * da * db);
                    }
                }
            }
            for a in 0..d {
                for b in a..d {
                    let mut val = cov.get(a, b) / denom;
                    if a == b {
                        val += reg_covar;
                    }
                    cov.set(a, b, val);
                    cov.set(b, a, val);
                }
            }
        }
        let w_sum: f64 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= w_sum;
        }
    }
}

impl GaussianMixtureModel {
    /// Reconstruct a [`GaussianMixtureModel`] from saved parameters.
    pub fn from_params(
        weights: Vec<f64>,
        means: Matrix,
        covariances: Vec<Matrix>,
        reg_covar: f64,
    ) -> Result<Self> {
        let k = weights.len();
        let d = means.cols;
        let mut cholesky_l = Vec::with_capacity(k);
        let mut log_det_half = Vec::with_capacity(k);
        GaussianMixture::compute_cholesky_factors(
            &covariances,
            reg_covar,
            &mut cholesky_l,
            &mut log_det_half,
        )?;
        Ok(Self {
            n_components: k,
            n_features: d,
            weights,
            means,
            covariances,
            cholesky_l,
            log_det_half,
        })
    }

    /// Predict most probable component for each row in `data`.
    pub fn predict(&self, data: &Matrix) -> Result<Vec<usize>> {
        let (n, d) = data.shape();
        if d != self.n_features {
            return Err(DemMcmError::ShapeMismatch(format!(
                "GMM expects {d} == {} features",
                self.n_features
            )));
        }
        let log_weights: Vec<f64> = self
            .weights
            .iter()
            .map(|&w| if w > 1e-15 { w.ln() } else { -100.0 })
            .collect();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let x = data.row(i);
            let mut best_c = 0usize;
            let mut best_lp = f64::NEG_INFINITY;
            for (c, &lw) in log_weights.iter().enumerate() {
                let lp = GaussianMixture::component_log_prob(
                    x,
                    self.means.row(c),
                    &self.cholesky_l[c],
                    self.log_det_half[c],
                    lw,
                );
                if lp > best_lp {
                    best_lp = lp;
                    best_c = c;
                }
            }
            out.push(best_c);
        }
        Ok(out)
    }
}

// ============================================================================
// SPECTRAL CLUSTERING
// ============================================================================

/// Spectral clustering on a symmetric RBF k-NN graph with normalized Laplacian.
#[derive(Debug, Clone)]
pub struct SpectralClustering {
    pub n_clusters: usize,
    pub n_neighbors: usize,
    pub gamma: f64,
    pub random_state: u64,
}

impl SpectralClustering {
    pub fn new(n_clusters: usize, n_neighbors: usize, random_state: u64) -> Self {
        Self {
            n_clusters,
            n_neighbors,
            gamma: 1.0,
            random_state,
        }
    }

    pub fn fit_predict(&self, data: &Matrix) -> Result<Vec<usize>> {
        let (n, d) = data.shape();
        if n == 0 {
            return Ok(Vec::new());
        }
        let k = self.n_clusters.min(n);
        let knn = self.n_neighbors.min(n.saturating_sub(1)).max(1);

        // Build sparse symmetrized affinity graph W: each entry (j, weight).
        // For N <= 5000 compute pairwise/tree affinities cleanly; for N <= 2000
        // scanning or using a local neighborhood keeps construction fast.
        let mut adj: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];
        // Compute k-nearest neighbours via brute-force partial select for N <= 5000:
        // 5000^2 = 25M ops (~15ms in Rust release / ~120ms debug), exact in any D!
        let mut dists: Vec<(f64, usize)> = Vec::with_capacity(n);
        for i in 0..n {
            dists.clear();
            let xi = data.row(i);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let xj = data.row(j);
                let mut d2 = 0.0;
                for c in 0..d {
                    let diff = xi[c] - xj[c];
                    d2 += diff * diff;
                }
                dists.push((d2, j));
            }
            if knn < dists.len() {
                dists.select_nth_unstable_by(knn, |a, b| a.0.total_cmp(&b.0));
                dists.truncate(knn);
            }
            for &(d2, j) in &dists {
                let w = (-self.gamma * d2).exp();
                adj[i]
                    .entry(j)
                    .and_modify(|e| *e = 0.5 * (*e + w))
                    .or_insert(0.5 * w);
                adj[j]
                    .entry(i)
                    .and_modify(|e| *e = 0.5 * (*e + w))
                    .or_insert(0.5 * w);
            }
        }

        // Convert to normalized adjacency S = D^{-1/2} W D^{-1/2}
        let deg: Vec<f64> = adj
            .iter()
            .map(|row| row.values().sum::<f64>().max(1e-12))
            .collect();
        let inv_sqrt_deg: Vec<f64> = deg.iter().map(|&d| 1.0 / d.sqrt()).collect();

        let sparse_rows: Vec<Vec<(usize, f64)>> = adj
            .into_iter()
            .enumerate()
            .map(|(i, m)| {
                m.into_iter()
                    .map(|(j, w)| (j, w * inv_sqrt_deg[i] * inv_sqrt_deg[j]))
                    .collect()
            })
            .collect();

        // Smallest k eigenvectors of L_sym = I - S are the largest k eigenvectors
        // of M = I + S (positive semidefinite, eigenvalues in [0, 2]).
        let embedding = if n <= 250 {
            let mut l_sym = Matrix::identity(n);
            for (i, row) in sparse_rows.iter().enumerate() {
                for &(j, s_ij) in row {
                    l_sym.set(i, j, l_sym.get(i, j) - s_ij);
                }
            }
            let (_evals, evecs) = jacobi_eigh(&l_sym)?;
            let mut emb = Matrix::zeros(n, k);
            for i in 0..n {
                for c in 0..k {
                    emb.set(i, c, evecs.get(i, c));
                }
            }
            emb
        } else {
            // Subspace iteration with Rayleigh-Ritz on M = I + S
            let r = (k + 8).min(n);
            let mut rng = Rng::seed_from_u64(self.random_state);
            let mut q = Matrix::zeros(n, r);
            for i in 0..n {
                for c in 0..r {
                    q.set(i, c, rng.next_normal());
                }
            }
            orthonormalize_columns(&mut q);

            for _iter in 0..45 {
                let mut z = Matrix::zeros(n, r);
                for (i, row) in sparse_rows.iter().enumerate() {
                    let q_i = q.row(i);
                    let z_i = z.row_mut(i);
                    z_i[..r].copy_from_slice(&q_i[..r]); // I * Q
                    for &(j, s_ij) in row {
                        let q_j = q.row(j);
                        for c in 0..r {
                            z_i[c] += s_ij * q_j[c];
                        }
                    }
                }
                orthonormalize_columns(&mut z);
                // Rayleigh-Ritz projection every 5 iterations
                if _iter % 5 == 4 || _iter == 44 {
                    let mut mz = Matrix::zeros(n, r);
                    for (i, row) in sparse_rows.iter().enumerate() {
                        let z_i = z.row(i);
                        let mz_i = mz.row_mut(i);
                        mz_i[..r].copy_from_slice(&z_i[..r]);
                        for &(j, s_ij) in row {
                            let z_j = z.row(j);
                            for c in 0..r {
                                mz_i[c] += s_ij * z_j[c];
                            }
                        }
                    }
                    let proj = z.transpose().mat_mul(&mz)?;
                    let (evals, evecs) = jacobi_eigh(&proj)?;
                    // Sort descending (largest eigenvalues of M = smallest of L_sym)
                    let mut rot = Matrix::zeros(r, r);
                    for new_c in 0..r {
                        let old_c = r - 1 - new_c;
                        let _ = evals[old_c];
                        for row_idx in 0..r {
                            rot.set(row_idx, new_c, evecs.get(row_idx, old_c));
                        }
                    }
                    q = z.mat_mul(&rot)?;
                } else {
                    q = z;
                }
            }

            let mut emb = Matrix::zeros(n, k);
            for i in 0..n {
                for c in 0..k {
                    emb.set(i, c, q.get(i, c));
                }
            }
            emb
        };

        // Row-normalize spectral embedding (Ng-Jordan-Weiss)
        let mut norm_emb = embedding;
        for i in 0..n {
            let row = norm_emb.row_mut(i);
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }

        let km = KMeans::new(k)
            .with_n_init(10)
            .with_random_state(self.random_state)
            .fit(&norm_emb)?;
        Ok(km.labels)
    }
}

/// Modified Gram-Schmidt column orthonormalization in-place.
fn orthonormalize_columns(mat: &mut Matrix) {
    let (m, n) = mat.shape();
    for c in 0..n {
        for prev in 0..c {
            let mut dot = 0.0;
            for r in 0..m {
                dot += mat.get(r, prev) * mat.get(r, c);
            }
            for r in 0..m {
                let cur = mat.get(r, c);
                mat.set(r, c, cur - dot * mat.get(r, prev));
            }
        }
        let mut norm2 = 0.0;
        for r in 0..m {
            let v = mat.get(r, c);
            norm2 += v * v;
        }
        let norm = norm2.sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for r in 0..m {
                mat.set(r, c, mat.get(r, c) * inv);
            }
        }
    }
}

// ============================================================================
// SPECTRAL BICLUSTERING (KLUGER ET AL. 2003)
// ============================================================================

#[derive(Debug, Clone)]
pub struct SpectralBiclusteringResult {
    pub row_labels: Vec<usize>,
    pub column_labels: Vec<usize>,
}

/// Spectral biclustering of an `(n_samples, n_features)` non-negative matrix.
#[derive(Debug, Clone)]
pub struct SpectralBiclustering {
    pub n_row_clusters: usize,
    pub n_col_clusters: usize,
    pub method: String,
    pub n_components: usize,
    pub n_best: usize,
    pub random_state: u64,
}

impl SpectralBiclustering {
    pub fn fit_predict(&self, data: &Matrix) -> Result<SpectralBiclusteringResult> {
        let (m, n) = data.shape();
        if m == 0 || n == 0 {
            return Err(DemMcmError::InvalidArgument(
                "SpectralBiclustering called on empty matrix".to_string(),
            ));
        }

        // Normalize matrix according to Kluger method ("log", "bistochastic", "scale")
        let normalized = match self.method.as_str() {
            "log" => {
                let mut l = Matrix::zeros(m, n);
                let mut row_means = vec![0.0; m];
                let mut col_means = vec![0.0; n];
                let mut total = 0.0;
                for (i, rm) in row_means.iter_mut().enumerate() {
                    for (j, cm) in col_means.iter_mut().enumerate() {
                        let v = data.get(i, j).max(1e-12).ln();
                        l.set(i, j, v);
                        *rm += v;
                        *cm += v;
                        total += v;
                    }
                }
                for rm in &mut row_means {
                    *rm /= n as f64;
                }
                for cm in &mut col_means {
                    *cm /= m as f64;
                }
                let grand = total / (m * n) as f64;
                for (i, &rm) in row_means.iter().enumerate() {
                    for (j, &cm) in col_means.iter().enumerate() {
                        let k_ij = l.get(i, j) - rm - cm + grand;
                        l.set(i, j, k_ij);
                    }
                }
                l
            }
            "bistochastic" | "scale" => {
                // D_r^{-1/2} X D_c^{-1/2} (Sinkhorn for bistochastic, single step for scale)
                let mut x = data.clone();
                let iters = if self.method == "bistochastic" { 10 } else { 1 };
                for _ in 0..iters {
                    let rs: Vec<f64> = x
                        .row_sums()
                        .into_iter()
                        .map(|s| 1.0 / s.abs().max(1e-12).sqrt())
                        .collect();
                    for (i, &r_scale) in rs.iter().enumerate() {
                        for j in 0..n {
                            x.set(i, j, x.get(i, j) * r_scale);
                        }
                    }
                    let cs: Vec<f64> = x
                        .col_sums()
                        .into_iter()
                        .map(|s| 1.0 / s.abs().max(1e-12).sqrt())
                        .collect();
                    for i in 0..m {
                        for (j, &c_scale) in cs.iter().enumerate() {
                            x.set(i, j, x.get(i, j) * c_scale);
                        }
                    }
                }
                x
            }
            other => {
                return Err(DemMcmError::InvalidArgument(format!(
                    "Unknown SpectralBiclustering method: {other:?}"
                )));
            }
        };

        let (u, _s, vt) = thin_svd(&normalized)?;
        let rank = u.cols.min(vt.rows);
        let skip = if self.method == "log" {
            0
        } else {
            1.min(rank.saturating_sub(1))
        };
        let take = (self.n_best.min(rank - skip)).max(1);

        // Build row & col spectral representations from top `take` components.
        let mut row_feat = Matrix::zeros(m, take);
        for i in 0..m {
            for c in 0..take {
                row_feat.set(i, c, u.get(i, skip + c));
            }
        }
        let mut col_feat = Matrix::zeros(n, take);
        for j in 0..n {
            for c in 0..take {
                col_feat.set(j, c, vt.get(skip + c, j));
            }
        }

        let km_row = KMeans::new(self.n_row_clusters.min(m))
            .with_n_init(10)
            .with_random_state(self.random_state)
            .fit(&row_feat)?;
        let km_col = KMeans::new(self.n_col_clusters.min(n))
            .with_n_init(10)
            .with_random_state(self.random_state)
            .fit(&col_feat)?;

        Ok(SpectralBiclusteringResult {
            row_labels: km_row.labels,
            column_labels: km_col.labels,
        })
    }
}

// ============================================================================
// LINEAR SVC (DUAL COORDINATE DESCENT)
// ============================================================================

/// L2-regularized L1-hinge-loss Linear Support Vector Classifier.
///
/// Used by [`crate::partitioners::OctreePartitioner`] (`oblique_method = "svm"`).
#[derive(Debug, Clone)]
pub struct LinearSvc {
    pub c: f64,
    pub max_iter: usize,
    pub random_state: u64,
}

impl LinearSvc {
    pub fn new(c: f64) -> Self {
        Self {
            c,
            max_iter: 1000,
            random_state: 42,
        }
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = seed;
        self
    }

    /// Fit binary classifier for labels `0` / `1` on `data`. Returns `(weights, intercept)`.
    pub fn fit(&self, data: &Matrix, labels: &[i32]) -> Result<(Vec<f64>, f64)> {
        let (n, d) = data.shape();
        if n != labels.len() {
            return Err(DemMcmError::ShapeMismatch(
                "LinearSvc: data rows != labels len".to_string(),
            ));
        }
        if n == 0 {
            return Ok((vec![1.0; d], 0.0));
        }

        // Map labels to {-1.0, +1.0}, augment feature vector with bias term 1.0
        let y: Vec<f64> = labels
            .iter()
            .map(|&l| if l > 0 { 1.0 } else { -1.0 })
            .collect();
        let d_aug = d + 1;
        let mut w = vec![0.0; d_aug];
        let mut alpha = vec![0.0; n];
        let mut q_diag = vec![0.0; n];
        for (i, qd) in q_diag.iter_mut().enumerate() {
            let row = data.row(i);
            let sq: f64 = row.iter().map(|&v| v * v).sum::<f64>() + 1.0; // bias=1^2
            *qd = sq;
        }

        let mut rng = Rng::seed_from_u64(self.random_state);
        let mut order: Vec<usize> = (0..n).collect();

        for _iter in 0..self.max_iter {
            // Shuffle index order
            for i in (1..n).rev() {
                let j = rng.gen_range(i + 1);
                order.swap(i, j);
            }
            let mut max_pg = 0.0f64;

            for &i in &order {
                let row = data.row(i);
                let mut dot = w[d]; // bias
                for j in 0..d {
                    dot += w[j] * row[j];
                }
                let g = y[i] * dot - 1.0;
                let pg = if alpha[i] <= 0.0 {
                    g.min(0.0)
                } else if alpha[i] >= self.c {
                    g.max(0.0)
                } else {
                    g
                };
                max_pg = max_pg.max(pg.abs());

                if pg.abs() > 1e-12 && q_diag[i] > 1e-15 {
                    let old_alpha = alpha[i];
                    let new_alpha = (old_alpha - g / q_diag[i]).clamp(0.0, self.c);
                    let delta = (new_alpha - old_alpha) * y[i];
                    if delta.abs() > 1e-15 {
                        for j in 0..d {
                            w[j] += delta * row[j];
                        }
                        w[d] += delta;
                        alpha[i] = new_alpha;
                    }
                }
            }
            if max_pg < 1e-4 {
                break;
            }
        }

        let intercept = w[d];
        w.truncate(d);
        Ok((w, intercept))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_separates_two_clusters() {
        let rows = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.1, 0.1, 0.0],
            vec![-0.1, 0.0, 0.1],
            vec![10.0, 10.0, 10.0],
            vec![10.1, 9.9, 10.0],
            vec![9.9, 10.1, 10.0],
        ];
        let m = Matrix::from_rows(&rows).unwrap();
        let model = KMeans::new(2).fit(&m).unwrap();
        assert_eq!(model.labels[0], model.labels[1]);
        assert_eq!(model.labels[1], model.labels[2]);
        assert_eq!(model.labels[3], model.labels[4]);
        assert_eq!(model.labels[4], model.labels[5]);
        assert_ne!(model.labels[0], model.labels[3]);
    }

    #[test]
    fn dbscan_identifies_cluster_and_noise() {
        let rows = vec![
            vec![0.0, 0.0],
            vec![0.05, 0.0],
            vec![0.0, 0.05],
            vec![0.05, 0.05],
            vec![5.0, 5.0],
            vec![5.04, 5.0],
            vec![5.0, 5.04],
            vec![5.04, 5.04],
            vec![100.0, 100.0], // noise
        ];
        let m = Matrix::from_rows(&rows).unwrap();
        let (labels, n_clusters) = Dbscan::new(0.2, 3).fit_predict(&m);
        assert_eq!(n_clusters, 2);
        assert_eq!(labels[8], -1);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn gmm_separates_gaussians() {
        let mut rng = Rng::seed_from_u64(99);
        let mut rows = Vec::new();
        for _ in 0..80 {
            rows.push(vec![
                rng.next_normal() * 0.2,
                rng.next_normal() * 0.2,
                rng.next_normal() * 0.2,
            ]);
        }
        for _ in 0..80 {
            rows.push(vec![
                5.0 + rng.next_normal() * 0.2,
                5.0 + rng.next_normal() * 0.2,
                5.0 + rng.next_normal() * 0.2,
            ]);
        }
        let m = Matrix::from_rows(&rows).unwrap();
        let model = GaussianMixture::new(2).fit(&m).unwrap();
        let preds = model.predict(&m).unwrap();
        assert!(preds[..80].iter().all(|&p| p == preds[0]));
        assert!(preds[80..].iter().all(|&p| p == preds[80]));
        assert_ne!(preds[0], preds[80]);
    }

    #[test]
    fn linear_svc_separates_halfspaces() {
        let rows = vec![
            vec![-2.0, 0.0, 0.0],
            vec![-1.0, 1.0, -1.0],
            vec![1.0, -1.0, 1.0],
            vec![2.0, 0.0, 0.0],
        ];
        let m = Matrix::from_rows(&rows).unwrap();
        let (w, b) = LinearSvc::new(1.0).fit(&m, &[0, 0, 1, 1]).unwrap();
        assert!(w[0] > 0.0);
        for (i, &lbl) in [0, 0, 1, 1].iter().enumerate() {
            let score = w[0] * rows[i][0] + w[1] * rows[i][1] + w[2] * rows[i][2] + b;
            if lbl == 0 {
                assert!(score < 0.0);
            } else {
                assert!(score > 0.0);
            }
        }
    }
}
