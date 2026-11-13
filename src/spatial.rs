//! Exact multi-dimensional KD-tree for nearest-neighbour and radius queries.
//!
//! Used by Voronoi, Physics-aware, Full-velocity, Spectral, Spectral-biclustering,
//! DBSCAN, and Octree fallback assignments.

use std::cmp::Ordering;

/// Balanced KD-tree storing points of fixed runtime dimension `dim`.
#[derive(Debug, Clone)]
pub struct KdTree {
    dim: usize,
    points: Vec<f64>,
    root: Option<Box<KdNode>>,
}

#[derive(Debug, Clone)]
struct KdNode {
    index: usize,
    axis: usize,
    split_val: f64,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
}

impl KdTree {
    /// Build a balanced KD-tree over `n_points` flattened row-major in `points`.
    pub fn build(points: &[f64], dim: usize) -> Self {
        assert!(dim > 0, "KD-tree dimension must be positive");
        assert_eq!(
            points.len() % dim,
            0,
            "points length must be multiple of dim"
        );
        let n = points.len() / dim;
        let owned = points.to_vec();
        let mut indices: Vec<usize> = (0..n).collect();
        let root = Self::build_rec(&owned, dim, &mut indices);
        Self {
            dim,
            points: owned,
            root,
        }
    }

    /// Convenience constructor from `&[Vec<f64>]`.
    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        if rows.is_empty() {
            return Self {
                dim: 1,
                points: Vec::new(),
                root: None,
            };
        }
        let dim = rows[0].len();
        let mut flat = Vec::with_capacity(rows.len() * dim);
        for r in rows {
            assert_eq!(r.len(), dim);
            flat.extend_from_slice(r);
        }
        Self::build(&flat, dim)
    }

    /// Convenience constructor from `&[[f64; 3]]`.
    pub fn from_vec3(rows: &[[f64; 3]]) -> Self {
        let mut flat = Vec::with_capacity(rows.len() * 3);
        for r in rows {
            flat.extend_from_slice(r);
        }
        Self::build(&flat, 3)
    }

    pub fn len(&self) -> usize {
        self.points.len() / self.dim
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    fn build_rec(points: &[f64], dim: usize, indices: &mut [usize]) -> Option<Box<KdNode>> {
        if indices.is_empty() {
            return None;
        }
        if indices.len() == 1 {
            let idx = indices[0];
            return Some(Box::new(KdNode {
                index: idx,
                axis: 0,
                split_val: points[idx * dim],
                left: None,
                right: None,
            }));
        }

        // Choose split axis with largest range among points in this subtree.
        let mut best_axis = 0usize;
        let mut best_spread = -1.0f64;
        for a in 0..dim {
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for &idx in indices.iter() {
                let v = points[idx * dim + a];
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
            let spread = max_v - min_v;
            if spread > best_spread {
                best_spread = spread;
                best_axis = a;
            }
        }

        let mid = indices.len() / 2;
        indices.select_nth_unstable_by(mid, |&a, &b| {
            let va = points[a * dim + best_axis];
            let vb = points[b * dim + best_axis];
            va.partial_cmp(&vb).unwrap_or(Ordering::Equal)
        });
        let median_idx = indices[mid];
        let split_val = points[median_idx * dim + best_axis];
        let (left_slice, right_with_mid) = indices.split_at_mut(mid);
        let right_slice = &mut right_with_mid[1..];

        Some(Box::new(KdNode {
            index: median_idx,
            axis: best_axis,
            split_val,
            left: Self::build_rec(points, dim, left_slice),
            right: Self::build_rec(points, dim, right_slice),
        }))
    }

    #[inline]
    fn sq_dist(&self, a_idx: usize, query: &[f64]) -> f64 {
        let base = a_idx * self.dim;
        self.points[base..base + self.dim]
            .iter()
            .zip(query.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum()
    }

    /// Find `(nearest_index, squared_distance)` for `query`.
    pub fn query_nearest(&self, query: &[f64]) -> Option<(usize, f64)> {
        assert_eq!(query.len(), self.dim);
        let root = self.root.as_deref()?;
        let mut best_idx = root.index;
        let mut best_sq = f64::INFINITY;
        self.search_nearest(root, query, &mut best_idx, &mut best_sq);
        Some((best_idx, best_sq))
    }

    fn search_nearest(
        &self,
        node: &KdNode,
        query: &[f64],
        best_idx: &mut usize,
        best_sq: &mut f64,
    ) {
        let d2 = self.sq_dist(node.index, query);
        if d2 < *best_sq || (d2 == *best_sq && node.index < *best_idx) {
            *best_sq = d2;
            *best_idx = node.index;
        }

        let diff = query[node.axis] - node.split_val;
        let (near, far) = if diff <= 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(n) = near.as_deref() {
            self.search_nearest(n, query, best_idx, best_sq);
        }
        if diff * diff <= *best_sq {
            if let Some(f) = far.as_deref() {
                self.search_nearest(f, query, best_idx, best_sq);
            }
        }
    }

    /// Query nearest neighbour index for every point in flattened `queries`.
    pub fn query_nearest_batch(&self, queries: &[f64]) -> Vec<usize> {
        assert_eq!(queries.len() % self.dim, 0);
        let n = queries.len() / self.dim;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let q = &queries[i * self.dim..(i + 1) * self.dim];
            let (idx, _) = self.query_nearest(q).expect("KD-tree must not be empty");
            out.push(idx);
        }
        out
    }

    /// Find all point indices within Euclidean distance `<= radius` of `query`.
    pub fn query_radius(&self, query: &[f64], radius: f64) -> Vec<usize> {
        assert_eq!(query.len(), self.dim);
        let mut out = Vec::new();
        if let Some(root) = self.root.as_deref() {
            self.search_radius(root, query, radius * radius, &mut out);
        }
        out
    }

    fn search_radius(&self, node: &KdNode, query: &[f64], r2: f64, out: &mut Vec<usize>) {
        if self.sq_dist(node.index, query) <= r2 {
            out.push(node.index);
        }
        let diff = query[node.axis] - node.split_val;
        let (near, far) = if diff <= 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };
        if let Some(n) = near.as_deref() {
            self.search_radius(n, query, r2, out);
        }
        if diff * diff <= r2 {
            if let Some(f) = far.as_deref() {
                self.search_radius(f, query, r2, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn nearest_and_radius_match_bruteforce() {
        let mut rng = Rng::seed_from_u64(7);
        let n = 400;
        let dim = 4;
        let pts: Vec<f64> = (0..n * dim).map(|_| rng.next_normal()).collect();
        let tree = KdTree::build(&pts, dim);

        for _ in 0..40 {
            let q: Vec<f64> = (0..dim).map(|_| rng.next_normal()).collect();
            let (got_idx, got_d2) = tree.query_nearest(&q).unwrap();

            let mut best_idx = 0;
            let mut best_d2 = f64::INFINITY;
            for i in 0..n {
                let d2: f64 = (0..dim).map(|d| (pts[i * dim + d] - q[d]).powi(2)).sum();
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_idx = i;
                }
            }
            assert_eq!(got_idx, best_idx);
            assert!((got_d2 - best_d2).abs() < 1e-12);

            let radius = 1.5;
            let mut got_r = tree.query_radius(&q, radius);
            got_r.sort_unstable();
            let mut exp_r: Vec<usize> = (0..n)
                .filter(|&i| {
                    let d2: f64 = (0..dim).map(|d| (pts[i * dim + d] - q[d]).powi(2)).sum();
                    d2 <= radius * radius
                })
                .collect();
            exp_r.sort_unstable();
            assert_eq!(got_r, exp_r);
        }
    }
}
