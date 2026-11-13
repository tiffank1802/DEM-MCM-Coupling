//! Deterministic pseudo-random number generation (SplitMix64 + Xoshiro256++).
//!
//! Provides reproducible sampling without replacement and standard normal variates
//! used by K-means++, GMM, octree random/2medians cuts, and subsampling.

use std::collections::HashSet;

/// Deterministic PRNG backed by `xoshiro256++`.
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
    spare_normal: Option<f64>,
}

impl Rng {
    /// Create a seeded generator from any `u64` seed (e.g. `42`).
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next_sm = || -> u64 {
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let state = [next_sm(), next_sm(), next_sm(), next_sm()];
        Self {
            state,
            spare_normal: None,
        }
    }

    /// Generate the next raw `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[0].wrapping_add(self.state[3]))
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform `f64` in `[0.0, 1.0)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Use top 53 bits for IEEE-754 double precision uniform variate.
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Uniform integer in `[0, upper)`.
    #[inline]
    pub fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        // Lemire's fast unbiased range reduction using 128-bit multiplication.
        let upper64 = upper as u64;
        loop {
            let prod = (self.next_u64() as u128) * (upper64 as u128);
            let low = prod as u64;
            if low >= upper64.wrapping_neg() % upper64 {
                return (prod >> 64) as usize;
            }
        }
    }

    /// Standard normal `N(0, 1)` variate via Box-Muller transform.
    pub fn next_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }
        let mut u1 = self.next_f64();
        while u1 <= 1e-300 {
            u1 = self.next_f64();
        }
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Sample `k` distinct indices in `0..n` without replacement.
    ///
    /// Uses Floyd's O(k) algorithm when `k * 4 < n` (ideal for subsampling
    /// 500k rows out of 6M) and partial Fisher-Yates otherwise.
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        let k = k.min(n);
        if k == 0 {
            return Vec::new();
        }
        if k == n {
            let mut idx: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = self.gen_range(i + 1);
                idx.swap(i, j);
            }
            return idx;
        }
        if k.saturating_mul(4) < n {
            // Floyd's algorithm preserves randomness in O(k) memory & time.
            let mut chosen = HashSet::with_capacity(k);
            let mut result = Vec::with_capacity(k);
            for j in (n - k)..n {
                let t = self.gen_range(j + 1);
                let picked = if chosen.contains(&t) { j } else { t };
                chosen.insert(picked);
                result.push(picked);
            }
            // Shuffle output so order isn't biased by Floyd insertion order.
            for i in (1..result.len()).rev() {
                let j = self.gen_range(i + 1);
                result.swap(i, j);
            }
            result
        } else {
            let mut pool: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = i + self.gen_range(n - i);
                pool.swap(i, j);
            }
            pool.truncate(k);
            pool
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_indices_distinct_and_in_bounds() {
        let mut rng = Rng::seed_from_u64(42);
        let small = rng.sample_indices(10_000, 100);
        assert_eq!(small.len(), 100);
        let set: HashSet<usize> = small.iter().copied().collect();
        assert_eq!(set.len(), 100);
        assert!(small.iter().all(|&x| x < 10_000));

        let large = rng.sample_indices(50, 40);
        assert_eq!(large.len(), 40);
        let set2: HashSet<usize> = large.iter().copied().collect();
        assert_eq!(set2.len(), 40);
    }

    #[test]
    fn normal_distribution_moments() {
        let mut rng = Rng::seed_from_u64(123);
        let n = 20_000;
        let vals: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let mean = vals.iter().sum::<f64>() / n as f64;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.03);
        assert!((var - 1.0).abs() < 0.05);
    }
}
