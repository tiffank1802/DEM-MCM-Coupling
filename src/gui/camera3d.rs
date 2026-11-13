//! Pure-Rust 3-D orbital camera and perspective projection mathematics.
//!
//! Designed so the exact same projection math powers both `egui::Painter`
//! (`gui-egui/src/viewport3d.rs`) and any headless/web preview renderer.

use crate::io::json::JsonValue;

/// Standard orthogonal / isometric camera viewpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraPreset {
    Iso,
    FrontXY,
    TopXZ,
    SideYZ,
}

/// 3-D orbital perspective camera looking at `target`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera3D {
    pub yaw: f64,
    pub pitch: f64,
    pub distance: f64,
    pub target: [f64; 3],
    pub fov_deg: f64,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            yaw: 0.65,
            pitch: 0.42,
            distance: 0.36,
            target: [0.0, 0.0, 0.0],
            fov_deg: 42.0,
        }
    }
}

/// Result of projecting a 3-D world point into 2-D viewport pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedPoint {
    pub x: f64,
    pub y: f64,
    /// Camera-space positive depth (used for painter's depth sorting).
    pub depth: f64,
    /// Projected scale factor `focal / depth` for sizing spheres/markers.
    pub scale: f64,
}

impl Camera3D {
    pub fn fit_bounds(bounds: [f64; 6]) -> Self {
        let cx = 0.5 * (bounds[0] + bounds[1]);
        let cy = 0.5 * (bounds[2] + bounds[3]);
        let cz = 0.5 * (bounds[4] + bounds[5]);
        let dx = bounds[1] - bounds[0];
        let dy = bounds[3] - bounds[2];
        let dz = bounds[5] - bounds[4];
        let diag = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-4);
        Self {
            yaw: 0.68,
            pitch: 0.44,
            distance: diag * 1.45,
            target: [cx, cy, cz],
            fov_deg: 42.0,
        }
    }

    pub fn set_preset(&mut self, preset: CameraPreset) {
        match preset {
            CameraPreset::Iso => {
                self.yaw = 0.68;
                self.pitch = 0.44;
            }
            CameraPreset::FrontXY => {
                self.yaw = 0.0;
                self.pitch = 0.0;
            }
            CameraPreset::TopXZ => {
                self.yaw = 0.0;
                self.pitch = 1.55;
            }
            CameraPreset::SideYZ => {
                self.yaw = std::f64::consts::FRAC_PI_2;
                self.pitch = 0.0;
            }
        }
    }

    pub fn orbit(&mut self, delta_yaw: f64, delta_pitch: f64) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-1.54, 1.54);
    }

    pub fn zoom(&mut self, factor: f64) {
        self.distance = (self.distance * factor).clamp(0.02, 50.0);
    }

    /// Orthonormal `(right, up, forward)` basis vectors in world coordinates.
    pub fn basis(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();

        // Eye relative to target: spherical coords
        let eye_dir = [cp * sy, sp, cp * cy];
        // Forward goes from eye toward target
        let forward = [-eye_dir[0], -eye_dir[1], -eye_dir[2]];
        // World up (0, 1, 0) -> right = normalize(forward x world_up)
        let world_up = [0.0, 1.0, 0.0];
        let rx = forward[1] * world_up[2] - forward[2] * world_up[1];
        let ry = forward[2] * world_up[0] - forward[0] * world_up[2];
        let rz = forward[0] * world_up[1] - forward[1] * world_up[0];
        let r_len = (rx * rx + ry * ry + rz * rz).sqrt().max(1e-12);
        let right = [rx / r_len, ry / r_len, rz / r_len];
        // True camera up = right x forward
        let up = [
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0],
        ];
        (right, up, forward)
    }

    pub fn pan(&mut self, dx_pixels: f64, dy_pixels: f64, viewport_h: f64) {
        let scale =
            2.0 * self.distance * (self.fov_deg.to_radians() * 0.5).tan() / viewport_h.max(1.0);
        let (right, up, _) = self.basis();
        for d in 0..3 {
            self.target[d] -= dx_pixels * scale * right[d] - dy_pixels * scale * up[d];
        }
    }

    /// Project a 3-D point into 2-D pixel coordinates inside `(width, height)`.
    pub fn project(&self, p: [f64; 3], width: f64, height: f64) -> Option<ProjectedPoint> {
        let (right, up, forward) = self.basis();
        let eye = [
            self.target[0] - forward[0] * self.distance,
            self.target[1] - forward[1] * self.distance,
            self.target[2] - forward[2] * self.distance,
        ];
        let rel = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
        let cx = rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2];
        let cy = rel[0] * up[0] + rel[1] * up[1] + rel[2] * up[2];
        let depth = rel[0] * forward[0] + rel[1] * forward[1] + rel[2] * forward[2];

        if depth <= 1e-4 {
            return None;
        }
        let focal = (0.5 * height) / (self.fov_deg.to_radians() * 0.5).tan();
        let scale = focal / depth;
        Some(ProjectedPoint {
            x: 0.5 * width + cx * scale,
            y: 0.5 * height - cy * scale,
            depth,
            scale,
        })
    }

    pub fn to_json(&self) -> JsonValue {
        let mut o = JsonValue::object();
        o.insert("yaw", JsonValue::Number(self.yaw));
        o.insert("pitch", JsonValue::Number(self.pitch));
        o.insert("distance", JsonValue::Number(self.distance));
        o.insert(
            "target",
            JsonValue::Array(self.target.iter().map(|&v| JsonValue::Number(v)).collect()),
        );
        o.insert("fov_deg", JsonValue::Number(self.fov_deg));
        o
    }
}
