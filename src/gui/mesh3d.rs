//! Pure-Rust 3-D mesh geometry builder for all 14 partitioning approaches.
//!
//! Converts any fitted [`Partitioner`] into an explicit list of 3-D cell
//! wireframes, convex faces, centroids, split planes, and drum bounds ready to
//! render either directly in `egui::Painter` (`gui-egui`) or over the live Web
//! preview (`dem-mcm-gui`).

use std::collections::HashSet;
use std::f64::consts::PI;

use crate::error::Result;
use crate::io::json::JsonValue;
use crate::partitioners::{ObliqueCellGeometry, Partitioner, PartitionerGeometry};

/// Wireframe + polygon-face geometry for a single Markov state cell in 3-D.
#[derive(Debug, Clone)]
pub struct MeshCell3D {
    pub id: usize,
    pub zone_index: usize,
    pub centroid: [f64; 3],
    pub bounds: [f64; 6],
    pub wire_segments: Vec<([f64; 3], [f64; 3])>,
    pub faces: Vec<Vec<[f64; 3]>>,
}

/// Optional internal separating plane (e.g. `Adaptive` or `MultiZone` y-split).
#[derive(Debug, Clone)]
pub struct SplitPlane3D {
    pub label: String,
    pub quad: [[f64; 3]; 4],
}

/// Bounding cylinder of the rotating drum inferred from particle coordinates.
#[derive(Debug, Clone)]
pub struct DrumCylinder3D {
    pub radius: f64,
    pub z_min: f64,
    pub z_max: f64,
}

/// Complete 3-D visualization representation of a spatial mesh.
#[derive(Debug, Clone)]
pub struct Mesh3DData {
    pub method: String,
    pub label: String,
    pub n_cells: usize,
    pub domain_bounds: [f64; 6],
    pub drum: DrumCylinder3D,
    pub cells: Vec<MeshCell3D>,
    pub split_planes: Vec<SplitPlane3D>,
}

/// Extract domain bounds `[xmin, xmax, ymin, ymax, zmin, zmax]` and rotating
/// drum cylinder parameters from 3-D particle coordinates.
pub fn infer_domain(coords: &[[f64; 3]]) -> ([f64; 6], DrumCylinder3D) {
    if coords.is_empty() {
        return (
            [-0.1, 0.1, -0.1, 0.1, -0.025, 0.025],
            DrumCylinder3D {
                radius: 0.1,
                z_min: -0.025,
                z_max: 0.025,
            },
        );
    }
    let mut mins = [f64::INFINITY; 3];
    let mut maxs = [f64::NEG_INFINITY; 3];
    let mut r_max = 0.0f64;
    for c in coords {
        for d in 0..3 {
            mins[d] = mins[d].min(c[d]);
            maxs[d] = maxs[d].max(c[d]);
        }
        r_max = r_max.max((c[0] * c[0] + c[1] * c[1]).sqrt());
    }
    (
        [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]],
        DrumCylinder3D {
            radius: r_max.max(1e-4),
            z_min: mins[2],
            z_max: maxs[2],
        },
    )
}

type WireSegment = ([f64; 3], [f64; 3]);
type PolygonFace = Vec<[f64; 3]>;
type BoxGeometry = ([f64; 3], Vec<WireSegment>, Vec<PolygonFace>);
type CellVolumeGeometry = ([f64; 3], [f64; 6], Vec<WireSegment>, Vec<PolygonFace>);

struct SectorSpec {
    xc: f64,
    yc: f64,
    r0: f64,
    r1: f64,
    t0: f64,
    t1: f64,
    z0: f64,
    z1: f64,
}

fn box_geometry(b: [f64; 6]) -> BoxGeometry {
    let [x0, x1, y0, y1, z0, z1] = b;
    let c = [
        [x0, y0, z0], // 0
        [x1, y0, z0], // 1
        [x1, y1, z0], // 2
        [x0, y1, z0], // 3
        [x0, y0, z1], // 4
        [x1, y0, z1], // 5
        [x1, y1, z1], // 6
        [x0, y1, z1], // 7
    ];
    let edges = vec![
        (c[0], c[1]),
        (c[1], c[2]),
        (c[2], c[3]),
        (c[3], c[0]),
        (c[4], c[5]),
        (c[5], c[6]),
        (c[6], c[7]),
        (c[7], c[4]),
        (c[0], c[4]),
        (c[1], c[5]),
        (c[2], c[6]),
        (c[3], c[7]),
    ];
    let faces = vec![
        vec![c[0], c[1], c[2], c[3]],
        vec![c[4], c[5], c[6], c[7]],
        vec![c[0], c[1], c[5], c[4]],
        vec![c[2], c[3], c[7], c[6]],
        vec![c[1], c[2], c[6], c[5]],
        vec![c[3], c[0], c[4], c[7]],
    ];
    let centroid = [0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)];
    (centroid, edges, faces)
}

fn sector_geometry(spec: SectorSpec) -> CellVolumeGeometry {
    let SectorSpec {
        xc,
        yc,
        r0,
        r1,
        t0,
        t1,
        z0,
        z1,
    } = spec;
    let n_arc = (((t1 - t0).abs() / (PI / 18.0)).ceil() as usize).clamp(3, 16);
    let mut edges = Vec::with_capacity(n_arc * 4 + 8);
    let mut top_poly = Vec::with_capacity(n_arc * 2 + 2);
    let mut bot_poly = Vec::with_capacity(n_arc * 2 + 2);

    let pt = |r: f64, th: f64, z: f64| [xc + r * th.cos(), yc + r * th.sin(), z];

    for i in 0..n_arc {
        let a = t0 + (t1 - t0) * (i as f64) / (n_arc as f64);
        let b = t0 + (t1 - t0) * ((i + 1) as f64) / (n_arc as f64);
        edges.push((pt(r1, a, z0), pt(r1, b, z0)));
        edges.push((pt(r1, a, z1), pt(r1, b, z1)));
        if r0 > 1e-9 {
            edges.push((pt(r0, a, z0), pt(r0, b, z0)));
            edges.push((pt(r0, a, z1), pt(r0, b, z1)));
        }
    }
    edges.push((pt(r0, t0, z0), pt(r1, t0, z0)));
    edges.push((pt(r0, t1, z0), pt(r1, t1, z0)));
    edges.push((pt(r0, t0, z1), pt(r1, t0, z1)));
    edges.push((pt(r0, t1, z1), pt(r1, t1, z1)));
    edges.push((pt(r1, t0, z0), pt(r1, t0, z1)));
    edges.push((pt(r1, t1, z0), pt(r1, t1, z1)));
    if r0 > 1e-9 {
        edges.push((pt(r0, t0, z0), pt(r0, t0, z1)));
        edges.push((pt(r0, t1, z0), pt(r0, t1, z1)));
    }

    for i in 0..=n_arc {
        let th = t0 + (t1 - t0) * (i as f64) / (n_arc as f64);
        top_poly.push(pt(r1, th, z1));
        bot_poly.push(pt(r1, th, z0));
    }
    if r0 > 1e-9 {
        for i in (0..=n_arc).rev() {
            let th = t0 + (t1 - t0) * (i as f64) / (n_arc as f64);
            top_poly.push(pt(r0, th, z1));
            bot_poly.push(pt(r0, th, z0));
        }
    } else {
        top_poly.push([xc, yc, z1]);
        bot_poly.push([xc, yc, z0]);
    }

    let r_mid = 0.5 * (r0 + r1);
    let t_mid = 0.5 * (t0 + t1);
    let centroid = pt(r_mid, t_mid, 0.5 * (z0 + z1));

    let mut bounds = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        z0,
        z1,
    ];
    for p in &top_poly {
        bounds[0] = bounds[0].min(p[0]);
        bounds[1] = bounds[1].max(p[0]);
        bounds[2] = bounds[2].min(p[1]);
        bounds[3] = bounds[3].max(p[1]);
    }
    let faces = vec![
        bot_poly,
        top_poly,
        vec![
            pt(r0, t0, z0),
            pt(r1, t0, z0),
            pt(r1, t0, z1),
            pt(r0, t0, z1),
        ],
        vec![
            pt(r0, t1, z0),
            pt(r1, t1, z0),
            pt(r1, t1, z1),
            pt(r0, t1, z1),
        ],
    ];
    (centroid, bounds, edges, faces)
}

/// Exact 3-D convex polyhedron representation clipped by linear half-spaces
/// `n · x <= offset`. Used both for Voronoi 3-D cells and oblique Octrees!
#[derive(Debug, Clone)]
struct Polyhedron3D {
    faces: Vec<Vec<[f64; 3]>>,
}

impl Polyhedron3D {
    fn from_box(b: [f64; 6]) -> Self {
        let (_, _, faces) = box_geometry(b);
        Self { faces }
    }

    fn clip_halfspace(&mut self, normal: [f64; 3], offset: f64, side_le: bool) {
        let sign = if side_le { 1.0 } else { -1.0 };
        let n = [normal[0] * sign, normal[1] * sign, normal[2] * sign];
        let off = offset * sign;
        let eps = 1e-10;
        let eval = |p: [f64; 3]| p[0] * n[0] + p[1] * n[1] + p[2] * n[2] - off;

        let mut new_faces: Vec<Vec<[f64; 3]>> = Vec::with_capacity(self.faces.len() + 1);
        let mut cap_points: Vec<[f64; 3]> = Vec::new();

        for poly in &self.faces {
            if poly.len() < 3 {
                continue;
            }
            let mut out = Vec::with_capacity(poly.len() + 2);
            let m = poly.len();
            for i in 0..m {
                let a = poly[i];
                let b = poly[(i + 1) % m];
                let da = eval(a);
                let db = eval(b);
                let in_a = da <= eps;
                let in_b = db <= eps;
                if in_a && in_b {
                    out.push(b);
                } else if in_a && !in_b {
                    let t = (da / (da - db)).clamp(0.0, 1.0);
                    let inter = [
                        a[0] + t * (b[0] - a[0]),
                        a[1] + t * (b[1] - a[1]),
                        a[2] + t * (b[2] - a[2]),
                    ];
                    out.push(inter);
                    cap_points.push(inter);
                } else if !in_a && in_b {
                    let t = (da / (da - db)).clamp(0.0, 1.0);
                    let inter = [
                        a[0] + t * (b[0] - a[0]),
                        a[1] + t * (b[1] - a[1]),
                        a[2] + t * (b[2] - a[2]),
                    ];
                    out.push(inter);
                    cap_points.push(inter);
                    out.push(b);
                }
            }
            // Deduplicate near-coincident vertices
            let dedup = deduplicate_ring(&out, 1e-8);
            if dedup.len() >= 3 {
                new_faces.push(dedup);
            }
        }

        // Order cap polygon counter-clockwise around normal `n`
        let cap = deduplicate_points(&cap_points, 1e-8);
        if cap.len() >= 3 {
            let mut center = [0.0; 3];
            for p in &cap {
                center[0] += p[0];
                center[1] += p[1];
                center[2] += p[2];
            }
            let inv = 1.0 / cap.len() as f64;
            center[0] *= inv;
            center[1] *= inv;
            center[2] *= inv;

            let n_len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
            let n_unit = [n[0] / n_len, n[1] / n_len, n[2] / n_len];
            let arb = if n_unit[0].abs() < 0.8 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            // u = normalize(n x arb), v = n x u
            let ux = n_unit[1] * arb[2] - n_unit[2] * arb[1];
            let uy = n_unit[2] * arb[0] - n_unit[0] * arb[2];
            let uz = n_unit[0] * arb[1] - n_unit[1] * arb[0];
            let ul = (ux * ux + uy * uy + uz * uz).sqrt().max(1e-12);
            let u = [ux / ul, uy / ul, uz / ul];
            let v = [
                n_unit[1] * u[2] - n_unit[2] * u[1],
                n_unit[2] * u[0] - n_unit[0] * u[2],
                n_unit[0] * u[1] - n_unit[1] * u[0],
            ];
            let mut sorted = cap;
            sorted.sort_by(|a, b| {
                let ra = [a[0] - center[0], a[1] - center[1], a[2] - center[2]];
                let rb = [b[0] - center[0], b[1] - center[1], b[2] - center[2]];
                let ang_a = (ra[0] * v[0] + ra[1] * v[1] + ra[2] * v[2])
                    .atan2(ra[0] * u[0] + ra[1] * u[1] + ra[2] * u[2]);
                let ang_b = (rb[0] * v[0] + rb[1] * v[1] + rb[2] * v[2])
                    .atan2(rb[0] * u[0] + rb[1] * u[1] + rb[2] * u[2]);
                ang_a.total_cmp(&ang_b)
            });
            new_faces.push(sorted);
        }

        self.faces = new_faces;
    }

    fn into_wire_and_faces(
        self,
        fallback_bounds: [f64; 6],
        fallback_centroid: [f64; 3],
    ) -> CellVolumeGeometry {
        if self.faces.is_empty() {
            let (c, e, f) = box_geometry(fallback_bounds);
            return (c, fallback_bounds, e, f);
        }
        let mut seen: HashSet<([i64; 3], [i64; 3])> = HashSet::new();
        let mut edges = Vec::new();
        let mut bounds = [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut pts = Vec::new();
        let quant = |p: [f64; 3]| -> [i64; 3] {
            [
                (p[0] * 1e6).round() as i64,
                (p[1] * 1e6).round() as i64,
                (p[2] * 1e6).round() as i64,
            ]
        };

        for poly in &self.faces {
            for (i, &a) in poly.iter().enumerate() {
                let b = poly[(i + 1) % poly.len()];
                for d in 0..3 {
                    bounds[2 * d] = bounds[2 * d].min(a[d]);
                    bounds[2 * d + 1] = bounds[2 * d + 1].max(a[d]);
                }
                pts.push(a);
                let qa = quant(a);
                let qb = quant(b);
                if qa == qb {
                    continue;
                }
                let key = if qa < qb { (qa, qb) } else { (qb, qa) };
                if seen.insert(key) {
                    edges.push((a, b));
                }
            }
        }
        let dedup_pts = deduplicate_points(&pts, 1e-7);
        let centroid = if dedup_pts.is_empty() {
            fallback_centroid
        } else {
            let inv = 1.0 / dedup_pts.len() as f64;
            let s = dedup_pts.iter().fold([0.0; 3], |acc, p| {
                [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]]
            });
            [s[0] * inv, s[1] * inv, s[2] * inv]
        };
        (centroid, bounds, edges, self.faces)
    }
}

fn deduplicate_ring(pts: &[[f64; 3]], tol: f64) -> Vec<[f64; 3]> {
    let t2 = tol * tol;
    let mut out: Vec<[f64; 3]> = Vec::with_capacity(pts.len());
    for &p in pts {
        if let Some(last) = out.last() {
            let d2 = (p[0] - last[0]).powi(2) + (p[1] - last[1]).powi(2) + (p[2] - last[2]).powi(2);
            if d2 <= t2 {
                continue;
            }
        }
        out.push(p);
    }
    if out.len() >= 2 {
        let first = out[0];
        let last = *out.last().unwrap();
        let d2 = (first[0] - last[0]).powi(2)
            + (first[1] - last[1]).powi(2)
            + (first[2] - last[2]).powi(2);
        if d2 <= t2 {
            out.pop();
        }
    }
    out
}

fn deduplicate_points(pts: &[[f64; 3]], tol: f64) -> Vec<[f64; 3]> {
    let t2 = tol * tol;
    let mut out: Vec<[f64; 3]> = Vec::new();
    for &p in pts {
        if !out
            .iter()
            .any(|q| (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2) <= t2)
        {
            out.push(p);
        }
    }
    out
}

struct BuildMeshContext<'a> {
    coords: &'a [[f64; 3]],
    states: &'a [i64],
    n_cells: usize,
    domain_bounds: [f64; 6],
    zone_index: usize,
    id_offset: usize,
}

fn build_cells_from_geometry(
    geom: &PartitionerGeometry,
    ctx: &BuildMeshContext<'_>,
    cells_out: &mut Vec<MeshCell3D>,
    planes_out: &mut Vec<SplitPlane3D>,
) {
    let coords = ctx.coords;
    let states = ctx.states;
    let n_cells = ctx.n_cells;
    let domain_bounds = ctx.domain_bounds;
    let zone_index = ctx.zone_index;
    let id_offset = ctx.id_offset;
    match geom {
        PartitionerGeometry::CartesianGrid { nx, ny, nz, bounds } => {
            let [x0, x1, y0, y1, z0, z1] = *bounds;
            let dx = (x1 - x0) / (*nx).max(1) as f64;
            let dy = (y1 - y0) / (*ny).max(1) as f64;
            let dz = (z1 - z0) / (*nz).max(1) as f64;
            for iz in 0..*nz {
                for iy in 0..*ny {
                    for ix in 0..*nx {
                        let id = id_offset + ix + iy * nx + iz * nx * ny;
                        let b = [
                            x0 + ix as f64 * dx,
                            x0 + (ix + 1) as f64 * dx,
                            y0 + iy as f64 * dy,
                            y0 + (iy + 1) as f64 * dy,
                            z0 + iz as f64 * dz,
                            z0 + (iz + 1) as f64 * dz,
                        ];
                        let (centroid, wire_segments, faces) = box_geometry(b);
                        cells_out.push(MeshCell3D {
                            id,
                            zone_index,
                            centroid,
                            bounds: b,
                            wire_segments,
                            faces,
                        });
                    }
                }
            }
        }
        PartitionerGeometry::TensorGrid {
            x_edges,
            y_edges,
            z_edges,
        } => {
            let nx = x_edges.len().saturating_sub(1);
            let ny = y_edges.len().saturating_sub(1);
            let nz = z_edges.len().saturating_sub(1);
            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let id = id_offset + ix + iy * nx + iz * nx * ny;
                        let b = [
                            x_edges[ix],
                            x_edges[ix + 1],
                            y_edges[iy],
                            y_edges[iy + 1],
                            z_edges[iz],
                            z_edges[iz + 1],
                        ];
                        let (centroid, wire_segments, faces) = box_geometry(b);
                        cells_out.push(MeshCell3D {
                            id,
                            zone_index,
                            centroid,
                            bounds: b,
                            wire_segments,
                            faces,
                        });
                    }
                }
            }
        }
        PartitionerGeometry::CylindricalGrid {
            x_center,
            y_center,
            r_edges,
            theta_edges,
            z_edges,
        } => {
            let nr = r_edges.len().saturating_sub(1);
            let nt = theta_edges.len().saturating_sub(1);
            let nz = z_edges.len().saturating_sub(1);
            for iz in 0..nz {
                for it in 0..nt {
                    for ir in 0..nr {
                        let id = id_offset + ir + it * nr + iz * nr * nt;
                        let (centroid, bounds, wire_segments, faces) =
                            sector_geometry(SectorSpec {
                                xc: *x_center,
                                yc: *y_center,
                                r0: r_edges[ir],
                                r1: r_edges[ir + 1],
                                t0: theta_edges[it],
                                t1: theta_edges[it + 1],
                                z0: z_edges[iz],
                                z1: z_edges[iz + 1],
                            });
                        cells_out.push(MeshCell3D {
                            id,
                            zone_index,
                            centroid,
                            bounds,
                            wire_segments,
                            faces,
                        });
                    }
                }
            }
        }
        PartitionerGeometry::AxisBoxes(boxes) => {
            for (i, &b) in boxes.iter().enumerate() {
                let (centroid, wire_segments, faces) = box_geometry(b);
                cells_out.push(MeshCell3D {
                    id: id_offset + i,
                    zone_index,
                    centroid,
                    bounds: b,
                    wire_segments,
                    faces,
                });
            }
        }
        PartitionerGeometry::ObliquePolyhedra {
            domain_bounds: ob_bounds,
            cells: ob_cells,
        } => {
            for (
                i,
                ObliqueCellGeometry {
                    bounds,
                    centroid,
                    halfspaces,
                },
            ) in ob_cells.iter().enumerate()
            {
                let mut poly = Polyhedron3D::from_box(*ob_bounds);
                for &(normal, offset, side_le) in halfspaces {
                    poly.clip_halfspace(normal, offset, side_le);
                }
                let (c, b, wire_segments, faces) = poly.into_wire_and_faces(*bounds, *centroid);
                cells_out.push(MeshCell3D {
                    id: id_offset + i,
                    zone_index,
                    centroid: c,
                    bounds: b,
                    wire_segments,
                    faces,
                });
            }
        }
        PartitionerGeometry::VoronoiSeeds(seeds) => {
            let k = seeds.len();
            if k == 1 {
                let (centroid, wire_segments, faces) = box_geometry(domain_bounds);
                cells_out.push(MeshCell3D {
                    id: id_offset,
                    zone_index,
                    centroid,
                    bounds: domain_bounds,
                    wire_segments,
                    faces,
                });
                return;
            }
            for (i, &ci) in seeds.iter().enumerate() {
                // Sort neighbors by distance and clip against closest neighbors
                // (or all neighbors if k <= 80) to build exact 3-D Voronoi polyhedron!
                let mut nbrs: Vec<(f64, usize)> = (0..k)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let d2 = (seeds[j][0] - ci[0]).powi(2)
                            + (seeds[j][1] - ci[1]).powi(2)
                            + (seeds[j][2] - ci[2]).powi(2);
                        (d2, j)
                    })
                    .collect();
                nbrs.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut poly = Polyhedron3D::from_box(domain_bounds);
                for (_, j) in nbrs.into_iter().take(48) {
                    let cj = seeds[j];
                    let n = [cj[0] - ci[0], cj[1] - ci[1], cj[2] - ci[2]];
                    let mid = [
                        0.5 * (ci[0] + cj[0]),
                        0.5 * (ci[1] + cj[1]),
                        0.5 * (ci[2] + cj[2]),
                    ];
                    let off = n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2];
                    poly.clip_halfspace(n, off, true);
                }
                let (_poly_cen, b, wire_segments, faces) = poly.into_wire_and_faces(
                    [
                        ci[0] - 0.01,
                        ci[0] + 0.01,
                        ci[1] - 0.01,
                        ci[1] + 0.01,
                        ci[2] - 0.01,
                        ci[2] + 0.01,
                    ],
                    ci,
                );
                cells_out.push(MeshCell3D {
                    id: id_offset + i,
                    zone_index,
                    centroid: ci,
                    bounds: b,
                    wire_segments,
                    faces,
                });
            }
        }
        PartitionerGeometry::AdaptiveZones {
            y_split,
            y_min: _,
            y_max: _,
            n_bottom,
            bottom,
            top,
        } => {
            let [x0, x1, _, _, z0, z1] = domain_bounds;
            planes_out.push(SplitPlane3D {
                label: format!("y_split = {y_split:.4}"),
                quad: [
                    [x0, *y_split, z0],
                    [x1, *y_split, z0],
                    [x1, *y_split, z1],
                    [x0, *y_split, z1],
                ],
            });
            let mut bot_bounds = domain_bounds;
            bot_bounds[3] = *y_split;
            let bot_ctx = BuildMeshContext {
                coords,
                states,
                n_cells: *n_bottom,
                domain_bounds: bot_bounds,
                zone_index: 0,
                id_offset,
            };
            build_cells_from_geometry(bottom, &bot_ctx, cells_out, planes_out);
            if let Some(tp) = top {
                let mut top_bounds = domain_bounds;
                top_bounds[2] = *y_split;
                let top_n = n_cells.saturating_sub(*n_bottom).max(1);
                let top_ctx = BuildMeshContext {
                    coords,
                    states,
                    n_cells: top_n,
                    domain_bounds: top_bounds,
                    zone_index: 1,
                    id_offset: id_offset + *n_bottom,
                };
                build_cells_from_geometry(tp, &top_ctx, cells_out, planes_out);
            }
        }
        PartitionerGeometry::MultiZone { zones } => {
            let [x0, x1, _, _, z0, z1] = domain_bounds;
            for (zi, (ymin, ymax, off, sub_g)) in zones.iter().enumerate() {
                if zi > 0 {
                    planes_out.push(SplitPlane3D {
                        label: format!("zone {zi} y = {ymin:.4}"),
                        quad: [
                            [x0, *ymin, z0],
                            [x1, *ymin, z0],
                            [x1, *ymin, z1],
                            [x0, *ymin, z1],
                        ],
                    });
                }
                let mut zb = domain_bounds;
                zb[2] = *ymin;
                zb[3] = *ymax;
                let next_off = zones.get(zi + 1).map_or(n_cells, |z| z.2);
                let sub_n = next_off.saturating_sub(*off).max(1);
                let z_ctx = BuildMeshContext {
                    coords,
                    states,
                    n_cells: sub_n,
                    domain_bounds: zb,
                    zone_index: zi,
                    id_offset: id_offset + *off,
                };
                build_cells_from_geometry(sub_g, &z_ctx, cells_out, planes_out);
            }
        }
        PartitionerGeometry::SingleDomain => {
            let (centroid, wire_segments, faces) = box_geometry(domain_bounds);
            cells_out.push(MeshCell3D {
                id: id_offset,
                zone_index,
                centroid,
                bounds: domain_bounds,
                wire_segments,
                faces,
            });
        }
        PartitionerGeometry::PointClusters => {
            // Reconstruct centroids & tight per-cluster AABBs from particle assignments
            let k = n_cells.max(1);
            let mut mins = vec![[f64::INFINITY; 3]; k];
            let mut maxs = vec![[f64::NEG_INFINITY; 3]; k];
            let mut sums = vec![[0.0f64; 3]; k];
            let mut cnts = vec![0usize; k];
            for (c, &s) in coords.iter().zip(states.iter()) {
                if s >= 0 && (s as usize) < k {
                    let cid = s as usize;
                    cnts[cid] += 1;
                    for (d, &cv) in c.iter().enumerate() {
                        mins[cid][d] = mins[cid][d].min(cv);
                        maxs[cid][d] = maxs[cid][d].max(cv);
                        sums[cid][d] += cv;
                    }
                }
            }
            for cid in 0..k {
                if cnts[cid] == 0 {
                    continue;
                }
                let inv = 1.0 / cnts[cid] as f64;
                let cen = [sums[cid][0] * inv, sums[cid][1] * inv, sums[cid][2] * inv];
                let b = [
                    mins[cid][0],
                    maxs[cid][0].max(mins[cid][0] + 1e-4),
                    mins[cid][1],
                    maxs[cid][1].max(mins[cid][1] + 1e-4),
                    mins[cid][2],
                    maxs[cid][2].max(mins[cid][2] + 1e-4),
                ];
                let (_c, wire_segments, faces) = box_geometry(b);
                cells_out.push(MeshCell3D {
                    id: id_offset + cid,
                    zone_index,
                    centroid: cen,
                    bounds: b,
                    wire_segments,
                    faces,
                });
            }
        }
    }
}

/// Extract a complete [`Mesh3DData`] descriptor from a fitted [`Partitioner`] and
/// sample DEM particle positions/states.
pub fn extract_mesh_3d(
    partitioner: &dyn Partitioner,
    coords: &[[f64; 3]],
    states: &[i64],
) -> Result<Mesh3DData> {
    let (domain_bounds, drum) = infer_domain(coords);
    let n_cells = partitioner.n_cells();
    let geom = partitioner.geometry_hint();
    let mut cells = Vec::new();
    let mut split_planes = Vec::new();
    let ctx = BuildMeshContext {
        coords,
        states,
        n_cells,
        domain_bounds,
        zone_index: 0,
        id_offset: 0,
    };
    build_cells_from_geometry(&geom, &ctx, &mut cells, &mut split_planes);
    Ok(Mesh3DData {
        method: partitioner.method().as_str().to_string(),
        label: partitioner.label(),
        n_cells,
        domain_bounds,
        drum,
        cells,
        split_planes,
    })
}

impl Mesh3DData {
    pub fn to_json(&self) -> JsonValue {
        let p3 = |v: &[f64; 3]| {
            JsonValue::Array(vec![
                JsonValue::Number(v[0]),
                JsonValue::Number(v[1]),
                JsonValue::Number(v[2]),
            ])
        };
        let mut root = JsonValue::object();
        root.insert("method", JsonValue::String(self.method.clone()));
        root.insert("label", JsonValue::String(self.label.clone()));
        root.insert("n_cells", JsonValue::Number(self.n_cells as f64));
        root.insert(
            "domain_bounds",
            JsonValue::Array(
                self.domain_bounds
                    .iter()
                    .map(|&v| JsonValue::Number(v))
                    .collect(),
            ),
        );

        let mut drum = JsonValue::object();
        drum.insert("radius", JsonValue::Number(self.drum.radius));
        drum.insert("z_min", JsonValue::Number(self.drum.z_min));
        drum.insert("z_max", JsonValue::Number(self.drum.z_max));
        root.insert("drum", drum);

        let cells_json: Vec<JsonValue> = self
            .cells
            .iter()
            .map(|c| {
                let mut o = JsonValue::object();
                o.insert("id", JsonValue::Number(c.id as f64));
                o.insert("zone", JsonValue::Number(c.zone_index as f64));
                o.insert("centroid", p3(&c.centroid));
                o.insert(
                    "bounds",
                    JsonValue::Array(c.bounds.iter().map(|&v| JsonValue::Number(v)).collect()),
                );
                o.insert(
                    "edges",
                    JsonValue::Array(
                        c.wire_segments
                            .iter()
                            .map(|(a, b)| JsonValue::Array(vec![p3(a), p3(b)]))
                            .collect(),
                    ),
                );
                o.insert(
                    "faces",
                    JsonValue::Array(
                        c.faces
                            .iter()
                            .map(|poly| JsonValue::Array(poly.iter().map(p3).collect()))
                            .collect(),
                    ),
                );
                o
            })
            .collect();
        root.insert("cells", JsonValue::Array(cells_json));

        let planes_json: Vec<JsonValue> = self
            .split_planes
            .iter()
            .map(|sp| {
                let mut o = JsonValue::object();
                o.insert("label", JsonValue::String(sp.label.clone()));
                o.insert("quad", JsonValue::Array(sp.quad.iter().map(p3).collect()));
                o
            })
            .collect();
        root.insert("split_planes", JsonValue::Array(planes_json));
        root
    }
}
