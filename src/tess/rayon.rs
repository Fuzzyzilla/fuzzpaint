use crate::brush;
use rayon::prelude::*;
pub struct RayonTessellator;

use super::{TessellatedStrokeInfo, TessellatedStrokeVertex, TessellationError};
use crate::{StrokeBrushSettings, StrokePoint};

impl RayonTessellator {
    fn do_stamp(point: &StrokePoint, brush: &StrokeBrushSettings) -> [TessellatedStrokeVertex; 6] {
        let size2 = (point.pressure * brush.size_mul) / 2.0;
        let color = [
            vulkano::half::f16::from_f32(brush.color_modulate[0]),
            vulkano::half::f16::from_f32(brush.color_modulate[1]),
            vulkano::half::f16::from_f32(brush.color_modulate[2]),
            vulkano::half::f16::from_f32(brush.color_modulate[3]),
        ];

        let rand = (point.pos[0] * 100.0).cos() * std::f32::consts::PI;
        let (sin, cos) = rand.sin_cos();
        let cos = cos * size2;
        let sin = sin * size2;

        let tl = TessellatedStrokeVertex {
            pos: [point.pos[0] - sin, point.pos[1] - cos],
            uv: [0.0, 0.0],
            color,
        };
        let tr = TessellatedStrokeVertex {
            pos: [point.pos[0] + cos, point.pos[1] - sin],
            uv: [1.0, 0.0],
            color,
        };
        let bl = TessellatedStrokeVertex {
            pos: [point.pos[0] - cos, point.pos[1] + sin],
            uv: [0.0, 1.0],
            color,
        };
        let br = TessellatedStrokeVertex {
            pos: [point.pos[0] + sin, point.pos[1] + cos],
            uv: [1.0, 1.0],
            color,
        };

        [tl, tr.clone(), bl.clone(), bl, tr, br]
    }
}
impl super::StrokeTessellator for RayonTessellator {
    fn tessellate(
        &self,
        strokes: &[crate::Stroke],
        infos_into: &mut [TessellatedStrokeInfo],
        vertices_into: crate::vk::Subbuffer<[TessellatedStrokeVertex]>,
        mut base_vertex: usize,
    ) -> std::result::Result<(), TessellationError> {
        if infos_into.len() < strokes.len() {
            return Err(TessellationError::InfosBufferTooSmall {
                needed_size: strokes.len(),
            });
        }
        let mut vertices_into = match vertices_into.write() {
            Ok(o) => o,
            Err(e) => return Err(TessellationError::BufferError(e)),
        };

        for (stroke, info) in strokes.iter().zip(infos_into.iter_mut()) {
            info.blend_constants = if stroke.brush.is_eraser {
                [0.0; 4]
            } else {
                [1.0; 4]
            };

            let brush = brush::todo_brush();

            // Perform tessellation!
            let points: Vec<TessellatedStrokeVertex> = match brush.style() {
                brush::BrushStyle::Stamped { spacing } => {
                    stroke
                        .points
                        .par_windows(2)
                        // Type: optimize for 6, but more is allowable (spills to heap).
                        // flat_map_iter, as each iter is small and thus wouldn't benifit from parallelization
                        .flat_map_iter(|win| -> smallvec::SmallVec<[TessellatedStrokeVertex; 6]> {
                            // Windows are always 2. no par_array_windows :V
                            let [a, b] = win else { unreachable!() };

                            // Sanity check - avoid division by zero and other weirdness.
                            if b.dist - a.dist <= 0.0 {
                                return Default::default();
                            }

                            // Offset of first stamp into this segment.
                            let offs = a.dist % spacing;
                            // Length of segment, after the first stamp (could be negative = no stamps)
                            let len = (b.dist - a.dist) - offs;

                            let mut current = 0.0;
                            let mut vertices = smallvec::SmallVec::new();
                            while current <= len {
                                // fractional [0, 1] distance between a and b.
                                let factor = (current + offs) / (b.dist - a.dist);

                                let point = a.lerp(b, factor);

                                vertices.extend(Self::do_stamp(&point, &stroke.brush).into_iter());

                                current += spacing;
                            }

                            vertices
                        })
                        .collect()
                }
                brush::BrushStyle::Rolled => unimplemented!(),
            };

            // Get some space in the buffer to write into
            let Some(slice) = vertices_into.get_mut(base_vertex..(base_vertex + points.len()))
            else {
                return Err(TessellationError::VertexBufferTooSmall {
                    needed_size: base_vertex + points.len(),
                });
            };

            // Populate infos
            info.first_vertex = base_vertex as u32;
            info.vertices = points.len() as u32;

            // Shift slice over for next stroke
            base_vertex += points.len();

            // Copy over.
            slice
                .iter_mut()
                .zip(points.into_iter())
                .for_each(|(into, from)| *into = from);
        }

        Ok(())
    }
    fn num_vertices_of(&self, stroke: &crate::Stroke) -> usize {
        // Somehow fetch the brush of this stroke
        let brush = brush::todo_brush();

        match brush.style() {
            brush::BrushStyle::Rolled => unimplemented!(),
            brush::BrushStyle::Stamped { spacing } => {
                if *spacing <= 0.0 {
                    // Sanity check.
                    0
                } else {
                    stroke
                        .points
                        .last()
                        .map(|last| {
                            let num_stamps = (last.dist / spacing).floor();
                            num_stamps as usize * 6
                        })
                        .unwrap_or(0usize)
                }
            }
        }
    }
}
