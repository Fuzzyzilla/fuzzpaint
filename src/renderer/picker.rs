use crate::picker::Picker;
use crate::vulkano_prelude::*;
use std::sync::Arc;

/// Given a `PickerInfo`, find the corners of the transfer region.
///
/// returns (`top_left`, `bottom_right`, viewCoord->UV matrix) or
/// `None` if the transform is malformed or wildly out of bounds.
fn calc_corners(
    info: super::requests::PickerInfo,
    stage_dimension: u32,
) -> Option<([i32; 2], [i32; 2], ultraviolet::Similarity2)> {
    use az::SaturatingAs;
    // If zoom < 100%, calculate the rectangle covered by `stage_dimension` square *input points*,
    //     (not viewport pixels!) centered around the picked location.
    // If zoom >= 100%, take `stage_dimension` square texels around the picked location.
    // This may result in a transfer region partially *outside* the viewport, however many windowing systems
    //  allow drag gestures to go beyond the bounds of the window!
    // A probably not worth it opt would be to only transfer the region covered by the total display area,
    //  but that'd only matter at like 1000% zoom and yea.

    let input_space_viewport = info
        .viewport
        .with_scale_factor(info.input_points_per_viewport_pixel)
        .calculate_transform()?;
    let scale = input_space_viewport.view_points_per_document_point();
    if
    /*scale >= 1.0*/
    true {
        use az::CheckedAs;
        // Take a `stage_dimension` square region of texels centered around input point.
        let input_point = input_space_viewport
            .unproject(cgmath::Point2 {
                x: info.sample_pos.x,
                y: info.sample_pos.y,
            })
            .ok()?;
        // As-cast ok, we just divided by two so the MAX will now certainly fit in i32.
        let half_dimension = (stage_dimension / 2) as i32;
        let input_point: [i32; 2] = [input_point.x.checked_as()?, input_point.y.checked_as()?];
        let top_left = [
            input_point[0].saturating_sub(half_dimension),
            input_point[1].saturating_sub(half_dimension),
        ];
        let bottom_right = [
            input_point[0].saturating_add(half_dimension),
            input_point[1].saturating_add(half_dimension),
        ];
        let xform = todo!();

        Some((top_left, bottom_right, xform))
    } else {
        // Take the largest possible AABB of texels such that every unique integer coordinate within the viewport
        // maps to a unique texel. DAJDOIAJfioadunsdfnk sdlf mighty complicated i don't wanna.
        todo!()
    }
}

mod stage;

// 256x256x8x2, ends up being a combined 1MiB of memory per stage.
const IMAGE_STAGE_DIMENSION: u32 = 256;

// Oncelock can't be initialized fallilbly, use this worse solution. x,3
static COLOR_STAGE: parking_lot::RwLock<Option<stage::ImageStage>> =
    parking_lot::const_rwlock(None);
static NE_ID_STAGE: parking_lot::RwLock<Option<stage::ImageStage>> =
    parking_lot::const_rwlock(None);

/// Checks the format is valid for interpreting texels as singular binary elements.
/// Returns a descriptive error if incorrect.
fn check_valid_binary_format(format: vk::Format) -> anyhow::Result<()> {
    // Are these sufficient checks? :O
    if format.block_extent() != [1; 3] {
        anyhow::bail!("must have non-block format")
    }
    if !format.planes().is_empty() {
        anyhow::bail!("must have non-planar format")
    }
    if format.compression().is_some() {
        anyhow::bail!("must not be compressed format")
    }

    Ok(())
}

/// Trivial picker from a Solid fill layer. :P
/// In the future when Fill layers become more.... more, this will do serious work,
/// such as calculating gradient values, patterns, ect. With a fill, we can generally
/// do the work on the host directly without needing to pester the device or use an image.
pub struct ConstantColorPicker {
    pub color: [f32; 4],
}
impl Picker for ConstantColorPicker {
    type Value = [f32; 4];
    fn pick(&self, _: ultraviolet::Vec2) -> Result<Self::Value, crate::picker::PickError> {
        Ok(self.color)
    }
}
/// Picker that acts on rendered image output, yielding linear, premultiplied RGBA.
/// This output could be a single layer, or a composite image.
///
/// Filtering is done "Nearest Neighbor"
pub struct RenderedColorPicker {
    offset: (u32, u32),
    extent: (u32, u32),
    inner_sampler: stage::OwnedSampler<[vulkano::half::f16; 4]>,
}
impl RenderedColorPicker {
    pub fn pull_from_image(
        ctx: &crate::render_device::RenderContext,
        image: Arc<vk::Image>,
        xform: (),
        viewport_rect: (),
    ) -> anyhow::Result<Self> {
        let mut stage_lock = COLOR_STAGE.write();
        // get or try insert:
        let stage = if let Some(stage) = stage_lock.as_mut() {
            stage
        } else {
            let new_stage = stage::ImageStage::new(
                // These are long-lived staging bufs and should probably use
                // their own bump allocator within a dedicated allocation rather than
                // muddying the global allocator.
                ctx.allocators().memory().clone(),
                crate::DOCUMENT_FORMAT,
                [IMAGE_STAGE_DIMENSION; 2],
            )?;
            stage_lock.insert(new_stage)
        };
        // Download and wait.
        stage
            .download(
                ctx,
                image,
                vk::ImageSubresourceLayers {
                    array_layers: 0..1,
                    aspects: vk::ImageAspects::COLOR,
                    mip_level: 0,
                },
                todo!(),
                todo!(),
            )?
            .detach()
            .wait(None)?;
        Ok(Self {
            offset: todo!(),
            extent: todo!(),
            inner_sampler: stage.owned_sampler()?,
        })
    }
}
impl Picker for RenderedColorPicker {
    type Value = [vulkano::half::f16; 4];
    fn pick(
        &self,
        viewport_coordinate: ultraviolet::Vec2,
    ) -> Result<Self::Value, crate::picker::PickError> {
        todo!()
    }
}

/// Picker from `NE_ID` image. These must be produced separately from the usual pipeline,
/// but yield a reference to the clicked stroke. This allows for precise color selection,
/// brush setting selection, ect.
///
/// Could be a single-layer image, or a composite.
pub struct StrokeIDPicker {}
impl crate::picker::Picker for StrokeIDPicker {
    // None if no element under cursor
    type Value = Option<crate::state::stroke_collection::ImmutableStrokeID>;
    fn pick(
        &self,
        viewport_coordinate: ultraviolet::Vec2,
    ) -> Result<Self::Value, crate::picker::PickError> {
        todo!()
    }
}
// /// Picker from NE_ID image. These must be produced separately from the usual pipeline,
// /// but yield a reference to the clicked layer.
//
// // this is just an idea, won't impl yet :3
// pub struct LeafIDPicker {}
// impl crate::picker::Picker for StrokeIDPicker {
//     type Value = crate::state::graph::LeafID;
//     fn pick(&self, viewport_coordinate: ultraviolet::Vec2) -> Option<Self::Value> {
//         None
//     }
// }

/// Option<NonZeroU64> in native endian. (`R64_uint` has very low support, this has very high support).
/// Vulkan spec guarantees that the device and host *must* have the same endian, so this can be a bitwise reinterpret to get the ID!
/// (alignment allowing, of course)
///
/// In the future, this could probably definitely be R32 or even R16,
/// and have a host-side mapping between texel value <-> `FuzzID` to decrease memory footprint and bandwidth.
/// For now, keep it simple!
const NE_ID_FORMAT: vk::Format = vk::Format::R32G32_UINT;

mod shaders {
    pub mod vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r#"#version 460
            
            layout(std430, push_constant) uniform Push {
                mat4 transform;
            };

            layout(location = 0) in vec2 pos;
            layout(location = 1) in float diameter;

            void main() {
                gl_Position = transform * vec4(pos, 0.0, 1.0);
            }
            "#
        }
    }
    pub mod frag {
        // there is literally nothing to do in this shader
        // I wonder if VRS is worth it or if it's already so trivial as to not matter :P
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r#"#version 460
            // Skips having this data travel through the whole pipeline since it's
            // unchanged anyway.
            layout(std430, push_constant) uniform Push {
                /// Option<NonZerou64>, in native endian.
                uvec2 ne_id;
            };

            layout(location = 0) out uvec2 out_ne_id;

            void main() {
                out_ne_id = ne_id;
            }
            "#
        }
    }
    pub mod geom {
        // Cryptic defines to configure the widelines shader.
        // See `src/shaders/widelines.geom` for docs.

        // Configure for no extra IO, no UV.
        vulkano_shaders::shader! {
            ty: "geometry",
            define: [
                ("WIDTH_LOCATION", "0"),
            ],
            path: "src/shaders/widelines.geom",
        }
    }
}

pub struct PickerImage {}

/// Renderer for the quick and dirty picker textures.
/// Currently, an `R32G32` texture is used to store an `Option<NonZerou64>` `FuzzID` in native endian.
///
/// This implementation detail is hidden behind the `Picker` trait, as other implementations are possible too -
/// Another idea is to use a lower color depth texture for decreased memory usage/bandwidth and have a host-side
/// map between texture values and `FuzzIDs`.
/// After all, `u64::MAX` distinct objects in a single image is *highly unlikely*. :P
pub struct PickerRenderer {}
impl PickerRenderer {
    fn make_pipeline(device: Arc<vk::Device>) -> anyhow::Result<Arc<vk::GraphicsPipeline>> {
        let vert = shaders::vert::load(device.clone())?;
        let geom = shaders::geom::load(device.clone())?;
        let frag = shaders::frag::load(device.clone())?;
        // Unwraps OK - main is the only valid GLSL entrypoint.
        let vert = vert.entry_point("main").unwrap();
        let geom = geom.entry_point("main").unwrap();
        let frag = frag.entry_point("main").unwrap();

        let stages = smallvec::smallvec![
            vk::PipelineShaderStageCreateInfo::new(vert.clone()),
            vk::PipelineShaderStageCreateInfo::new(geom),
            vk::PipelineShaderStageCreateInfo::new(frag),
        ];

        let matrix_range = vk::PushConstantRange {
            offset: 0,
            size: 4 * 4 * 4, // mat4
            stages: vk::ShaderStages::VERTEX | vk::ShaderStages::GEOMETRY,
        };
        let ne_id_range = vk::PushConstantRange {
            offset: 4 * 4 * 4, //after mat4
            size: 4 * 2,       // uvec2
            stages: vk::ShaderStages::FRAGMENT,
        };
        let layout = vk::PipelineLayout::new(
            device.clone(),
            vk::PipelineLayoutCreateInfo {
                push_constant_ranges: vec![matrix_range, ne_id_range],
                ..Default::default()
            },
        )?;

        vk::GraphicsPipeline::new(
            device,
            None,
            vk::GraphicsPipelineCreateInfo {
                subpass: Some(vk::PipelineSubpassType::BeginRendering(
                    vk::PipelineRenderingCreateInfo {
                        color_attachment_formats: vec![Some(NE_ID_FORMAT)],
                        ..Default::default()
                    },
                )),
                stages,
                // Scissor irrelevant, viewport dynamic.
                dynamic_state: [vk::DynamicState::Viewport].into_iter().collect(),
                viewport_state: Some(vk::ViewportState::default()),
                input_assembly_state: Some(vk::InputAssemblyState {
                    primitive_restart_enable: false,
                    topology: vk::PrimitiveTopology::LineStrip,
                    ..Default::default()
                }),
                rasterization_state: Some(vk::RasterizationState::default()),
                vertex_input_state: Some(
                    crate::StrokePoint::per_instance().definition(&vert.info().input_interface)?,
                ),

                // No blend - use as is!
                color_blend_state: Some(vk::ColorBlendState::with_attachment_states(
                    1,
                    vk::ColorBlendAttachmentState::default(),
                )),
                ..vk::GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .map_err(Into::into)
    }
    pub fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
        let pipeline = Self::make_pipeline(context.device().clone())?;
        todo!()
    }
}
