use crate::picker::Picker;
use crate::vulkano_prelude::*;
use std::sync::Arc;

mod stage;

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
    fn pick(&self, _: ultraviolet::Vec2) -> Option<Self::Value> {
        Some(self.color)
    }
}
/// Picker that acts on rendered image output, yielding linear, premultiplied RGBA.
/// This output could be a single layer, or a composite image.
///
/// Filtering is done "Nearest Neighbor"
pub struct RenderedColorPicker {
    offset: (u32, u32),
    extent: (u32, u32),
}
impl RenderedColorPicker {
    pub fn pull_from_image(
        image: Arc<vk::Image>,
        xform: (),
        viewport_rect: (),
    ) -> anyhow::Result<Self> {
        todo!()
    }
}
impl Picker for RenderedColorPicker {
    type Value = [vulkano::half::f16; 4];
    fn pick(&self, viewport_coordinate: ultraviolet::Vec2) -> Option<Self::Value> {
        todo!()
    }
}

/// Picker from NE_ID image. These must be produced separately from the usual pipeline,
/// but yield a reference to the clicked stroke. This allows for precise color selection,
/// brush setting selection, ect.
///
/// Could be a single-layer image, or a composite.
pub struct StrokeIDPicker {}
impl crate::picker::Picker for StrokeIDPicker {
    type Value = crate::state::stroke_collection::ImmutableStrokeID;
    fn pick(&self, viewport_coordinate: ultraviolet::Vec2) -> Option<Self::Value> {
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
