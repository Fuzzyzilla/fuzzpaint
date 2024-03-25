use crate::vulkano_prelude::*;
use std::sync::Arc;

#[derive(vk::Vertex, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct GizmoVertex {
    /// Position, in the gizmo's local coordinates.
    #[format(R32G32_SFLOAT)]
    pub pos: [f32; 2],
    /// Straight, linear RGBA color
    #[format(R8G8B8A8_UNORM)]
    pub color: [u8; 4],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
}

#[derive(Clone, Copy, crate::vk::Vertex, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct WideLineVertex {
    /// X,Y in coordinate space determined by `GizmoTransform`
    #[format(R32G32_SFLOAT)]
    pub pos: [f32; 2],
    /// Multiplied by the texture and gizmo color. Non-pre-multiplied linear.
    #[format(R8G8B8A8_UNORM)]
    pub color: [u8; 4],
    /// One-dimensional texture coordinate. Corresponds to the U component in the output
    /// wide-line, where V increases from zero to one from "right" to "left" (relative to the line's forward vector)
    #[format(R32_SFLOAT)]
    pub tex_coord: f32,
    /// Diameter of the line, in the same unit as `pos`
    /// TODO: would be nice to have a separate coordinate space for this :V
    #[format(R32_SFLOAT)]
    pub width: f32,
}

#[derive(PartialEq, Eq)]
enum VertexBuffer {
    WideLines(vk::Subbuffer<[WideLineVertex]>),
    Normal(vk::Subbuffer<[GizmoVertex]>),
}

mod shaders {
    #[derive(Copy, Clone, Eq, PartialEq, Hash, strum::EnumCount)]
    pub enum VertexProcessing {
        Normal,
        WideLine,
    }
    #[derive(Copy, Clone, Eq, PartialEq, Hash, strum::EnumCount)]
    pub enum FragmentProcessing {
        Solid,
        Textured,
        AntTrail,
    }

    pub fn processing_of(
        visual: &super::super::Visual,
    ) -> Option<(VertexProcessing, FragmentProcessing)> {
        use super::super::{MeshMode, TextureMode};
        let vertex = match visual.mesh {
            MeshMode::None => return None,
            MeshMode::Shape(..) | MeshMode::Triangles => VertexProcessing::Normal,
            MeshMode::WideLineStrip(..) => VertexProcessing::WideLine,
        };
        let fragment = match visual.texture {
            TextureMode::AntTrail => FragmentProcessing::AntTrail,
            TextureMode::Solid(..) => FragmentProcessing::Solid,
            TextureMode::Texture { .. } => FragmentProcessing::Textured,
        };

        Some((vertex, fragment))
    }
    #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
    #[repr(C)]
    pub struct PushConstants {
        /// Should be an affine2 transform, but glsl std430 align and packing of such is unclear.
        pub transform: [[f32; 4]; 4],
        /// The color the whole object is multiplied by.
        pub color: [f32; 4],
    }
    pub mod vertex {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r#"#version 460
            
            layout(std430, push_constant) uniform Push {
                mat4 transform;
                vec4 gizmo_color;
            };

            layout(location = 0) in vec2 pos;
            layout(location = 1) in vec4 color;
            layout(location = 2) in vec2 uv;

            layout(location = 0) out vec4 outColor;
            layout(location = 1) out vec2 outUV;

            void main() {
                outColor = color * gizmo_color;
                outUV = uv;

                gl_Position = transform * vec4(pos, 0.0, 1.0);
            }
            "#
        }
    }
    pub mod fragment_textured {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r#"#version 460
            
            layout(set = 0, binding = 0) uniform sampler2D tex;

            layout(location = 0) in vec4 inColor;
            layout(location = 1) in vec2 inUV;

            layout(location = 0) out vec4 outColor;

            void main() {
                outColor = texture(tex, inUV) * inColor;
            }
            "#
        }
    }
    pub mod fragment_untextured {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r#"#version 460

            layout(location = 0) in vec4 inColor;
            layout(location = 1) in vec2 _;

            layout(location = 0) out vec4 outColor;

            void main() {
                outColor = inColor;
            }
            "#
        }
    }
    pub mod fragment_ant_trail {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r#"#version 460

            // Arbitrary looping time, [0, 1).
            layout(location = 0) in vec4 inTime;
            layout(location = 1) in vec2 _1;

            layout(location = 0) out vec4 outColor;
            const uint PERIOD = 8;
            const float TAU = 6.283185307;

            void main() {
                // We could get a positive/negative color effect by configuring the blend hardware to:
                // SRC_COLOR * ONE + DST_COLOR * SRC_ALPHA and using [0, 0, 0, 1] for positive, [1, 1, 1, -1] for negative.
                // However, on grey this wouldn't be visible. We can use RGB in this situation to add some other color constant?
                // ISSUE: drawing directly into the swapchain image, which is UNORM. no negative alpha allowed :,<
                
                // For now, just black/white stripe.

                uint pos = uint(gl_FragCoord.x + gl_FragCoord.y);
                // Over the course of a loop, move a whole period.
                pos += uint(inTime.a * float(PERIOD));

                // Like a periodic threshold but smoother, purely arbitrary lol
                float phase = float(pos) / float(PERIOD) * TAU;
                float brite = sin(phase) * 0.5 + 0.5;
                outColor = vec4(vec3(brite * brite), 0.75);
            }
            "#
        }
    }
    pub mod thick_polyline {
        pub mod vert {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: r#"#version 460
                
                layout(std430, push_constant) uniform Push {
                    mat4 transform;
                    vec4 gizmo_color;
                };
    
                layout(location = 0) in vec2 pos;
                layout(location = 1) in vec4 color;
                // Polyline has a single-dimension UV, or just U (that would be confusing tho lol)
                // Expands into UV in the widening geometry shader
                layout(location = 2) in float tex_coord;
                // Width, in transform units.
                layout(location = 3) in float width;
    
                layout(location = 0) out vec4 out_color;
                layout(location = 1) out float out_texcoord;
                layout(location = 2) out float out_width;
    
                void main() {
                    out_color = color * gizmo_color;
                    out_texcoord = tex_coord;
                    out_width = width;

                    gl_Position = transform * vec4(pos, 0.0, 1.0);
                }
                "#
            }
        }
        // Takes lines adjacency and turns each segment into a
        // wide rectangle.
        pub mod geom {
            // Cryptic defines to configure the widelines shader.
            // See `src/shaders/widelines.geom` for docs.

            // uv mighhtttt not be working :V
            vulkano_shaders::shader! {
                ty: "geometry",
                define: [
                    ("WIDTH_LOCATION", "2"),
                    ("INPUTS", r"
                layout(location = 0) in vec4 in_color[4];
                layout(location = 1) in float in_texcoord[4];
                "),
                    ("IN_U_NAME", "in_texcoord"),
                    ("OUTPUTS", r"
                layout(location = 0) out vec4 out_color;
                layout(location = 1) out vec2 out_uv;
                "),
                    ("OUT_UV_NAME", "out_uv"),
                    ("COPY_B", "out_color = in_color[B];"),
                    ("COPY_C", "out_color = in_color[C];"),
                ],
                path: "src/shaders/widelines.geom",
            }
        }
    }
}

mod arc_tools {
    use std::{
        cmp::{Eq, PartialEq},
        sync::{Arc, Weak},
    };
    /// Arc wrapper that implements `Hash`, `Eq`, based on pointers instead of contents.
    ///
    /// In order for this to be sound, `T` should not be interior mutable or `dyn`.
    #[repr(transparent)]
    pub struct ArcByPtr<T: ?Sized>(pub Arc<T>);
    impl<T: ?Sized> std::hash::Hash for ArcByPtr<T> {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            state.write_usize(Arc::as_ptr(&self.0).cast::<()>() as usize);
        }
    }
    impl<T: ?Sized> std::ops::Deref for ArcByPtr<T> {
        type Target = Arc<T>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<T: ?Sized> PartialEq for ArcByPtr<T> {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.0, &other.0)
        }
    }
    impl<T: ?Sized> Eq for ArcByPtr<T> {}

    impl<T: ?Sized> PartialEq<WeakByPtr<T>> for ArcByPtr<T> {
        fn eq(&self, other: &WeakByPtr<T>) -> bool {
            // Soundness - WeakByPtr is made from a real Arc, so it is mem backed and ptr is stable.
            // Use addr_eq to compare thin ptrs - same behavior as `{Arc, Weak}::ptr_eq`
            std::ptr::addr_eq(Arc::as_ptr(&self.0), Weak::as_ptr(&other.0))
        }
    }

    // this weak is NOT public.
    // It is unsound to construct this with a Weak::new, it MUST come from Arc
    /// Weak wrapper that implements `Hash`, `Eq`, based on pointers instead of contents.
    /// Never dangling.
    ///
    /// In order for this to be sound, `T` should not be interior mutable or `dyn`.
    #[repr(transparent)]
    pub struct WeakByPtr<T: ?Sized>(Weak<T>);
    impl<T: ?Sized> WeakByPtr<T> {
        pub fn from_arc(from: &Arc<T>) -> Self {
            Self(Arc::downgrade(from))
        }
    }
    impl<T: ?Sized> std::ops::Deref for WeakByPtr<T> {
        type Target = Weak<T>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<T: ?Sized> std::hash::Hash for WeakByPtr<T> {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            // Soundness - This weak is backed by heap mem, this ptr is stable!
            state.write_usize(Weak::as_ptr(&self.0).cast::<()>() as usize);
        }
    }
    impl<T: ?Sized> PartialEq for WeakByPtr<T> {
        fn eq(&self, other: &Self) -> bool {
            // Soundness - Both weaks are backed by heap mem, ptrs stable!
            Weak::ptr_eq(&self.0, &other.0)
        }
    }
    impl<T: ?Sized> Eq for WeakByPtr<T> {}
    impl<T: ?Sized> PartialEq<ArcByPtr<T>> for WeakByPtr<T> {
        fn eq(&self, other: &ArcByPtr<T>) -> bool {
            other == self
        }
    }
}
pub struct Renderer {
    context: Arc<crate::render_device::RenderContext>,
    /// Map from processings -> compiled pipeline.
    lazy_pipelines: parking_lot::RwLock<
        hashbrown::HashMap<
            (shaders::VertexProcessing, shaders::FragmentProcessing),
            std::sync::Arc<vk::GraphicsPipeline>,
        >,
    >,
    interned_widelines: parking_lot::Mutex<
        hashbrown::HashMap<
            // Soundness - [WideLineVertex] is neither interior mutable nor dyn.
            // We CANNOT use *T for this, as that addr can be re-used if freed and allocated again.
            // Weak will keep the alloc alive and stable.
            arc_tools::WeakByPtr<[WideLineVertex]>,
            vk::Subbuffer<[WideLineVertex]>,
        >,
    >,

    // Premade, static vertex buffers for common shapes.
    triangulated_shapes: vk::Subbuffer<[GizmoVertex]>,
    triangulated_square: vk::Subbuffer<[GizmoVertex]>,
    triangulated_circle: vk::Subbuffer<[GizmoVertex]>,
}
impl Renderer {
    const CIRCLE_RES: usize = 32;
    /// Make static shape buffers. (unit square origin at 0.0, unit circle origin at 0.0)
    fn make_shapes(
        context: &crate::render_device::RenderContext,
    ) -> anyhow::Result<(
        vk::Subbuffer<[GizmoVertex]>,
        vk::Subbuffer<[GizmoVertex]>,
        vk::Subbuffer<[GizmoVertex]>,
    )> {
        let mut vertices = Vec::with_capacity(6 + Self::CIRCLE_RES * 3);
        // Construct square
        {
            let top_right = GizmoVertex {
                pos: [1.0, 0.0],
                uv: [1.0, 1.0],
                color: [255; 4],
            };
            let bottom_left = GizmoVertex {
                pos: [0.0, 1.0],
                uv: [0.0, 0.0],
                color: [255; 4],
            };
            vertices.extend_from_slice(&[
                GizmoVertex {
                    pos: [0.0, 0.0],
                    uv: [0.0, 1.0],
                    color: [255; 4],
                },
                top_right,
                bottom_left,
                bottom_left,
                top_right,
                GizmoVertex {
                    pos: [1.0, 1.0],
                    uv: [1.0, 0.0],
                    color: [255; 4],
                },
            ]);
        };
        // construct circle
        let circle = (0..Self::CIRCLE_RES).flat_map(|idx| {
            let proportion = std::f32::consts::TAU * (idx as f32) / (Self::CIRCLE_RES as f32);
            let proportion_next =
                std::f32::consts::TAU * ((idx + 1) as f32) / (Self::CIRCLE_RES as f32);

            let (first_y, first_x) = proportion.sin_cos();
            let (next_y, next_x) = proportion_next.sin_cos();

            [
                // Center
                GizmoVertex {
                    pos: [0.0, 0.0],
                    uv: [0.5, 0.5],
                    color: [255; 4],
                },
                GizmoVertex {
                    pos: [first_x, first_y],
                    uv: [first_x / 2.0 + 0.5, 1.0 - (first_y / 2.0 + 0.5)],
                    color: [255; 4],
                },
                GizmoVertex {
                    pos: [next_x, next_y],
                    uv: [next_x / 2.0 + 0.5, 1.0 - (next_y / 2.0 + 0.5)],
                    color: [255; 4],
                },
            ]
        });
        vertices.extend(circle);
        let triangulated_shapes = vk::Buffer::from_iter(
            context.allocators().memory().clone(),
            vk::BufferCreateInfo {
                sharing: vk::Sharing::Exclusive,
                usage: vk::BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                // Prefer device memory, slow access from CPU is fine but it needs to write once.
                // Should this be staged?
                memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices.into_iter(),
        )?;

        let square = triangulated_shapes.clone().slice(0..6);
        let circle = triangulated_shapes.clone().slice(6..);

        Ok((triangulated_shapes, square, circle))
    }

    fn vertices_for(&self, mesh_mode: &super::MeshMode) -> anyhow::Result<VertexBuffer> {
        match mesh_mode {
            super::MeshMode::None => anyhow::bail!("requested mesh for MeshMode::None"),
            super::MeshMode::Shape(s) => match s {
                super::RenderShape::Ellipse { .. } => {
                    Ok(VertexBuffer::Normal(self.triangulated_circle.clone()))
                }
                super::RenderShape::Rectangle { .. } => {
                    Ok(VertexBuffer::Normal(self.triangulated_square.clone()))
                }
            },
            super::MeshMode::WideLineStrip(mesh) => {
                self.intern_wide_lines(mesh).map(VertexBuffer::WideLines)
            }
            super::MeshMode::Triangles => unimplemented!(),
        }
    }
    /// Intern this collection of wide lines into a buffer slice.
    /// Maintains a Weak pointer to it, so that the buffer may be freed
    /// when it becomes inaccessible.
    fn intern_wide_lines(
        &self,
        mesh_mode: &std::sync::Arc<[WideLineVertex]>,
    ) -> anyhow::Result<vk::Subbuffer<[WideLineVertex]>> {
        let data = mesh_mode.as_ref();
        if data.is_empty() {
            anyhow::bail!("cannot upload empty wide lines buffer");
        }
        let mut map = self.interned_widelines.lock();
        // TODO: how often to call this?
        Self::cleanup_wide_lines(&mut map);

        match map.entry(arc_tools::WeakByPtr::from_arc(mesh_mode)) {
            hashbrown::hash_map::Entry::Occupied(o) => Ok(o.get().clone()),
            hashbrown::hash_map::Entry::Vacant(v) => {
                let buffer = vk::Buffer::new_slice::<WideLineVertex>(
                    self.context.allocators().memory().clone(),
                    vk::BufferCreateInfo {
                        usage: vk::BufferUsage::VERTEX_BUFFER,
                        sharing: vk::Sharing::Exclusive,
                        ..Default::default()
                    },
                    vk::AllocationCreateInfo {
                        memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    data.len().try_into()?,
                )?;
                // Unwrap ok - we definitely have exclusive access cause we just made it!
                {
                    let mut write = buffer.write().unwrap();
                    // Won't panic. We made the buffer with size data.len().
                    write.copy_from_slice(data);
                }

                Ok(v.insert(buffer).clone())
            }
        }
    }
    /// Cleans up every buffer which is no longer accessible.
    fn cleanup_wide_lines(
        map: &mut hashbrown::HashMap<
            arc_tools::WeakByPtr<[WideLineVertex]>,
            vk::Subbuffer<[WideLineVertex]>,
        >,
    ) {
        // Remove all which no strong pointers exist anymore, and are thus gone.
        // Subbuffer::drop should do all the cleanup we need.
        map.retain(|pointer, _| pointer.strong_count() > 0);
    }
    /// Visuals specify some combination of Vertex processing and Texturing.
    /// As more options for each of these are added, it would be silly to create them in
    /// bulk, instead they are built lazily as needed.
    #[allow(clippy::too_many_lines)]
    fn lazy_pipeline_for(
        &self,
        vertex: shaders::VertexProcessing,
        fragment: shaders::FragmentProcessing,
    ) -> anyhow::Result<std::sync::Arc<vk::GraphicsPipeline>> {
        let key = (vertex, fragment);
        {
            // NOT upgradable read, as that would just be a simple mutex and
            // we get no benifit for RwLocking. The miss case is very rare in steadystate!
            let read = self.lazy_pipelines.read();
            if let Some(pipeline) = read.get(&key) {
                return Ok(pipeline.clone());
            }
        }
        // Fellthrough! We need to build this pipeline.
        let mut write = self.lazy_pipelines.write();
        let entry = write.entry(key);
        match entry {
            // Since the upgrade from read -> write was *not* atomic, it may have become
            // available in the meantime!
            hashbrown::hash_map::Entry::Occupied(o) => Ok(o.get().clone()),
            hashbrown::hash_map::Entry::Vacant(v) => {
                let device = self.context.device().clone();
                let vertex_format = match vertex {
                    shaders::VertexProcessing::Normal => GizmoVertex::per_vertex(),
                    shaders::VertexProcessing::WideLine => WideLineVertex::per_vertex(),
                };
                let topology = match vertex {
                    shaders::VertexProcessing::Normal => vk::PrimitiveTopology::TriangleList,
                    shaders::VertexProcessing::WideLine => {
                        vk::PrimitiveTopology::LineStripWithAdjacency
                    }
                };
                let texture_descriptor = if fragment == shaders::FragmentProcessing::Textured {
                    Some(vk::DescriptorSetLayout::new(
                        device.clone(),
                        vk::DescriptorSetLayoutCreateInfo {
                            bindings: [(
                                0,
                                vk::DescriptorSetLayoutBinding {
                                    descriptor_count: 1,
                                    stages: vk::ShaderStages::FRAGMENT,
                                    ..vk::DescriptorSetLayoutBinding::descriptor_type(
                                        vk::DescriptorType::CombinedImageSampler,
                                    )
                                },
                            )]
                            .into_iter()
                            .collect(),
                            ..Default::default()
                        },
                    )?)
                } else {
                    None
                };
                let (vertex, geometry) = match vertex {
                    shaders::VertexProcessing::Normal => {
                        (shaders::vertex::load(device.clone())?, None)
                    }
                    shaders::VertexProcessing::WideLine => (
                        shaders::thick_polyline::vert::load(device.clone())?,
                        Some(shaders::thick_polyline::geom::load(device.clone())?),
                    ),
                };
                let fragment = match fragment {
                    shaders::FragmentProcessing::AntTrail => {
                        shaders::fragment_ant_trail::load(device.clone())?
                    }
                    shaders::FragmentProcessing::Solid => {
                        shaders::fragment_untextured::load(device.clone())?
                    }
                    shaders::FragmentProcessing::Textured => {
                        shaders::fragment_textured::load(device.clone())?
                    }
                };

                let push_constant_ranges = {
                    let mut ranges = Vec::with_capacity(2);
                    // Vertex always needs xform and color
                    let matrix_color_range = vk::PushConstantRange {
                        offset: 0,
                        stages: vk::ShaderStages::VERTEX,
                        size: 4 * 4 * 4 + 4 * 4, //4x4 matrix of f32, + vec4 of f32
                    };
                    ranges.push(matrix_color_range);

                    // If geometry, give it access to the xform
                    if geometry.is_some() {
                        let matrix_range = vk::PushConstantRange {
                            offset: 0,
                            stages: vk::ShaderStages::GEOMETRY,
                            size: 4 * 4 * 4, //4x4 matrix of f32
                        };
                        ranges.push(matrix_range);
                    }
                    ranges
                };

                let mut stages = smallvec::smallvec![];
                // Unwraps OK - main is the only valid GLSL entrypoint.
                let vertex_entry = vertex.entry_point("main").unwrap();
                stages.push(vk::PipelineShaderStageCreateInfo::new(vertex_entry.clone()));
                if let Some(geometry) = geometry {
                    stages.push(vk::PipelineShaderStageCreateInfo::new(
                        geometry.entry_point("main").unwrap(),
                    ));
                }
                stages.push(vk::PipelineShaderStageCreateInfo::new(
                    fragment.entry_point("main").unwrap(),
                ));

                let color_blend_state = {
                    let blend_mode = vk::AttachmentBlend::alpha();
                    let blend_states = vk::ColorBlendAttachmentState {
                        blend: Some(blend_mode),
                        ..Default::default()
                    };
                    vk::ColorBlendState::with_attachment_states(1, blend_states)
                };

                let input_assembly_state = vk::InputAssemblyState {
                    topology,
                    primitive_restart_enable: false,
                    ..Default::default()
                };
                // ad hoc rendering for now, lazy lazy
                let subpass =
                    vk::PipelineSubpassType::BeginRendering(vk::PipelineRenderingCreateInfo {
                        // FIXME!
                        color_attachment_formats: vec![Some(vk::Format::B8G8R8A8_SRGB)],
                        ..Default::default()
                    });

                let layout = vk::PipelineLayout::new(
                    device.clone(),
                    vk::PipelineLayoutCreateInfo {
                        push_constant_ranges,
                        // Empty, or the image if some.
                        set_layouts: texture_descriptor.into_iter().collect(),
                        ..Default::default()
                    },
                )?;

                let graphics_pipe = vk::GraphicsPipeline::new(
                    device,
                    None,
                    vk::GraphicsPipelineCreateInfo {
                        stages,
                        vertex_input_state: Some(
                            vertex_format.definition(&vertex_entry.info().input_interface)?,
                        ),
                        input_assembly_state: Some(input_assembly_state),
                        // Viewport doesn't matter (dynamic) scissor irrelevant.
                        viewport_state: Some(vk::ViewportState::default()),
                        // don't cull
                        rasterization_state: Some(vk::RasterizationState::default()),
                        multisample_state: Some(vk::MultisampleState::default()),
                        color_blend_state: Some(color_blend_state),
                        dynamic_state: [vk::DynamicState::Viewport].into_iter().collect(),
                        subpass: Some(subpass),

                        ..vk::GraphicsPipelineCreateInfo::layout(layout)
                    },
                )?;

                Ok(v.insert(graphics_pipe).clone())
            }
        }
    }
    pub fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
        let (shapes, square, circle) = Self::make_shapes(context.as_ref())?;

        // Largest possible is the combinations of both!
        let lazy_pipelines = hashbrown::HashMap::with_capacity(
            <shaders::VertexProcessing as strum::EnumCount>::COUNT
                * <shaders::FragmentProcessing as strum::EnumCount>::COUNT,
        );

        Ok(Self {
            context,
            lazy_pipelines: lazy_pipelines.into(),
            interned_widelines: hashbrown::HashMap::new().into(),
            triangulated_shapes: shapes,
            triangulated_circle: circle,
            triangulated_square: square,
        })
    }
    // Temporary api. passing around swapchain images and proj matrices like this feels dirty :P
    pub fn render_visit(
        &self,
        into_image: Arc<vk::Image>,
        image_size: [f32; 2],
        document_transform: crate::view_transform::ViewTransform,
        proj: cgmath::Matrix4<f32>,
    ) -> anyhow::Result<RenderVisitor<'_>> {
        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            self.context.allocators().command_buffer(),
            self.context.queues().graphics().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;
        let attachment = vk::RenderingAttachmentInfo {
            clear_value: None,
            load_op: vk::AttachmentLoadOp::Load,
            store_op: vk::AttachmentStoreOp::Store,
            ..vk::RenderingAttachmentInfo::image_view(vk::ImageView::new_default(into_image)?)
        };
        command_buffer
            .begin_rendering(vk::RenderingInfo {
                color_attachments: vec![Some(attachment)],
                contents: vk::SubpassContents::Inline,
                depth_attachment: None,
                ..Default::default()
            })?
            .set_viewport(
                0,
                smallvec::smallvec![vk::Viewport {
                    depth_range: 0.0..=1.0,
                    extent: image_size,
                    offset: [0.0; 2],
                }],
            )?;

        Ok(RenderVisitor {
            renderer: self,
            xform_stack: vec![document_transform],
            command_buffer,
            current_pipeline: None,
            proj,
        })
    }
}

pub struct RenderVisitor<'a> {
    renderer: &'a Renderer,
    xform_stack: Vec<crate::view_transform::ViewTransform>,
    command_buffer: vk::AutoCommandBufferBuilder<vk::PrimaryAutoCommandBuffer>,
    current_pipeline: Option<Arc<vk::GraphicsPipeline>>,
    // Rebinds after every change, even if they're adjacent D:
    // this optimization thus is worthless....
    // would be nice to use a big buffer and just cursor around it with first_vertex, todo!
    // current_vertex_buffer: Option<VertexBuffer>,
    proj: cgmath::Matrix4<f32>,
}
impl RenderVisitor<'_> {
    pub fn build(mut self) -> anyhow::Result<Arc<vk::PrimaryAutoCommandBuffer>> {
        self.command_buffer.end_rendering()?;
        let build = self.command_buffer.build()?;

        Ok(build)
    }
}
// Thats... a unique way to propogate errors... :P
impl<'a> super::GizmoVisitor<anyhow::Error> for RenderVisitor<'a> {
    fn visit_collection(
        &mut self,
        gizmo: &super::Collection,
    ) -> std::ops::ControlFlow<anyhow::Error> {
        let Some(parent_xform) = self.xform_stack.last() else {
            // Shouldn't happen! visit_ and end_collection should be symmetric.
            // Some to short circuit the visitation
            return std::ops::ControlFlow::Break(anyhow::anyhow!("xform stack empty!"));
        };
        // unwrap ok - checked above.
        let base_xform = self.xform_stack.first().unwrap();
        let new_xform = gizmo.transform.apply(base_xform, parent_xform);

        self.xform_stack.push(new_xform);

        std::ops::ControlFlow::Continue(())
    }
    fn visit_gizmo(&mut self, gizmo: &super::Gizmo) -> std::ops::ControlFlow<anyhow::Error> {
        // Skip if nothing to draw.
        let Some((vertex, fragment)) = shaders::processing_of(&gizmo.visual) else {
            return std::ops::ControlFlow::Continue(());
        };
        // try_block macro doesn't impl FnMut it's kinda weird :V
        // We use this to map Result<> to ControlFlow
        let mut try_block = || -> anyhow::Result<()> {
            // Calculate Xform
            let Some(parent_xform) = self.xform_stack.last() else {
                // Shouldn't happen! visit_ and end_collection should be symmetric.
                anyhow::bail!("xform stack empty!")
            };
            // unwrap ok - checked above.
            let base_xform = self.xform_stack.first().unwrap();
            let local_xform = gizmo.transform.apply(base_xform, parent_xform);

            // `MeshMode::None` handled gracefully above
            if matches!(&gizmo.visual.mesh, super::MeshMode::Triangles) {
                anyhow::bail!("todo!")
            }

            let shape_xform: cgmath::Matrix4<f32> = match &gizmo.visual.mesh {
                super::MeshMode::Shape(shape) => {
                    let (offs, scale, rotation) = match *shape {
                        super::RenderShape::Rectangle {
                            position,
                            size,
                            rotation,
                        } => (position, size, rotation),
                        super::RenderShape::Ellipse {
                            origin,
                            radii,
                            rotation,
                        } => (origin, radii, rotation),
                    };
                    cgmath::Matrix4::from_translation(cgmath::Vector3 {
                        x: offs.x,
                        y: offs.y,
                        z: 0.0,
                    }) * cgmath::Matrix4::from_nonuniform_scale(scale.x, scale.y, 1.0)
                        * cgmath::Matrix4::from_angle_z(cgmath::Rad(rotation))
                }
                _ => <cgmath::Matrix4<_> as cgmath::One>::one(),
            };
            let matrix: cgmath::Matrix4<f32> = local_xform.into();
            // Stretch/position shape, then move from local to viewspace, then project to NDC
            let matrix = self.proj * matrix * shape_xform;
            let color = match gizmo.visual.texture {
                super::TextureMode::AntTrail => {
                    // Hack to give AntTrail access to the current time, since it does not accept a color.
                    let time_millisecs = std::time::SystemTime::now()
                        .duration_since(std::time::SystemTime::UNIX_EPOCH)
                        .ok()
                        .map_or(0, |dur| dur.subsec_millis());
                    // 0..250, looping.
                    // time_millisecs ranges from 0..1000 already but just to prove the unwrap is sound :P
                    let time: u8 = (time_millisecs % 1000 / 4).try_into().unwrap();
                    [time; 4]
                }
                super::TextureMode::Solid(c) => c,
                super::TextureMode::Texture { modulate: _, .. } => {
                    // Todo: bind texture descriptor.
                    unimplemented!();
                }
            };
            let push_constants = shaders::PushConstants {
                color: [
                    f32::from(color[0]) / 255.0,
                    f32::from(color[1]) / 255.0,
                    f32::from(color[2]) / 255.0,
                    f32::from(color[3]) / 255.0,
                ],
                transform: matrix.into(),
            };

            let pipeline = self.renderer.lazy_pipeline_for(vertex, fragment)?;
            // Not the same, rebind!
            // Push constants ALWAYS compatible.
            // Descriptors are not compatible,
            if self.current_pipeline.as_ref() != Some(&pipeline) {
                self.command_buffer
                    .bind_pipeline_graphics(pipeline.clone())?;
                self.current_pipeline = Some(pipeline.clone());
            }

            let vertex_buffer = self.renderer.vertices_for(&gizmo.visual.mesh)?;
            let num_verts = match vertex_buffer {
                VertexBuffer::Normal(n) => {
                    let len = n.len();
                    self.command_buffer.bind_vertex_buffers(0, n)?;
                    len
                }
                VertexBuffer::WideLines(w) => {
                    let len = w.len();
                    self.command_buffer.bind_vertex_buffers(0, w)?;
                    len
                }
            };

            self.command_buffer
                .push_constants(pipeline.layout().clone(), 0, push_constants)?
                .draw(num_verts.try_into()?, 1, 0, 0)?;
            Ok(())
        };
        match try_block() {
            Ok(()) => std::ops::ControlFlow::Continue(()),
            Err(anyhow) => std::ops::ControlFlow::Break(anyhow),
        }
    }
    fn end_collection(&mut self, _: &super::Collection) -> std::ops::ControlFlow<anyhow::Error> {
        if self.xform_stack.pop().is_some() {
            std::ops::ControlFlow::Continue(())
        } else {
            // would be a gizmo implementation error.
            std::ops::ControlFlow::Break(anyhow::anyhow!("Unbalanced gizmo tree!"))
        }
    }
}
