//! # Text rendering
//!
//! We will use `ttf_parser` to load face data, `rustybuzz` to shape it (turn unicode `&str` into a list of
//! face character indices and locations), and `lyon` to tesselate each unique requested `(face, variation, index, maxsize)` combination.
//!
//! The usual text rendering scheme is to make various pre-rendered textures containing the characters, rendering them
//! onto quads. Instead, we opt to tesselate them into actual meshes, avoiding issues with hinting becoming off when resized.
//! Images may still be of good use, particularly for UI - prerendering the font's name in the font itself to an image would be great!
//!
//! Characters are loaded and tesselated as needed, and stored in an LRU in potentially several levels-of-detail. Using lyon,
//! we tesselate with 1px of tolerance, which gives very different results for large vs small text. We can always use
//! a larger character tesselation, but never smaller - thus several versions of the same character may exist.
//!
//! Faces may also specify a PNG or bitmap - we won't handle them yet, but they can be handled using standard HW-accelerated
//! mipmapping. It does create limited quality which goes against the whole spirit of this app! D:
//!
//! The result is a large vertex + index buffer of assorted chars, an instance buffer containing transforms, and a set of
//! instanced+indexed indirect draws.
//!
//! ~~Fonts without color data have vertices consisting of only position. With color data adds an extra vertex color attribute,
//! and will be handled separately - thus, a shaping request can end up with up to 2 batches to draw, with differing pipelines.~~
//! *correction: turns out that the extent of color support in `ttf_parser` (`COLRv0`) is fully representable by layered instance-colored glyphs!*

use rustybuzz::ttf_parser;
use vulkano::buffer::Subbuffer;

use crate::vulkano_prelude::*;
mod cache;
pub use cache::{CacheInsertError, ColorInsertError};
mod tessellator;
pub use tessellator::{ColorError, TessellateError};

/// Based on the `fsType` field of OTF/os2. Does not correspond exactly with those flags.
///
/// Describes the licensing restriction of a font in increasing permissiveness, and how it may be used in relation
/// to exported fuzzpaint documents. The comment description is NOT legal advice, but
/// [Microsoft's](https://learn.microsoft.com/en-us/typography/opentype/spec/os2#fst) is probably closer.
#[derive(Copy, Clone, Debug)]
pub enum EmbedRestrictionLevel {
    /// The font data cannot be embedded for our purposes.
    LocalOnly,
    /// The font data may be embedded into a read-only fuzzpaint document - no part of the document can be modified on a
    /// system that did not author it as long as this font is embedded. The wording in Microsoft's document is permissive
    /// enough to allow opening the document in a writable mode, as long as the font data is not loaded in this process and is discarded.
    ReadOnly,
    /// The font data may be embedded into an editable fuzzpaint document.
    Writable,
    /// The font data may be extracted from the document by an end user, for permanent use on their system.
    Installable,
}
#[derive(Copy, Clone, Debug)]
pub struct EmbedRestriction {
    /// Can we save size by embedding just the used subset?
    /// (we don't even have this capability yet and it seems unlikely I'll do it any time soon
    /// + would interfere with document sharability)
    ///
    /// Meaningless if `level == LocalOnly`, but still parsed.
    can_subset: bool,
    /// To what extent are we allowed to embed this font?
    /// Bitmap-only embeds are considered Non-embeddable, as that defeats the whole point.
    level: EmbedRestrictionLevel,
}
impl EmbedRestriction {
    // I originally wrote all of the bitparsing logic for all four versions, before realizing
    // ttf_parser exposes all the fields itself. Whoops!
    /// Extract the embed restrictions from the given table, or None if table failed to parse.
    #[must_use]
    pub fn from_table(table: &ttf_parser::os2::Table) -> Option<Self> {
        let permissions = table.permissions()?;
        // These flags are meaningless unless ^^ is some, that's okay!
        let can_subset = table.is_subsetting_allowed();
        // Misnomer - this checks if bit 9 is zero.
        // If bit unset, this means all data embeddable, set means bitmap only.
        let bitmap_only = !table.is_bitmap_embedding_allowed();

        let level = if bitmap_only {
            // Bitmap-only mode means there's no use in us embedding, at least for our purposes!
            // (todo: for a bitmap-only font, this is actually permissive. We don't handle those yet anyway!)
            EmbedRestrictionLevel::ReadOnly
        } else {
            match permissions {
                ttf_parser::Permissions::Restricted => EmbedRestrictionLevel::LocalOnly,
                ttf_parser::Permissions::PreviewAndPrint => EmbedRestrictionLevel::ReadOnly,
                ttf_parser::Permissions::Editable => EmbedRestrictionLevel::Writable,
                ttf_parser::Permissions::Installable => EmbedRestrictionLevel::Installable,
            }
        };
        Some(Self { can_subset, level })
    }
}
#[derive(Debug)]
pub enum GlyphColorMode {
    /// Static color. This may vary by face variation axes.
    Srgba(ttf_parser::RgbaColor),
    /// Use the user-selected text color
    Foreground,
}

/// Uniquely identifies a glyph from a face with it's variations. Together with [`SizeClass`], identifies a tesselated
/// glyph.
#[derive(Clone, Hash, PartialEq, Eq)]
struct UnsizedGlyphKey {
    face: (), // todo
    // ttf_parser uses [(u32, f32); 32] - 256 bytes total, ouch!
    // todo
    variation: (),
    glyph: ttf_parser::GlyphId,
}
impl UnsizedGlyphKey {
    pub fn from_glyph_face(glyph: ttf_parser::GlyphId, face: &ttf_parser::Face) -> Self {
        Self {
            face: (),
            variation: (),
            glyph,
        }
    }
}
/// A size modifier. Together with [`UnsizedGlyphKey`], identifies a tesselated glyph.
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct SizeClass(
    /// scale factor is `(2**self.0)`.
    ///
    /// limited to -127..=127 to uphold invariants, but practical range is much smaller.
    i8,
);
impl SizeClass {
    /// Smallest representable size class, `2^(-127)`
    pub const MIN: Self = Self(-127);
    /// The size class that represents a 1x scale factor.
    pub const ONE: Self = Self(0);
    /// Largest representable size class, `2^(127)`
    pub const MAX: Self = Self(127);
    /// Calculate the size class from a given scale factor, rounded up to the nearest class.
    /// None if `NaN`, `Inf`, or `<= 0.0`
    #[must_use]
    pub fn from_scale_factor(factor: f32) -> Option<Self> {
        if factor.is_nan() || factor <= 0.0 || factor.is_infinite() {
            None
        } else {
            // pow is -127..=127
            let pow: i8 = if factor.is_normal() {
                // Is this precise?
                // After trying it out, seemingly...!

                // f32::MIN_POSITIVE results in -126
                // Panics on NaN, shouldn't be possible to get that result here.
                az::saturating_cast(factor.log2().ceil())
            } else {
                // Every subnormal has an exponent too small to fit, and would saturate to -128 anyway.
                // 2**-128 is 0 (ish), so use -127 to prevent that.
                -127
            };
            // Guaranteed to be Self::MIN..=Self::MAX
            Some(Self(pow))
        }
    }
    /// Take the `log2` of the size factor. This is an precise operation.
    #[must_use]
    pub fn log2(self) -> i8 {
        self.0
    }
    /// Take the reciprocal of Self. This is a precise operation.
    ///
    /// `1.0/self.scale_factor() == self.recip().scale_factor()` (possibly with greater precision)
    #[must_use]
    pub fn recip(self) -> Self {
        Self(
            // Min val is -127, so unchecked negation is OK
            -self.0,
        )
    }
    /// Get the largest scale factor of this scale class.
    /// Always Non-NaN, finite, and larger than zero.
    #[must_use]
    pub fn scale_factor(self) -> f32 {
        // Seems like 2**x should have a precise fastpath, hmmm...
        // (it does, and it was fun to write, but not worth the loss of expressiveness :P )
        // Self is clamped to -127 to uphold invariant (becomes zero otherwise)
        2.0f32.powi(i32::from(self.0))
    }
}

pub enum OutputColor {
    /// Color varies between glyphs.
    PerInstance,
    /// Every glyph uses the same color.
    Solid([f32; 4]),
}

pub struct DrawOutput {
    /// Full vertex buffer. Not all parts are used. Read-only!
    pub vertices: Subbuffer<[interface::Vertex]>,
    /// Full index buffer. Base vertex is dynamic. Read-only!
    pub indices: Subbuffer<[u16]>,

    /// String-specific instance buffer, to be uploaded prior to drawing.
    pub instances: Vec<interface::Instance>,
    /// String-specific indirects. Handles the dispatch of every indexed instance draw.
    ///
    /// requires `drawIndirectFirstInstance`, or the consumer can draw manually.
    pub indirects: Vec<vk::DrawIndexedIndirectCommand>,

    /// Describes whether rendering can be done in a colorless space.
    /// Instance buffers always contain glyph color regardless.
    pub color_mode: OutputColor,

    /// Finished glyph buffer. Primarily for mem reclaimation.
    pub glyphs: rustybuzz::GlyphBuffer,
}

pub struct TextBuilder {
    tessellator: lyon_tessellation::FillTessellator,
    cache: cache::GlyphCache,
}
impl TextBuilder {
    /// Create a new text builder with default allocated mem.
    pub fn allocate_new(
        memory: std::sync::Arc<dyn vulkano::memory::allocator::MemoryAllocator>,
    ) -> Result<Self, vulkano::Validated<vulkano::buffer::AllocateBufferError>> {
        const BASE_SIZE: u64 = 512 * 1024;
        // Assume 2:1 ratio of indices to vertices. This is a total guess :P
        const INDEX_SIZE: u64 = BASE_SIZE * 2 * std::mem::size_of::<u16>() as u64;
        const VERTEX_SIZE: u64 = BASE_SIZE * 1 * std::mem::size_of::<interface::Vertex>() as u64;
        const TOTAL_SIZE: u64 = INDEX_SIZE + VERTEX_SIZE;
        let buffer = vk::Buffer::new(
            memory,
            vk::BufferCreateInfo {
                sharing: vk::Sharing::Exclusive,
                usage: vk::BufferUsage::VERTEX_BUFFER | vk::BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            vk::AllocationCreateInfo {
                memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | vk::MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            vulkano::memory::allocator::DeviceLayout::new(
                // Of course it's nonzero!
                TOTAL_SIZE.try_into().unwrap(),
                vulkano::memory::DeviceAlignment::of::<interface::Vertex>(),
            )
            // Panics if too big for u64, but it's only a few MB
            .unwrap(),
        )?;
        // Head is for vertices, as it is stricter align than u16
        let vertices = vk::Subbuffer::new(buffer.clone())
            .slice(0..VERTEX_SIZE)
            .reinterpret::<[interface::Vertex]>();
        let indices = vk::Subbuffer::new(buffer)
            .slice(VERTEX_SIZE..)
            .reinterpret::<[u16]>();
        Ok(Self::new(vertices, indices))
    }
    /// Create a text renderer with the given backing mem.
    /// Consider the ownership of this mem to be taken by this renderer!
    pub fn new(vertices: Subbuffer<[interface::Vertex]>, indices: Subbuffer<[u16]>) -> Self {
        Self {
            tessellator: lyon_tessellation::FillTessellator::new(),
            cache: cache::GlyphCache::new(vertices, indices),
        }
    }
    /// Prepare to draw a single line string, tessellating if necessary.
    ///
    /// The output units are untouched and straight from the font data - it is up to the consumer to transform
    /// absolute positioning and size from instance and vertex buffers.
    // Todo: it would be AWESOME if this was async. The reciever can be recording command buffer
    // while tess into the buffer is *still happening!*
    pub fn tess_draw(
        &mut self,
        face: &rustybuzz::Face,
        plan: &rustybuzz::ShapePlan,
        size_class: SizeClass,
        text: rustybuzz::UnicodeBuffer,
        color: [f32; 4],
    ) -> Result<DrawOutput, DrawError> {
        // Currently there is no color support, and much work needs to be done
        // (in particular, current logic is totally unstable ordered)

        // We don't set any buffer properties (tbh I don't know what they do)
        // so they wont be incompatible.
        let buffer = rustybuzz::shape_with_plan(face, plan, text);

        self.tess(face.as_ref(), size_class, &buffer)?;
        // self.cache should now have every glyph needed!

        // One for every *unique* glyph. Just oversize for now lmao
        let mut glyph_instances =
            hashbrown::HashMap::<UnsizedGlyphKey, Vec<interface::Instance>>::with_capacity(
                buffer.len(),
            );

        let mut cursor = [0i32; 2];
        for (glyph, pos) in buffer
            .glyph_infos()
            .iter()
            .zip(buffer.glyph_positions().iter())
        {
            // rustybuzz guarantees that glyph_id is a u16.. strange!
            let glyph = ttf_parser::GlyphId(glyph.glyph_id.try_into().unwrap());
            let instance = interface::Instance {
                // Current cursor plus offset
                glyph_position: [
                    cursor[0].wrapping_add(pos.x_offset),
                    cursor[1].wrapping_add(pos.y_offset),
                ],
                glyph_color: color,
            };
            // advance cursor afterwards
            cursor = [
                cursor[0].wrapping_add(pos.x_advance),
                cursor[1].wrapping_add(pos.y_advance),
            ];
            // Into this glyph's entry, insert the instance info
            glyph_instances
                .entry(UnsizedGlyphKey::from_glyph_face(glyph, face.as_ref()))
                .or_default()
                .push(instance);
        }

        // Len is the number of unique glyph meshes, and thus the number of unique indirects!
        let mut indirects = Vec::with_capacity(glyph_instances.len());
        // Collect instances as we go
        let mut flat_instances = Vec::with_capacity(glyph_instances.len());
        indirects.extend(
            glyph_instances
                .into_iter()
                .filter_map(|(glyph, instances)| {
                    // We expect this to not need to be filtered due to `tess` postcondition.
                    // Failed to express this on a type level :<
                    let ([first_vertex, _], [first_index, index_count]) =
                        self.cache.get_mesh(&glyph, size_class)?;

                    let first_instance = flat_instances.len().try_into().unwrap();
                    let instance_count = instances.len().try_into().unwrap();
                    // Collect the instances into linear mem
                    flat_instances.extend_from_slice(&instances);

                    Some(vk::DrawIndexedIndirectCommand {
                        first_instance,
                        instance_count,
                        vertex_offset: first_vertex,
                        first_index,
                        index_count,
                    })
                }),
        );

        let (vertices, indices) = self.cache.buffers();
        Ok(DrawOutput {
            vertices: vertices.clone(),
            indices: indices.clone(),
            indirects,
            instances: flat_instances,
            color_mode: OutputColor::Solid(color),
            glyphs: buffer,
        })
    }
    /// From a glyph buffer, fill in the internal data cache with the tessellated glyphs
    fn tess(
        &mut self,
        face: &ttf_parser::Face,
        size_class: SizeClass,
        glyphs: &rustybuzz::GlyphBuffer,
    ) -> Result<(), DrawError> {
        // todo: bountiful threading

        // These are recycled repeatedly during tess
        let mut vertices = vec![];
        let mut indices = vec![];

        for glyph in glyphs.glyph_infos() {
            let glyph = ttf_parser::GlyphId(glyph.glyph_id.try_into().unwrap());
            let glyph_key = UnsizedGlyphKey::from_glyph_face(glyph, face);

            // Already exists, skip!
            if self.cache.get_mesh(&glyph_key, size_class).is_some() {
                continue;
            }

            let res = tessellator::tessellate_glyph(
                face,
                glyph,
                size_class,
                &mut self.tessellator,
                &mut vertices,
                &mut indices,
            );
            match res {
                Err(TessellateError::GlyphNotFound) => continue,
                Err(e) => return Err(e.into()),
                Ok(()) => (),
            };

            let res = unsafe {
                self.cache
                    // Safety: indices are guaranteed correct by tessellate_glyph postcondition!
                    .insert_mesh(glyph_key, size_class, &vertices, &indices)
            };

            match res {
                Err(CacheInsertError::AlreadyExists) => (),
                Err(e) => return Err(e.into()),
                _ => (),
            }
        }

        Ok(())
    }
}

pub mod interface {
    //! We do a bit of a funny here. We use instance buffers to transmit per-object data, even
    //! if there is only one instance for the draw call. This would usually be handled by a push-constant,
    //! but this way allows it to be dispatched all at once with a set of `VkDrawIndexedIndirect`.

    /// Font vertices for fonts without color data.
    /// Color is provided on a per-instance (char) basis
    #[derive(super::Vertex, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
    #[repr(C)]
    pub struct Vertex {
        #[format(R32G32_SFLOAT)]
        pub position: [f32; 2],
    }
    #[derive(super::Vertex, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    #[repr(C)]
    pub struct Instance {
        #[format(R32G32_SINT)]
        /// Absolute position of the character's origin point, in font points.
        pub glyph_position: [i32; 2],
        /// Per-character premultiplied linear color
        // We use a higher precision here than `vertex_colored`, in line with the rest of the color system,
        // as this is a user-provided color.
        #[format(R32G32B32A32_SFLOAT)]
        pub glyph_color: [f32; 4],
    }
}

#[derive(thiserror::Error, Debug)]
pub enum DrawError {
    #[error(transparent)]
    Tessellation(#[from] TessellateError),
    #[error(transparent)]
    InsertError(#[from] CacheInsertError),
}
pub mod renderer {
    use crate::vulkano_prelude::*;
    use std::sync::Arc;

    mod shaders {
        /// Write glyphs into monochrome MSAA buff
        pub mod monochrome {
            pub mod vert {
                vulkano_shaders::shader! {
                    ty: "vertex",
                    src: r"
                        #version 460
                        // Per-vertex
                        layout(location = 0) in vec2 position; // units unknown yet
                        // Per-instance (per-glyph)
                        layout(location = 1) in ivec2 glyph_position; // units unknown yet
                        layout(location = 2) in vec4 glyph_color;     // monochrome - dontcare!

                        void main() {
                            gl_Position = vec4((position + vec2(glyph_position)) / 12000.0 - vec2(0.8), 0.0, 1.0);
                        }
                    "
                }
            }
            pub mod frag {
                vulkano_shaders::shader! {
                    ty: "fragment",
                    src: r"
                        #version 460
                        // Rendering into R8_UNORM
                        // Cleared to zero, frags set to one.
                        layout(location = 0) out float color;
                        void main() {
                            color = 1.0;
                        }
                    "
                }
            }
        }
        /// Take a resolved coverage texture and color it.
        pub mod colorize {
            pub mod vert {
                vulkano_shaders::shader! {
                    ty: "vertex",
                    src: r"
                        #version 460

                        void main() {
                            // fullscreen tri
                            gl_Position = vec4(
                                float((gl_VertexIndex & 1) * 4) - 1.0,
                                float((gl_VertexIndex & 2) * 2) - 1.0,
                                0.0,
                                1.0
                            );
                        }
                    "
                }
            }
            pub mod frag {
                vulkano_shaders::shader! {
                    ty: "fragment",
                    src: r"
                        #version 460
                        // R8_UNORM from resolved prepass
                        layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput in_coverage;
                        layout(push_constant) uniform Push {
                            /// Premultipled color, directly multiplied by coverage.
                            vec4 modulate;
                        };

                        layout(location = 0) out vec4 color;

                        void main() {
                            color = modulate * subpassLoad(in_coverage).rrrr;
                        }
                    "
                }
            }
        }
    }

    pub struct TextRenderer {
        context: Arc<crate::render_device::RenderContext>,
        monochrome_renderpass: Arc<vk::RenderPass>,
        /// Draws the tris into MSAA buff
        monochrome_pipeline: Arc<vk::GraphicsPipeline>,
        /// MSAA buffer for antialaised rendering, immediately resolved to a regular image.
        /// (thus, can be transient attatchment)
        multisample_grey: Arc<vk::ImageView>,
        /// Resolve target for MSAA, also transient. Used as input for a coloring stage.
        resolve_grey: Arc<vk::ImageView>,
        resolve_input_set: Arc<vk::PersistentDescriptorSet>,
        /// Colors resolved greyscale image into color image
        colorize_pipeline: Arc<vk::GraphicsPipeline>,
    }
    impl TextRenderer {
        const SAMPLES: vk::SampleCount = vk::SampleCount::Sample8;
        pub fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
            let multisample_grey = vk::Image::new(
                context.allocators().memory().clone(),
                vk::ImageCreateInfo {
                    format: vk::Format::R8_UNORM,
                    samples: Self::SAMPLES,
                    // Don't need long-lived data. Render, and resolve source.
                    // DEAR FUTURE ME - the vulkano says this will err at runtime, but vulkan says otherwise.
                    usage: //vk::ImageUsage::TRANSIENT_ATTACHMENT
                         vk::ImageUsage::TRANSFER_SRC
                        | vk::ImageUsage::COLOR_ATTACHMENT,
                    // Todo: query whether this is supported
                    extent: [crate::DOCUMENT_DIMENSION, crate::DOCUMENT_DIMENSION, 1],
                    sharing: vk::Sharing::Exclusive,
                    ..Default::default()
                },
                vk::AllocationCreateInfo {
                    memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )?;
            let resolve_grey = vk::Image::new(
                context.allocators().memory().clone(),
                vk::ImageCreateInfo {
                    format: vk::Format::R8_UNORM,
                    // Don't need long-lived data. resolve destination and input.
                    usage: //vk::ImageUsage::TRANSIENT_ATTACHMENT
                         vk::ImageUsage::TRANSFER_DST
                         | vk::ImageUsage::COLOR_ATTACHMENT
                        | vk::ImageUsage::INPUT_ATTACHMENT,
                    // Todo: query whether this is supported
                    extent: [crate::DOCUMENT_DIMENSION, crate::DOCUMENT_DIMENSION, 1],
                    sharing: vk::Sharing::Exclusive,
                    ..Default::default()
                },
                vk::AllocationCreateInfo {
                    memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )?;

            // Tiling renderer shenanigans, mostly for practice lol
            // Forms a per-fragment pipe from MSAA -> Resolve -> Color
            let monochrome_renderpass = vulkano::ordered_passes_renderpass! {
                context.device().clone(),
                attachments: {
                    msaa_target: {
                        format: vk::Format::R8_UNORM,
                        samples: Self::SAMPLES,
                        load_op: Clear,
                        store_op: DontCare,
                    },
                    msaa_resolve: {
                        format: vk::Format::R8_UNORM,
                        samples: 1,
                        load_op: DontCare,
                        store_op: DontCare,
                    },
                    color_target: {
                        format: vk::Format::R16G16B16A16_SFLOAT,
                        samples: 1,
                        load_op: DontCare,
                        store_op: Store,
                    },
                },
                passes: [
                    {
                        // Render into MSAA, then resolve.
                        color: [msaa_target,],
                        color_resolve: [msaa_resolve,],
                        depth_stencil: {},
                        input: [],
                    },
                    {
                        // Take resolved image as input, color it into output.
                        color: [color_target,],
                        depth_stencil: {},
                        input: [msaa_resolve],
                    },
                ],
            }?;

            let resolve_grey = vk::ImageView::new_default(resolve_grey)?;
            let multisample_grey = vk::ImageView::new_default(multisample_grey)?;

            let mut pass = monochrome_renderpass.clone().first_subpass();
            let monochrome_pipeline =
                Self::make_monochrome_pipe(context.device().clone(), pass.clone())?;

            pass.next_subpass();
            let (colorize_pipeline, resolve_input_set) =
                Self::make_colorize_pipe(context.as_ref(), pass, resolve_grey.clone())?;

            Ok(Self {
                context,
                monochrome_renderpass,
                monochrome_pipeline,
                multisample_grey,
                resolve_grey,
                resolve_input_set,
                colorize_pipeline,
            })
        }
        /// Make a framebuffer compatible with `self.monochrome_renderpass`
        fn make_framebuffer(
            &self,
            render_into: Arc<vk::ImageView>,
        ) -> anyhow::Result<Arc<vk::Framebuffer>> {
            vk::Framebuffer::new(
                self.monochrome_renderpass.clone(),
                vk::FramebufferCreateInfo {
                    attachments: vec![
                        self.multisample_grey.clone(),
                        self.resolve_grey.clone(),
                        render_into,
                    ],
                    ..Default::default()
                },
            )
            .map_err(Into::into)
        }
        fn make_monochrome_pipe(
            device: Arc<vk::Device>,
            subpass: vk::Subpass,
        ) -> anyhow::Result<Arc<vk::GraphicsPipeline>> {
            let vert = shaders::monochrome::vert::load(device.clone())?;
            let vert = vert.entry_point("main").unwrap();
            let frag = shaders::monochrome::frag::load(device.clone())?;
            let frag = frag.entry_point("main").unwrap();

            let vertex_description = [
                super::interface::Vertex::per_vertex(),
                super::interface::Instance::per_instance(),
            ]
            .definition(&vert.info().input_interface)?;

            let stages = smallvec::smallvec![
                vk::PipelineShaderStageCreateInfo::new(vert),
                vk::PipelineShaderStageCreateInfo::new(frag),
            ];

            // Nothin to describe!
            // Will prolly have a push constant in the future.
            let layout =
                vk::PipelineLayout::new(device.clone(), vk::PipelineLayoutCreateInfo::default())?;

            vk::GraphicsPipeline::new(
                device,
                None,
                vk::GraphicsPipelineCreateInfo {
                    stages,
                    vertex_input_state: Some(vertex_description),
                    input_assembly_state: Some(vk::InputAssemblyState::default()),
                    color_blend_state: Some(vk::ColorBlendState::with_attachment_states(
                        1,
                        vk::ColorBlendAttachmentState::default(),
                    )),
                    rasterization_state: Some(vk::RasterizationState::default()),
                    subpass: Some(subpass.into()),
                    multisample_state: Some(vk::MultisampleState {
                        rasterization_samples: Self::SAMPLES,
                        ..Default::default()
                    }),
                    // Viewport dynamic, scissor irrelevant.
                    viewport_state: Some(vk::ViewportState::default()),
                    dynamic_state: [vk::DynamicState::Viewport].into_iter().collect(),
                    ..vk::GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .map_err(Into::into)
        }
        fn make_colorize_pipe(
            context: &crate::render_device::RenderContext,
            subpass: vk::Subpass,
            resolve_input_image: Arc<vk::ImageView>,
        ) -> anyhow::Result<(Arc<vk::GraphicsPipeline>, Arc<vk::PersistentDescriptorSet>)> {
            use vulkano::descriptor_set::DescriptorSet;
            let device = context.device();
            let vert = shaders::colorize::vert::load(device.clone())?;
            let vert = vert.entry_point("main").unwrap();
            let frag = shaders::colorize::frag::load(device.clone())?;
            let frag = frag.entry_point("main").unwrap();

            // No input.
            let vertex_description = vk::VertexInputState::new();

            let stages = smallvec::smallvec![
                vk::PipelineShaderStageCreateInfo::new(vert),
                vk::PipelineShaderStageCreateInfo::new(frag),
            ];

            let color_range = vk::PushConstantRange {
                offset: 0,
                size: std::mem::size_of::<shaders::colorize::frag::Push>()
                    .try_into()
                    .unwrap(),
                stages: vk::ShaderStages::FRAGMENT,
            };

            let resolve_input_set = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                vk::DescriptorSetLayout::new(
                    device.clone(),
                    vk::DescriptorSetLayoutCreateInfo {
                        bindings: [(
                            0,
                            vk::DescriptorSetLayoutBinding {
                                stages: vk::ShaderStages::FRAGMENT,
                                ..vk::DescriptorSetLayoutBinding::descriptor_type(
                                    vk::DescriptorType::InputAttachment,
                                )
                            },
                        )]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    },
                )?,
                [vk::WriteDescriptorSet::image_view(0, resolve_input_image)],
                [],
            )?;

            let layout = vk::PipelineLayout::new(
                device.clone(),
                vk::PipelineLayoutCreateInfo {
                    set_layouts: vec![resolve_input_set.layout().clone()],
                    push_constant_ranges: vec![color_range],
                    ..Default::default()
                },
            )?;

            Ok((
                vk::GraphicsPipeline::new(
                    device.clone(),
                    None,
                    vk::GraphicsPipelineCreateInfo {
                        stages,
                        vertex_input_state: Some(vertex_description),
                        input_assembly_state: Some(vk::InputAssemblyState::default()),
                        color_blend_state: Some(vk::ColorBlendState::with_attachment_states(
                            1,
                            vk::ColorBlendAttachmentState::default(),
                        )),
                        rasterization_state: Some(vk::RasterizationState::default()),
                        multisample_state: Some(vk::MultisampleState::default()),
                        subpass: Some(subpass.into()),
                        // Viewport dynamic, scissor irrelevant.
                        viewport_state: Some(vk::ViewportState::default()),
                        dynamic_state: [vk::DynamicState::Viewport].into_iter().collect(),
                        ..vk::GraphicsPipelineCreateInfo::layout(layout)
                    },
                )?,
                resolve_input_set,
            ))
        }
        // Upload instances and indirects. Ok(None) if either is empty.
        fn predraw_upload(
            &self,
            output: &super::DrawOutput,
        ) -> anyhow::Result<
            Option<(
                vk::Subbuffer<[super::interface::Instance]>,
                vk::Subbuffer<[vk::DrawIndexedIndirectCommand]>,
            )>,
        > {
            if output.instances.is_empty() || output.indirects.is_empty() {
                Ok(None)
            } else {
                let instances_len_bytes = std::mem::size_of_val(output.instances.as_slice()) as u64;
                let indirects_len_bytes = std::mem::size_of_val(output.indirects.as_slice()) as u64;
                let scratch_size = instances_len_bytes + indirects_len_bytes;

                let align = std::mem::align_of_val(output.instances.as_slice());
                // make sure align requirements are sound between the two bufs
                assert!(align >= std::mem::align_of_val(output.indirects.as_slice()));

                let scratch_buffer = vk::Buffer::new(
                    self.context.allocators().memory().clone(),
                    vk::BufferCreateInfo {
                        sharing: vulkano::sync::Sharing::Exclusive,
                        // We make a host accessible instance buffer... this kinda sucks!
                        usage: vk::BufferUsage::INDIRECT_BUFFER | vk::BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    vk::AllocationCreateInfo {
                        memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                            | vk::MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                    vulkano::memory::allocator::DeviceLayout::new(
                        scratch_size
                            .try_into()
                            // Guarded by if, not empty!
                            .unwrap(),
                        // Std guarantees power of two
                        vulkano::memory::DeviceAlignment::new(align as u64).unwrap(),
                        // Unwrap ok - it fits in host mem just fine (<= device address space), the size won't overflow.
                    )
                    .unwrap(),
                )?;
                // Slices ok - both known non-zero by if guard
                let instance_buffer = vk::Subbuffer::new(scratch_buffer.clone())
                    .slice(0..instances_len_bytes)
                    .reinterpret::<[super::interface::Instance]>();
                {
                    // We just made it - no concurrent access.
                    let mut write = instance_buffer.write().unwrap();
                    write.copy_from_slice(&output.instances);
                }
                let indirect_buffer = vk::Subbuffer::new(scratch_buffer)
                    // Align is OK - we ickily checked that the aligns of the two types are compatible
                    .slice(instances_len_bytes..)
                    .reinterpret::<[vk::DrawIndexedIndirectCommand]>();
                {
                    // We just made it - no concurrent access.
                    let mut write = indirect_buffer.write().unwrap();
                    write.copy_from_slice(&output.indirects);
                }

                Ok(Some((instance_buffer, indirect_buffer)))
            }
        }
        /// Create commands to render `DrawOutput` into the given image.
        ///
        /// `self` is in exclusive use through the duration of the returned buffer's execution.
        pub fn draw(
            &self,
            render_into: Arc<vk::ImageView>,
            output: &super::DrawOutput,
        ) -> anyhow::Result<Arc<vk::PrimaryAutoCommandBuffer>> {
            // Even if none to draw, we can still clear and return.
            let instances_indirects = self.predraw_upload(output)?;

            let mut commands = vk::AutoCommandBufferBuilder::primary(
                self.context.allocators().command_buffer(),
                self.context.queues().graphics().idx(),
                vk::CommandBufferUsage::OneTimeSubmit,
            )?;

            if let Some((instances, indirects)) = instances_indirects {
                let framebuffer = self.make_framebuffer(render_into)?;
                let viewport = vk::Viewport {
                    depth_range: 0.0..=1.0,
                    offset: [0.0; 2],
                    extent: [
                        framebuffer.extent()[0] as f32,
                        framebuffer.extent()[1] as f32,
                    ],
                };

                commands
                    .begin_render_pass(
                        vk::RenderPassBeginInfo {
                            // Start is cleared, other two are dontcare (immediately overwritten)
                            clear_values: vec![Some([0.0; 4].into()), None, None],
                            ..vk::RenderPassBeginInfo::framebuffer(framebuffer)
                        },
                        vk::SubpassBeginInfo {
                            contents: vk::SubpassContents::Inline,
                            ..Default::default()
                        },
                    )?
                    .bind_pipeline_graphics(self.monochrome_pipeline.clone())?
                    .set_viewport(0, smallvec::smallvec![viewport.clone()])?
                    .bind_vertex_buffers(0, (output.vertices.clone(), instances))?
                    .bind_index_buffer(output.indices.clone())?
                    .draw_indexed_indirect(indirects)?
                    .next_subpass(
                        vk::SubpassEndInfo::default(),
                        vk::SubpassBeginInfo {
                            contents: vk::SubpassContents::Inline,
                            ..Default::default()
                        },
                    )?
                    .bind_pipeline_graphics(self.colorize_pipeline.clone())?
                    .set_viewport(0, smallvec::smallvec![viewport.clone()])?
                    .bind_descriptor_sets(
                        vk::PipelineBindPoint::Graphics,
                        self.colorize_pipeline.layout().clone(),
                        0,
                        self.resolve_input_set.clone(),
                    )?
                    .push_constants(
                        self.colorize_pipeline.layout().clone(),
                        0,
                        shaders::colorize::frag::Push {
                            modulate: [0.0, 0.0, 0.0, 1.0],
                        },
                    )?
                    .draw(3, 1, 0, 0)?
                    .end_render_pass(vk::SubpassEndInfo::default())?;

                commands.build().map_err(Into::into)
            } else {
                // Nothing to draw. Just clear the output image for consistent behavior
                let clear = vk::ClearColorImageInfo {
                    regions: smallvec::smallvec![render_into.subresource_range().clone()],
                    ..vk::ClearColorImageInfo::image(render_into.image().clone())
                };

                commands.clear_color_image(clear)?;

                commands.build().map_err(Into::into)
            }
        }
    }
}
