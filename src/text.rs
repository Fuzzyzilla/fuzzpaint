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

use egui::TextBuffer;
use rustybuzz::ttf_parser;
use vulkano::buffer::Subbuffer;

use crate::vulkano_prelude::*;
mod cache;
pub use cache::{CacheInsertError, ColorInsertError};
mod tessellator;
pub use tessellator::{ColorError, TessellateError};

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
    /// Convert from an exponent, where scale factor is `2**exp`.
    /// Clamps to valid range.
    #[must_use]
    pub fn from_exp_lossy(exp: i8) -> Self {
        // Clamp out invalid exponents
        let exp = exp.clamp(Self::MIN.0, Self::MAX.0);
        Self(exp)
    }
    /// Take the `log2` of the size factor. This is a precise operation.
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
    /// Multiply two scale classes, saturating to `MIN` or `MAX` should this operation surpass it.
    ///
    /// ```
    /// let two = ScaleClass::from_scale_factor(2.0).unwrap();
    /// assert_eq!(
    ///     two.saturating_mul(two),
    ///     ScaleClass::from_scale_factor(4.0).unwrap()
    /// );
    /// ```
    #[must_use = "does not modify self"]
    pub fn saturating_mul(self, other: Self) -> Self {
        // logarithmic representation, addition is scale factor mul!
        let new_exp = self.0.saturating_add(other.0);
        Self::from_exp_lossy(new_exp)
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
#[derive(PartialEq, Clone, Copy)]
pub enum OutputColor {
    /// Color varies between glyphs.
    PerInstance,
    /// Every glyph uses the same color.
    Solid([f32; 4]),
}

/// Data needed to render tessellated, shaped, and colored glyphs.
#[derive(Clone)]
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

    /// The bounding box around the rendered content, in the same units as vertex and instance position.
    /// (min, max)
    pub bound: Option<([i32; 2], [i32; 2])>,
    /// Primary direction of the contained text, or None if inconsistent.
    ///
    /// (i.e., multiple different lines with different primary directionality are merged)
    pub main_direction: Option<rustybuzz::Direction>,
    /// Cross-axis direction, where newlines take the cursor. E.g. in latin script this is top-to-bottom.
    pub cross_direction: Option<rustybuzz::Direction>,
}

#[derive(thiserror::Error, Debug)]
pub enum DrawAppendError {
    #[error("too many instances")]
    /// Vulkan's indirects specify u32 for the instance offset, cannot exceed this.
    TooLong,
    /// Draws come from different vertex/index caches/builders.
    #[error("sources differ")]
    SourceMismatch,
}
#[derive(thiserror::Error, Debug)]
pub enum DrawError {
    #[error(transparent)]
    Tessellation(#[from] TessellateError),
    #[error(transparent)]
    InsertError(#[from] CacheInsertError),
}
#[derive(thiserror::Error, Debug)]
pub enum DrawMultilineError {
    #[error(transparent)]
    Draw(#[from] DrawError),
    #[error(transparent)]
    Append(#[from] DrawAppendError),
}
impl DrawOutput {
    /// Translate render data by the given amount. This is not a free operation, but
    /// should be fairly cheap. font units.
    ///
    /// `self::glyphs` is not affected.
    pub fn translate(&mut self, by: [i32; 2]) {
        // Nothing to do.
        if by == [0, 0] {
            return;
        }
        for instance in &mut self.instances {
            instance.glyph_position[0] = instance.glyph_position[0].saturating_add(by[0]);
            instance.glyph_position[1] = instance.glyph_position[1].saturating_add(by[1]);
        }
        if let Some(bound) = &mut self.bound {
            // No other arithmetic behavior here would be correct :<
            // Good news is that this remains a correct bound even if saturation occurs.
            bound.0[0] = bound.0[0].saturating_add(by[0]);
            bound.1[0] = bound.1[0].saturating_add(by[0]);

            bound.0[1] = bound.0[1].saturating_add(by[1]);
            bound.1[1] = bound.1[1].saturating_add(by[1]);
        }
    }
    /// Calculate the anchor of a given direction and align.
    /// E.g. Left-to-right + start will give the left edge,
    ///
    /// None if the direction is `Invalid` or `self.bound.is_none()`
    pub fn anchor(&self, direction: rustybuzz::Direction, align: Align) -> Option<i32> {
        let (min, max) = self.bound?;
        let (start, end) = match direction {
            rustybuzz::Direction::LeftToRight => (min[0], max[0]),
            rustybuzz::Direction::RightToLeft => (max[0], min[0]),
            rustybuzz::Direction::TopToBottom => (min[1], max[1]),
            rustybuzz::Direction::BottomToTop => (max[1], min[1]),
            rustybuzz::Direction::Invalid => return None,
        };
        Some(match align {
            Align::Start => start,
            Align::End => end,
            // midpoint
            // for each bit: One if both set, else half if either set.
            Align::Center => (start & end) + ((start ^ end) >> 1),
        })
    }
    pub fn align_to(&mut self, origin: [i32; 2], main: Option<Align>, cross: Option<Align>) {
        let mut x_offs = None;
        let mut y_offs = None;
        // ======== Main
        if let Some((align, direction)) = main.zip(self.main_direction) {
            if let Some(anchor) = self.anchor(direction, align) {
                match direction {
                    rustybuzz::Direction::LeftToRight | rustybuzz::Direction::RightToLeft => {
                        // x-axis
                        x_offs = Some(origin[0] - anchor);
                    }
                    rustybuzz::Direction::TopToBottom | rustybuzz::Direction::BottomToTop => {
                        // y-axis
                        y_offs = Some(origin[1] - anchor);
                    }
                    // self.anchor would have returned none.
                    rustybuzz::Direction::Invalid => unreachable!(),
                }
            }
        }
        // ======== Cross
        if let Some((align, direction)) = cross.zip(self.cross_direction) {
            if let Some(anchor) = self.anchor(direction, align) {
                match direction {
                    rustybuzz::Direction::LeftToRight | rustybuzz::Direction::RightToLeft => {
                        // x-axis
                        x_offs = Some(origin[0] - anchor);
                    }
                    rustybuzz::Direction::TopToBottom | rustybuzz::Direction::BottomToTop => {
                        // y-axis
                        y_offs = Some(origin[1] - anchor);
                    }
                    // self.anchor would have returned none.
                    rustybuzz::Direction::Invalid => unreachable!(),
                }
            }
        }
        let offs = [x_offs.unwrap_or(0), y_offs.unwrap_or(0)];
        self.translate(offs);
    }
    /// Tight extent of the rendered output, in font units.
    pub fn extent(&self) -> [u32; 2] {
        if let Some((min, max)) = self.bound {
            // We assume max >= min.
            [min[0].abs_diff(max[0]), min[1].abs_diff(max[1])]
        } else {
            // None = no output = zero size.
            [0; 2]
        }
    }
    /// Combine text rendering operations. `other` is rendered *after* self (this could change!)
    ///
    /// `self` is not modified if an error is returned.
    pub fn append(&mut self, other: &Self) -> Result<(), DrawAppendError> {
        // ================= Buffers ==================
        {
            use vulkano::VulkanObject;
            if (self.indices.buffer().handle() != other.indices.buffer().handle())
                || (self.vertices.buffer().handle() != other.vertices.buffer().handle())
            {
                return Err(DrawAppendError::SourceMismatch);
            }
        }

        // ================= Instances ==================
        // Vulkan only allows u32 as indexes into the instance array, ensure we don't exceed that.
        // (this is over-strict, only the base-instance has this restriction, base+count is allowed to exceed)
        // (not super concerned about the 2-billion-instance edgecase, tho :P )

        if !self
            .instances
            .len()
            .checked_add(other.instances.len())
            // Didn't overflow and is a valid u32
            // (inverted to overflow or invalid)
            .is_some_and(|total| u32::try_from(total).is_ok())
        {
            return Err(DrawAppendError::TooLong);
        }
        // TODO: merge instead of append.
        // (insert like draws next to each other and increase indirect instance cound instead of more indirects)
        // Unwrap fine, Checked at start.
        let base_instance_shift = u32::try_from(self.instances.len()).unwrap();
        self.instances.extend_from_slice(&other.instances);

        let base_indirect = self.indirects.len();
        self.indirects.extend_from_slice(&other.indirects);
        // For each new indirect we just added...
        for other_indirect in &mut self.indirects[base_indirect..] {
            // We put these instances on the end, but they used to be zero-indexed.
            // Shift that base up!
            // Won't overflow - checked at the start.
            other_indirect.first_instance += base_instance_shift;
        }

        // ================= Colors ==================
        self.color_mode = if self.color_mode == other.color_mode {
            self.color_mode
        } else {
            // Different modes, fallback on instance data (always correct)
            OutputColor::PerInstance
        };

        // ================= Bounds ==================
        self.bound = match (self.bound, other.bound) {
            // None is 'empty bound'. Thus the union is whichever is set.
            (None, None) => None,
            (Some(b), None) | (None, Some(b)) => Some(b),
            // But if both set, find the AABB union of the two.
            (Some((a_min, a_max)), Some((b_min, b_max))) => Some((
                [a_min[0].min(b_min[0]), a_min[1].min(b_min[1])],
                [a_min[1].max(b_max[1]), a_max[1].max(b_max[1])],
            )),
        };

        // ================ Direction =================
        self.main_direction = if self.main_direction == other.main_direction {
            self.main_direction
        } else {
            // Mixed primary directionality, weh
            None
        };
        self.cross_direction = if self.cross_direction == other.cross_direction {
            self.cross_direction
        } else {
            None
        };

        Ok(())
    }
}
pub struct FullOutput {
    /// Rendering info
    pub draw: DrawOutput,

    /// Finished glyph buffer. Primarily for mem reclaimation.
    pub glyphs: rustybuzz::GlyphBuffer,
}

/// Direction/Axis agnostic Align.
#[derive(Clone, Copy)]
pub enum Align {
    Start,
    Center,
    End,
    // Justify is a whole task. no thanks! x3
}

/// Wayyy too many args for a fn!
pub struct MultilineInfo<'a> {
    /// Text containing zero or more lines, delimited by `\n` or `\r\n` with optional newline at the end.
    pub text: &'a str,
    /// Language the string is in, None to guess. Must match the plan.
    pub language: Option<rustybuzz::Language>,
    /// Script the string is in, None to guess. Must match the plan.
    pub script: Option<rustybuzz::Script>,
    /// Main text direction. Must match the plan.
    pub main_direction: rustybuzz::Direction,
    /// Multiplier by the font's line spacing
    pub line_spacing_mul: f32,
    /// Main-axis align. Eg. in latin script, this is horizontal align with `Start` = left.
    ///
    /// Cross-axis is not yet implemented.
    // Fixme, this is Flexbox language, not typographic language!
    pub main_align: Align,
    /// Order each new line goes in relation to the previous, must be perendicular to `main_direction`.
    pub cross_direction: rustybuzz::Direction,
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
    pub fn tess_draw_multiline(
        &mut self,
        face: &rustybuzz::Face,
        plan: &rustybuzz::ShapePlan,
        size_class: SizeClass,
        info: &MultilineInfo,
        color: [f32; 4],
    ) -> Result<DrawOutput, DrawMultilineError> {
        // This is a sealed implementation detail of rustybuzz that I snuck a peak at...
        const CONTEXT_CHARS: usize = 5;
        type Context = [char; CONTEXT_CHARS];
        // Every unicode scalar is at most 4 utf8 bytes,
        const CONTEXT_BYTES: usize = CONTEXT_CHARS * 4;
        // In which I greatly overcomplicate it in the name of not having to allocate `Strings`:
        fn make_context_str<'bytes>(
            context: &[char],
            context_bytes: &'bytes mut [u8; CONTEXT_BYTES],
        ) -> &'bytes str {
            let context = if context.len() > CONTEXT_CHARS {
                &context[context.len() - CONTEXT_CHARS..]
            } else {
                context
            };
            let mut cursor = 0;
            for c in context {
                let c_str = c.encode_utf8(&mut context_bytes[cursor..]);
                cursor += c_str.len();
            }
            // Safety: we just went char-by-char encoding into valid utf-8!
            debug_assert!(std::str::from_utf8(&context_bytes[..cursor]).is_ok());
            unsafe { std::str::from_utf8_unchecked(&context_bytes[..cursor]) }
        }

        const BREAK_CHAR: char = ' ';

        // This seems like the wrong property to use, but line height is always zero!
        // future self, this is called "Leading"
        let line_height = i32::from(face.height());
        let mut vcursor = 0i32;

        let mut lines = info.text.lines();

        let mut prev = None::<&str>;
        // If no text, branch with Ok(empty)
        let Some(mut cur) = lines.next() else {
            return Ok(self.empty_draw());
        };
        // Purposefully none-able
        let mut next = lines.next();

        let mut outer_buffer = Some(rustybuzz::UnicodeBuffer::new());
        let mut output = None::<DrawOutput>;

        loop {
            // Shape `cur`, using `prev` and `next` to inform context, if any.
            // Append render data to collection
            // while lines.next() is some, line -> next -> cur -> prev.

            // Always some, put back at end of loop.
            let mut buffer = outer_buffer.take().unwrap();
            // *meticulously avoids string copies only to have to do it every loop lmao*
            if let Some(lang) = info.language.clone() {
                buffer.set_language(lang);
            }
            if let Some(script) = info.script {
                buffer.set_script(script);
            }
            buffer.set_direction(info.main_direction);

            // ==============================  Set str, pre-post context.
            // (empty by default due to clear at the end)
            buffer.push_str(cur);
            let mut flags = rustybuzz::BufferFlags::empty();
            if let Some(prev) = prev {
                // Take *last* 4 chars of the last line
                // Wont ever allocate.
                let mut last_chars: smallvec::SmallVec<Context> =
                    prev.chars().rev().take(CONTEXT_CHARS - 1).collect();
                last_chars.reverse();
                // Put a space on the end, to hint to shaper that there is a break here.
                last_chars.push(BREAK_CHAR);
                buffer.set_pre_context(make_context_str(&last_chars, &mut [0u8; CONTEXT_BYTES]));
            } else {
                flags |= rustybuzz::BufferFlags::BEGINNING_OF_TEXT;
            }
            if let Some(next) = next {
                // Take *first* 4 chars of the next line
                // Wont ever allocate.
                let mut first_chars: smallvec::SmallVec<Context> =
                    next.chars().take(CONTEXT_CHARS - 1).collect();
                // Put a space on the front, to hint to shaper that there is a break here.
                first_chars.insert(0, BREAK_CHAR);
                buffer.set_post_context(make_context_str(&first_chars, &mut [0u8; CONTEXT_BYTES]));
            } else {
                flags |= rustybuzz::BufferFlags::END_OF_TEXT;
            }
            buffer.set_flags(flags);

            // =============================== Shape and tess
            let mut new_output = self.tess_draw(face, plan, size_class, buffer, color)?;
            if vcursor != 0 {
                new_output.draw.translate([0, vcursor]);
            }
            vcursor -= line_height;
            new_output
                .draw
                .align_to([0; 2], Some(info.main_align), None);
            match output.as_mut() {
                None => output = Some(new_output.draw),
                Some(output) => output.append(&new_output.draw)?,
            }

            // ================================= Continue
            outer_buffer = Some(new_output.glyphs.clear());
            if let Some(n) = next {
                // More to do, shift down.
                prev = Some(cur);
                cur = n;
                // Could be none - breaks out at end of next pass.
                next = lines.next();
            } else {
                // Ran out of text!
                break;
            }
        }

        // Todo: should be FullOutput.
        // Shouldn't ever be none.
        Ok(output.unwrap_or_else(|| self.empty_draw()))
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
    ) -> Result<FullOutput, DrawError> {
        // Currently there is no color support, and much work needs to be done
        // (in particular, current logic is totally unstable ordered)

        // We don't set any buffer properties (tbh I don't know what they do)
        // so they wont be incompatible.
        let main_direction = match text.direction() {
            rustybuzz::Direction::Invalid => None,
            dir => Some(dir),
        };
        let buffer = rustybuzz::shape_with_plan(face, plan, text);

        self.tess(face.as_ref(), size_class, &buffer)?;
        // self.cache should now have every glyph needed!

        // One for every *unique* glyph. Just oversize for now lmao
        let mut glyph_instances =
            hashbrown::HashMap::<UnsizedGlyphKey, Vec<interface::Instance>>::with_capacity(
                buffer.len(),
            );

        let mut cursor = [0i32; 2];
        let mut render_bound = ([i32::MAX; 2], [i32::MIN; 2]);
        for (glyph, pos) in buffer
            .glyph_infos()
            .iter()
            .zip(buffer.glyph_positions().iter())
        {
            // rustybuzz guarantees that glyph_id is a u16 repr as u32.. strange!
            let glyph = ttf_parser::GlyphId(glyph.glyph_id.try_into().unwrap());
            let glyph_position = [
                cursor[0].wrapping_add(pos.x_offset),
                cursor[1].wrapping_add(pos.y_offset),
            ];
            // todo: turns out bounding_box is massively expensive and we should cache this during tess.
            if let Some(bound) = face.glyph_bounding_box(glyph) {
                // copy+paste errors abound
                // Offset box by position
                let x_min = i32::from(bound.x_min).saturating_add(glyph_position[0]);
                let x_max = i32::from(bound.x_max).saturating_add(glyph_position[0]);

                let y_min = i32::from(bound.y_min).saturating_add(glyph_position[1]);
                let y_max = i32::from(bound.y_max).saturating_add(glyph_position[1]);

                // Expand as necessary
                // horrible horrible!!
                render_bound.0[0] = render_bound.0[0].min(x_min);
                render_bound.1[0] = render_bound.1[0].max(x_max);
                render_bound.0[1] = render_bound.0[1].min(y_min);
                render_bound.1[1] = render_bound.1[1].max(y_max);
            }
            let instance = interface::Instance {
                // Current cursor plus offset
                glyph_position,
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
        // Only some if valid box. This fails if no glyphs were rendered.
        let render_bound = (render_bound.0[0] <= render_bound.1[0]
            && render_bound.0[1] <= render_bound.1[1])
            .then_some(render_bound);

        // Len is the number of unique glyph meshes, and thus the number of unique indirects!
        let mut indirects = Vec::with_capacity(glyph_instances.len());
        // Collect instances as we go
        let mut flat_instances = Vec::with_capacity(glyph_instances.len());
        indirects.extend(
            glyph_instances
                .into_iter()
                .filter_map(|(glyph, instances)| {
                    // Not every glyph have a mesh, and this is ok - spaces, for example.
                    // incorrect chars are explicitly replaced with the "tofu" glyph by rustybuzz.
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
        Ok(FullOutput {
            draw: DrawOutput {
                vertices: vertices.clone(),
                indices: indices.clone(),
                indirects,
                instances: flat_instances,
                bound: render_bound,
                color_mode: OutputColor::Solid(color),
                main_direction,
                // No data for this in a single-line context.
                cross_direction: None,
            },
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
    fn empty_draw(&self) -> DrawOutput {
        let (vertices, indices) = self.cache.buffers();
        DrawOutput {
            vertices: vertices.clone(),
            indices: indices.clone(),
            instances: vec![],
            indirects: vec![],
            color_mode: OutputColor::Solid([1.0; 4]),
            bound: None,
            cross_direction: None,
            main_direction: None,
        }
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
                        layout(push_constant) uniform Matrix {
                            mat4 mvp;
                        };
                        // Per-vertex
                        layout(location = 0) in vec2 position;        // arbitrary font units (not em)
                        // Per-instance (per-glyph)
                        layout(location = 1) in ivec2 glyph_position; // arbitrary font units (not em)
                        layout(location = 2) in vec4 glyph_color;     // monochrome pipeline - dontcare!

                        void main() {
                            gl_Position = mvp * vec4(position + vec2(glyph_position), 0.0, 1.0);
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
                        layout(push_constant) uniform Color {
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
        monochrome_samples: vk::SampleCount,
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
        /// Get the internal scale factor due to multisampling.
        /// Should be multiplied by the tessellation factor prior to building for best looking results.
        pub fn internal_size_class(&self) -> super::SizeClass {
            use vk::SampleCount;
            // Spacial resolution multiplier is sqrt(sample count).
            // To convert to SizeClass, take the log2 of this, ceil.
            let log2_sqrt_samples = match self.monochrome_samples {
                SampleCount::Sample1 => 0,
                SampleCount::Sample2 | SampleCount::Sample4 => 1,
                SampleCount::Sample8 | SampleCount::Sample16 => 2,
                SampleCount::Sample32 | SampleCount::Sample64 => 3,
                // We should never be able to choose a sample count we can't even observe!
                _ => unimplemented!("unknown sample count"),
            };

            super::SizeClass::from_exp_lossy(log2_sqrt_samples)
        }
        pub fn make_renderpass(
            device: Arc<vk::Device>,
            sample_count: vk::SampleCount,
        ) -> anyhow::Result<Arc<vk::RenderPass>> {
            use vulkano::render_pass::{
                AttachmentDescription, AttachmentReference, RenderPass, RenderPassCreateInfo,
                SubpassDependency, SubpassDescription,
            };
            use vulkano::sync::{AccessFlags, DependencyFlags, PipelineStages};

            // Vulkano doesn't support this usecase of transient images, so we do it ourselves >:3c
            let attachments = vec![
                // First render attach. Transient. Clear, dontcare for store.
                AttachmentDescription {
                    load_op: vk::AttachmentLoadOp::Clear,
                    store_op: vk::AttachmentStoreOp::DontCare,
                    format: vk::Format::R8_UNORM,
                    samples: sample_count,
                    initial_layout: vk::ImageLayout::Undefined,
                    final_layout: vk::ImageLayout::ColorAttachmentOptimal,
                    ..Default::default()
                },
                // Second attach, for resolving first. Transient input for final stage. dontcare for load/store.
                AttachmentDescription {
                    load_op: vk::AttachmentLoadOp::DontCare,
                    store_op: vk::AttachmentStoreOp::DontCare,
                    format: vk::Format::R8_UNORM,
                    samples: vk::SampleCount::Sample1,
                    initial_layout: vk::ImageLayout::Undefined,
                    final_layout: vk::ImageLayout::ColorAttachmentOptimal,
                    ..Default::default()
                },
                // Output color, user provided.
                AttachmentDescription {
                    // We overwrite every texel, dontcare for load.
                    load_op: vk::AttachmentLoadOp::DontCare,
                    store_op: vk::AttachmentStoreOp::Store,
                    format: vk::Format::R16G16B16A16_SFLOAT,
                    samples: vk::SampleCount::Sample1,
                    // We clear it anyway, don't mind the initial layout.
                    initial_layout: vk::ImageLayout::Undefined,
                    final_layout: vk::ImageLayout::ColorAttachmentOptimal,
                    ..Default::default()
                },
            ];
            let attachment_references = vec![
                AttachmentReference {
                    attachment: 0,
                    layout: vk::ImageLayout::ColorAttachmentOptimal,
                    stencil_layout: None,
                    ..Default::default()
                },
                // Pre-resolve:
                AttachmentReference {
                    attachment: 1,
                    layout: vk::ImageLayout::ColorAttachmentOptimal,
                    stencil_layout: None,
                    ..Default::default()
                },
                // Post-resolve:
                AttachmentReference {
                    attachment: 1,
                    layout: vk::ImageLayout::ShaderReadOnlyOptimal,
                    stencil_layout: None,
                    aspects: vk::ImageAspects::COLOR,
                    ..Default::default()
                },
                AttachmentReference {
                    attachment: 2,
                    layout: vk::ImageLayout::ColorAttachmentOptimal,
                    stencil_layout: None,
                    ..Default::default()
                },
            ];

            let subpasses = vec![
                SubpassDescription {
                    color_attachments: vec![Some(attachment_references[0].clone())],
                    // Afterwards, resolve into other transient attachment.
                    color_resolve_attachments: vec![Some(attachment_references[1].clone())],
                    ..Default::default()
                },
                SubpassDescription {
                    color_attachments: vec![Some(attachment_references[3].clone())],
                    input_attachments: vec![Some(attachment_references[2].clone())],
                    ..Default::default()
                },
            ];
            let dependencies = vec![SubpassDependency {
                src_subpass: Some(0),
                dst_subpass: Some(1),
                // Second subpass only cares about pixel-local info, so individual tiles can move on asynchronously.
                dependency_flags: DependencyFlags::BY_REGION,
                // After we resolve...
                src_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                // Then we can read from fragment...
                dst_access: AccessFlags::INPUT_ATTACHMENT_READ,
                dst_stages: PipelineStages::FRAGMENT_SHADER,
                ..Default::default()
            }];

            RenderPass::new(
                device,
                RenderPassCreateInfo {
                    attachments,
                    subpasses,
                    dependencies,
                    ..Default::default()
                },
            )
            .map_err(Into::into)
        }
        pub fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
            let monochrome_samples = context
                .physical_device()
                .image_format_properties(vulkano::image::ImageFormatInfo {
                    format: vk::Format::R8_UNORM,
                    tiling: vulkano::image::ImageTiling::Optimal,
                    usage: vk::ImageUsage::TRANSIENT_ATTACHMENT | vk::ImageUsage::COLOR_ATTACHMENT,
                    ..Default::default()
                })?
                .ok_or_else(|| anyhow::anyhow!("unsupported image configuration"))?
                .sample_counts
                .max_count();
            if monochrome_samples == vk::SampleCount::Sample1 {
                // The failure path here involves a whole different pipeline :V
                anyhow::bail!("msaa unsupported")
            }
            let multisample_grey = vk::Image::new(
                context.allocators().memory().clone(),
                vk::ImageCreateInfo {
                    format: vk::Format::R8_UNORM,
                    samples: monochrome_samples,
                    // Don't need long-lived data. Render, and resolve source.
                    usage: vk::ImageUsage::TRANSIENT_ATTACHMENT | vk::ImageUsage::COLOR_ATTACHMENT,
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
                    usage: vk::ImageUsage::TRANSIENT_ATTACHMENT
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
            let monochrome_renderpass =
                Self::make_renderpass(context.device().clone(), monochrome_samples)?;

            let resolve_grey = vk::ImageView::new_default(resolve_grey)?;
            let multisample_grey = vk::ImageView::new_default(multisample_grey)?;

            let mut pass = monochrome_renderpass.clone().first_subpass();
            let monochrome_pipeline = Self::make_monochrome_pipe(
                context.device().clone(),
                pass.clone(),
                monochrome_samples,
            )?;

            pass.next_subpass();
            let (colorize_pipeline, resolve_input_set) =
                Self::make_colorize_pipe(context.as_ref(), pass, resolve_grey.clone())?;

            Ok(Self {
                monochrome_samples,
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
            samples: vk::SampleCount,
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

            let matrix_range = vk::PushConstantRange {
                offset: 0,
                size: std::mem::size_of::<shaders::monochrome::vert::Matrix>()
                    .try_into()
                    .unwrap(),
                stages: vk::ShaderStages::VERTEX,
            };

            let layout = vk::PipelineLayout::new(
                device.clone(),
                vk::PipelineLayoutCreateInfo {
                    push_constant_ranges: vec![matrix_range],
                    ..Default::default()
                },
            )?;

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
                        rasterization_samples: samples,
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
                size: std::mem::size_of::<shaders::colorize::frag::Color>()
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
        /// `xform` should be a combined view+proj matrix, tranforming face's arbitrary units (NOT em) to NDC.
        /// See [`rustybuzz::ttf_parser::Face::units_per_em`].
        ///
        /// `self` is in exclusive use through the duration of the returned buffer's execution.
        pub fn draw(
            &self,
            xform: ultraviolet::Mat4,
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
                    .push_constants(
                        self.monochrome_pipeline.layout().clone(),
                        0,
                        shaders::monochrome::vert::Matrix { mvp: xform.into() },
                    )?
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
                        shaders::colorize::frag::Color {
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
