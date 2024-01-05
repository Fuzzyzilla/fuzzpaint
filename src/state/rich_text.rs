//! # Rich text
//!
//! The owndership of a text is removed from the text layers, such that editting the contents is
//! not an edit of the layer but an edit of the text object. This makes even more sense when text
//! layers can have more than one text object within.
//!
//! Rich text effects operate on a grapheme basis. All indices will be in terms of graphemes, as it
//! never makes sense to color, say, half of a combining emoji or part of a combining Hangul character.
//! Similarly, the text caret should operate on graphemes as well (egui does not currently)
//!
//! It is not, however, bidi aware - it works out just as nicely without adding that layer of complexity.
//! It is up to the editor to handle the fact that a single visually contiguous seleciton *can be up to three
//! discontinuous spans* in a bidi context.

use crate::util::{Color, NonNanF32};

/// When inserting, is the caret glued to the span before or after it?
/// Used when choosing what properties to inherit.
pub enum CaretAffinity {
    Before,
    After,
}
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct Style {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strike: bool,
}
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Face {
    pub face: crate::repositories::fonts::FaceID,
}
struct TextProperties {
    color: rangemap::RangeMap<usize, Color>,
    face: rangemap::RangeMap<usize, Face>,
    size: rangemap::RangeMap<usize, NonNanF32>,
    style: rangemap::RangeMap<usize, Style>,
}
impl TextProperties {
    fn new() -> Self {
        Self {
            color: rangemap::RangeMap::new(),
            face: rangemap::RangeMap::new(),
            size: rangemap::RangeMap::new(),
            style: rangemap::RangeMap::new(),
        }
    }
    /// Set the graphemes to these styles. `None` clears the styles.
    fn set(
        &mut self,
        range: std::ops::Range<usize>,
        color: Option<Color>,
        size: Option<NonNanF32>,
        face: Option<Face>,
        style: Option<Style>,
    ) {
        // Short circuit for nonsense range.
        if range.start >= range.end {
            return;
        }
        if let Some(color) = color {
            self.color.insert(range.clone(), color);
        } else {
            self.color.remove(range.clone());
        }
        if let Some(size) = size {
            self.size.insert(range.clone(), size);
        } else {
            self.size.remove(range.clone());
        }
        if let Some(face) = face {
            self.face.insert(range.clone(), face);
        } else {
            self.face.remove(range.clone());
        }
        if let Some(style) = style {
            self.style.insert(range, style);
        } else {
            self.style.remove(range);
        }
    }
}
pub struct RichText {
    properties: TextProperties,
    len_graphemes: usize,
    text: String,
}
impl RichText {
    pub fn new(text: String) -> Self {
        use unicode_segmentation::UnicodeSegmentation;
        Self {
            properties: TextProperties::new(),
            len_graphemes: text.graphemes(true).count(),
            text,
        }
    }
    /// Iterate spans of distinct styles.
    /// Every returned span is guaranteed to be properly aligned to graphemes.
    #[must_use = "returns an iterator"]
    pub fn spans(&'_ self) -> RichTextSpanIter<'_> {
        RichTextSpanIter {
            cur_grapheme: 0,
            after_end_cursor: unicode_segmentation::GraphemeCursor::new(0, self.text.len(), true),
            text: &self.text,
            color: self.properties.color.iter().peekable(),
            style: self.properties.style.iter().peekable(),
            size: self.properties.size.iter().peekable(),
            face: self.properties.face.iter().peekable(),
        }
    }
    fn clamp_range(
        &self,
        grapheme_range: impl std::ops::RangeBounds<usize>,
    ) -> std::ops::Range<usize> {
        let start = match grapheme_range.start_bound() {
            std::ops::Bound::Unbounded => 0,
            std::ops::Bound::Included(&i) => i.min(self.len_graphemes),
            std::ops::Bound::Excluded(&i) => i.saturating_add(1).min(self.len_graphemes),
        };
        let end = match grapheme_range.end_bound() {
            std::ops::Bound::Unbounded => self.len_graphemes,
            std::ops::Bound::Included(&i) => (i.saturating_add(1)).min(self.len_graphemes),
            std::ops::Bound::Excluded(&i) => i.min(self.len_graphemes),
        };
        start..end
    }
    /// Set the graphemes to these styles. `None` clears the styles.
    /// range will be clamped to the range of self.
    pub fn set(
        &mut self,
        grapheme_range: impl std::ops::RangeBounds<usize>,
        color: Option<Color>,
        size: Option<NonNanF32>,
        face: Option<Face>,
        style: Option<Style>,
    ) {
        self.properties
            .set(self.clamp_range(grapheme_range), color, size, face, style);
    }
}
#[derive(Debug, PartialEq)]
pub struct RichTextSpan<'a> {
    /// Pre-context, if available. Not available at the start of a paragraph.
    pub pre: Option<&'a str>,
    /// The text, guaranteed to be aligned to graphemes.
    pub span: &'a str,
    /// Post-context, if available. Not available at the end of a paragraph.
    pub post: Option<&'a str>,
    pub color: Option<&'a Color>,
    pub size: Option<&'a NonNanF32>,
    pub face: Option<&'a Face>,
    pub style: Option<&'a Style>,
}
type PropertyIter<'a, V> = rangemap::map::Iter<'a, usize, V>;
use std::iter::Peekable;
pub struct RichTextSpanIter<'a> {
    /// Grapheme index of `after_end_cursor`
    cur_grapheme: usize,
    /// Current grapheme cursor
    // Holds valuable pre-processed info, so we keep it around between iterations.
    after_end_cursor: unicode_segmentation::GraphemeCursor,
    /// Full text of the RichText
    text: &'a str,
    // A bunch of iters to deal with in parallel
    color: Peekable<PropertyIter<'a, Color>>,
    size: Peekable<PropertyIter<'a, NonNanF32>>,
    face: Peekable<PropertyIter<'a, Face>>,
    style: Peekable<PropertyIter<'a, Style>>,
}
// Finds the next property event (start or end), along with the property that applies during `start..next`
// Consumes as it goes, so start_grapheme must be non-decreasing across calls with the same iter.
fn next_prop<'peek, 'inner: 'peek, V>(
    start_grapheme: usize,
    iter: &'peek mut Peekable<PropertyIter<'inner, V>>,
) -> Option<(usize, Option<&'inner V>)> {
    // Find the first event who's end is after `start` (could be the peeked item)
    let mut peek = iter.peek()?;
    // (In practice, this will run 0 or 1 times. is the codegen worse for it?)
    while peek.0.end <= start_grapheme {
        // Discard peeked.
        iter.next();
        // observe next
        peek = iter.peek()?;
    }
    // Now, peek is the first event after start.
    // If the start of that event is before the start, return position and the property of it. (already active)
    // If the start of that event is past start, return position and None. (hasn't started yet)
    if peek.0.start <= start_grapheme {
        // Lifetime funnies
        // even though `peek` is borrowed for 'peek, the reference therein is borrowed for 'inner
        Some((peek.0.end, Some(peek.1)))
    } else {
        Some((peek.0.start, None))
    }
}
impl<'a> Iterator for RichTextSpanIter<'a> {
    type Item = RichTextSpan<'a>;
    fn next(&mut self) -> Option<RichTextSpan<'a>> {
        // Start index, *in bytes*
        // definitely aligned to grapheme.
        let start_idx = self.after_end_cursor.cur_cursor();
        if start_idx >= self.text.len() {
            return None;
        }
        let color = next_prop(self.cur_grapheme, &mut self.color);
        let style = next_prop(self.cur_grapheme, &mut self.style);
        let face = next_prop(self.cur_grapheme, &mut self.face);
        let size = next_prop(self.cur_grapheme, &mut self.size);

        let end = [
            color.map(|(e, _)| e),
            style.map(|(e, _)| e),
            face.map(|(e, _)| e),
            size.map(|(e, _)| e),
        ]
        .into_iter()
        .flatten()
        .min();
        if let Some(end) = end {
            // Expect this to not underflow due to next_prop postconditions
            let dist = end - self.cur_grapheme;
            // Advance by `dist` graphemes
            for _ in 0..dist {
                // We expect this to never fail with context needed err,
                // as self.text is full text and thus should not end or begin
                // with partial grapheme.
                self.after_end_cursor.next_boundary(self.text, 0).unwrap();
            }
            self.cur_grapheme = end;
            // After end idx, *in bytes*
            let after_end_idx = self.after_end_cursor.cur_cursor();
            let span = &self.text[start_idx..after_end_idx];
            // No pre context if at the start.
            let pre = (start_idx != 0).then_some(&self.text[..start_idx]);
            // No post context if going until the end.
            let post = (after_end_idx < self.text.len()).then_some(&self.text[after_end_idx..]);
            Some(RichTextSpan {
                pre,
                post,
                span,
                color: color.and_then(|(_, v)| v),
                style: style.and_then(|(_, v)| v),
                size: size.and_then(|(_, v)| v),
                face: face.and_then(|(_, v)| v),
            })
        } else {
            // No events after this point, Return the rest of the string as one span.
            // Mark for next pass that we're done.
            self.after_end_cursor.set_cursor(self.text.len());
            // No pre context if the whole string is the span.
            let pre = (start_idx != 0).then_some(&self.text[..start_idx]);
            Some(RichTextSpan {
                pre,
                // Checked at start that this is fine
                span: &self.text[start_idx..],
                post: None,
                color: None,
                face: None,
                size: None,
                style: None,
            })
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn iter_nostyles() {
        let rt = super::RichText::new("Uwu!!".to_owned());

        assert_eq!(
            rt.spans().collect::<Vec<_>>(),
            [super::RichTextSpan {
                pre: None,
                post: None,
                span: "Uwu!!",
                color: None,
                size: None,
                style: None,
                face: None,
            }]
            .into_iter()
            .collect::<Vec<_>>(),
        );
    }
    #[test]
    fn iter_unpaired_graphemes() {
        let rt = super::RichText::new("Wow!\u{1F1E6}".to_owned());

        assert_eq!(rt.spans().count(), 1);
    }
    #[test]
    fn iter_empty() {
        let rt = super::RichText::new(String::new());

        assert_eq!(rt.spans().count(), 0);
    }
    #[test]
    fn context() {
        let mut rt = super::RichText::new("Wow".to_owned());
        // Setting the style of the "o" forces this into three spans:
        rt.set(1..2, Some(super::Color::BLACK), None, None, None);

        // Check each span has proper context:
        assert_eq!(
            rt.spans().collect::<Vec<_>>(),
            [
                super::RichTextSpan {
                    pre: None,
                    span: "W",
                    post: Some("ow"),
                    color: None,
                    face: None,
                    size: None,
                    style: None,
                },
                super::RichTextSpan {
                    pre: Some("W"),
                    span: "o",
                    post: Some("w"),
                    color: Some(&super::Color::BLACK),
                    face: None,
                    size: None,
                    style: None,
                },
                super::RichTextSpan {
                    pre: Some("Wo"),
                    span: "w",
                    post: None,
                    color: None,
                    face: None,
                    size: None,
                    style: None,
                }
            ]
            .into_iter()
            .collect::<Vec<_>>(),
        );
    }
    #[test]
    fn overlapping_spans() {
        let mut rt = super::RichText::new("0123456789".to_owned());
        rt.set(1..5, Some(super::Color::BLACK), None, None, None);
        // API aint great at time of writing lol
        rt.set(
            3..8,
            Some(super::Color::BLACK),
            Some(super::NonNanF32::ONE),
            None,
            None,
        );
        rt.set(5..8, None, Some(super::NonNanF32::ONE), None, None);
        // we have two overlapping style ranges:
        // should create 5 spans.
        //       [---------)
        //   [-------)
        // 0 1 2 3 4 5 6 7 8 9
        assert_eq!(
            rt.spans().collect::<Vec<_>>(),
            [
                super::RichTextSpan {
                    pre: None,
                    span: "0",
                    post: Some("123456789"),
                    color: None,
                    face: None,
                    size: None,
                    style: None,
                },
                super::RichTextSpan {
                    pre: Some("0"),
                    span: "12",
                    post: Some("3456789"),
                    color: Some(&super::Color::BLACK),
                    face: None,
                    size: None,
                    style: None,
                },
                super::RichTextSpan {
                    pre: Some("012"),
                    span: "34",
                    post: Some("56789"),
                    color: Some(&super::Color::BLACK),
                    face: None,
                    size: Some(&super::NonNanF32::ONE),
                    style: None,
                },
                super::RichTextSpan {
                    pre: Some("01234"),
                    span: "567",
                    post: Some("89"),
                    color: None,
                    face: None,
                    size: Some(&super::NonNanF32::ONE),
                    style: None,
                },
                super::RichTextSpan {
                    pre: Some("01234567"),
                    span: "89",
                    post: None,
                    color: None,
                    face: None,
                    size: None,
                    style: None,
                }
            ]
            .into_iter()
            .collect::<Vec<_>>(),
        );
    }
}
