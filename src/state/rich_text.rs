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
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
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
/// Expand from the given position, inheriting properties based on the affinity.
fn insert_inherited<T: Eq + Clone>(
    map: &mut rangemap::RangeMap<usize, T>,
    pos: usize,
    len: usize,
    affinity: CaretAffinity,
) {
    // impossible, or highly hack-y.
    // Do i need a fork? :V
    todo!()
}
/// Checks that the insert doesn't fuse with pre or post. (ie, the sum of
/// grapheme count of each is the same as the grapheme count of them concatenated).
/// `false` if they fuse.
///
/// It is assumed that the string from which `pre` and `post` are
/// taken was split at a grapheme boundary.
fn check_doesnt_fuse(pre: &str, insert: &str, post: &str) -> bool {
    use unicode_segmentation::{GraphemeCursor, GraphemeIncomplete};
    // Make a cursor into the theoretical merged text:
    let mut cursor = GraphemeCursor::new(pre.len(), pre.len() + insert.len() + post.len(), true);
    // Check start:
    match cursor.is_boundary(insert, pre.len()) {
        Ok(false) => return false,
        Ok(true) => (),
        Err(GraphemeIncomplete::PreContext(end)) => {
            // Asked for further context, provide the start and try again:
            cursor.provide_context(&pre[..end], 0);
            // Failed again. Any "incomplete" errors here hint at merging.
            if !cursor.is_boundary(insert, pre.len()).is_ok_and(|b| b) {
                return false;
            }
        }
        // Other err kinds do not apply to this operation.
        Err(_) => unreachable!(),
    }

    // Check end:
    // Yummy syntactic recursion >w>
    cursor.set_cursor(pre.len() + insert.len());
    match cursor.is_boundary(post, pre.len() + insert.len()) {
        Ok(false) => false,
        Ok(true) => true,
        Err(GraphemeIncomplete::PreContext(end)) => {
            // These assume the order it will ask for context in. I don't wanna fix that. Blegh.
            // Asked for further context, provide `insert` and try again:
            cursor.provide_context(&insert[..(end - pre.len())], pre.len());
            match cursor.is_boundary(insert, pre.len()) {
                Ok(false) => false,
                Ok(true) => true,
                Err(GraphemeIncomplete::PreContext(end)) => {
                    // Asked for further context, provide the start and try again:
                    cursor.provide_context(&pre[..end], 0);
                    // Failed again. Any "incomplete" errors here hint at merging.
                    cursor.is_boundary(insert, pre.len()).is_ok_and(|b| b)
                }
                // Other err kinds do not apply to this operation.
                Err(_) => unreachable!(),
            }
        }
        // Other err kinds do not apply to this operation.
        Err(_) => unreachable!(),
    }
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
    /// Expand from the given position, inheriting properties based on the affinity.
    fn insert_inherited(&mut self, pos: usize, len: usize, affinity: CaretAffinity) {
        insert_inherited(&mut self.color, pos, len, affinity);
        insert_inherited(&mut self.size, pos, len, affinity);
        insert_inherited(&mut self.face, pos, len, affinity);
        insert_inherited(&mut self.style, pos, len, affinity);
    }
}
pub struct RichTextParagraph {
    properties: TextProperties,
    len_graphemes: usize,
    text: String,
}
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum TextInsertError {
    #[error("inserted text has incomplete graphemes on either end")]
    DanglingGraphemes,
}
impl RichTextParagraph {
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
    pub fn spans(&'_ self) -> RichTextParagraphSpans<'_> {
        RichTextParagraphSpans {
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
    pub fn insert(
        &mut self,
        text: &str,
        grapheme_position: usize,
        affinity: CaretAffinity,
    ) -> Result<(), TextInsertError> {
        use unicode_segmentation::{GraphemeCursor, UnicodeSegmentation};
        // Avoid weirdness
        if text.is_empty() {
            return Ok(());
        }

        // Clamp to end
        let grapheme_position = grapheme_position.min(self.len_graphemes);
        // How many graphemes are we inserting?
        let num_graphemes = text.graphemes(true).count();
        // Into what byte position do we insert these chars?
        let insert_position = self
            .text
            .grapheme_indices(true)
            .nth(grapheme_position)
            .map_or(self.text.len(), |(pos, _)| pos);
        debug_assert!(insert_position <= self.text.len());

        // Check that, upon insertion of this text, the world wont explode
        // (ie, that the graphemes from either side won't merge with the new text,
        // ruining all the indices in the process)
        if !check_doesnt_fuse(
            &self.text[..insert_position],
            text,
            &self.text[insert_position..],
        ) {
            return Err(TextInsertError::DanglingGraphemes);
        }

        // Make space in the properties
        self.properties
            .insert_inherited(grapheme_position, num_graphemes, affinity);
        // Insert the string into the made space.
        self.text.insert_str(insert_position, text);
        self.len_graphemes += num_graphemes;

        Ok(())
    }
}
#[derive(Debug, PartialEq)]
pub struct RichTextSpan<'a> {
    /// Pre-context, if available. Not available at the start of a line or paragraph.
    pub pre: Option<&'a str>,
    /// The text, guaranteed to be aligned to graphemes and containing no line breaks or paragraph breaks.
    pub span: &'a str,
    /// Post-context, if available. Not available at the end of a line or paragraph.
    pub post: Option<&'a str>,
    pub color: Option<&'a Color>,
    pub size: Option<&'a NonNanF32>,
    pub face: Option<&'a Face>,
    pub style: Option<&'a Style>,
}
type PropertyIter<'a, V> = rangemap::map::Iter<'a, usize, V>;
use std::iter::Peekable;
pub struct RichTextParagraphSpans<'a> {
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
impl<'a> Iterator for RichTextParagraphSpans<'a> {
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
        let rt = super::RichTextParagraph::new("Uwu!!".to_owned());

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
        let rt = super::RichTextParagraph::new("Wow!\u{1F1E6}".to_owned());

        assert_eq!(rt.spans().count(), 1);
    }
    #[test]
    fn iter_empty() {
        let rt = super::RichTextParagraph::new(String::new());

        assert_eq!(rt.spans().count(), 0);
    }
    #[test]
    fn context() {
        let mut rt = super::RichTextParagraph::new("Wow".to_owned());
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
        let mut rt = super::RichTextParagraph::new("0123456789".to_owned());
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
    #[test]
    fn merge_error() {
        // Ends with "REGIONAL INDICATOR A"
        let mut rt = super::RichTextParagraph::new("Wow!\u{1F1E6}".to_owned());
        // Inserting another regional indicator forms a flag, and should not be allowed for now.
        // (will later imply more logic that must occur to make everything dandy)
        assert_eq!(
            rt.insert("\u{1F1E6}", 100, super::CaretAffinity::Before),
            Err(super::TextInsertError::DanglingGraphemes)
        );

        let mut rt = super::RichTextParagraph::new("\u{1F1E6}Soup!!".to_owned());
        assert_eq!(
            rt.insert("\u{1F1E6}", 0, super::CaretAffinity::Before),
            Err(super::TextInsertError::DanglingGraphemes)
        );
        let mut rt = super::RichTextParagraph::new("Wwawawaawawawa\u{1F1E6}Soup!!".to_owned());
        // Inserting regional indicator here doesn't merge and is fine.
        assert_eq!(
            rt.insert("\u{1F1E6}", 2, super::CaretAffinity::Before),
            Ok(())
        );
    }
}
