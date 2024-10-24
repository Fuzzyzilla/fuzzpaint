#[derive(thiserror::Error, Debug, Copy, Clone)]
pub enum CreatePickerError {
    #[error("unknown document id")]
    UnknownDocument,
    #[error("unknown leaf or node id")]
    UnknownLayer,
    /// Eg, trying to pick color from a `Note`.
    #[error("target object does carry the requested data")]
    Uninhabited,
    #[error("picker transform is malformed")]
    BadTransform,
}

use tokio::sync::oneshot::Sender as RequestResponse;
type PickerResponse<Picker> = RequestResponse<Result<Picker, CreatePickerError>>;
#[derive(Copy, Clone)]
pub struct PickerInfo {
    /// Current viewport transform and size
    pub viewport: crate::view_transform::ViewInfo,
    /// A hint as to where a sample will take place, in viewport space.
    /// The returned picker may only be valid for only for some unspecified range around this position,
    /// outside of which it will return [`crate::picker::PickError::NeedsRefresh`]
    pub sample_pos: ultraviolet::Vec2,
    /// Maximum granularity of the pointing device being used to sample.
    pub input_points_per_viewport_pixel: f32,
}
pub enum PickerRequest {
    /// Sample from the final, composited image.
    Composited(PickerResponse<super::picker::RenderedColorPicker>),
    /// Sample from the rendered output of a given leaf or node.
    Rendered(
        fuzzpaint_core::state::graph::AnyID,
        PickerResponse<super::picker::RenderedColorPicker>,
    ),
}
pub enum RenderRequest {
    CreatePicker {
        document: fuzzpaint_core::state::document::ID,
        picker: PickerRequest,
        info: PickerInfo,
    },
}
pub(super) async fn handler(mut recv: tokio::sync::mpsc::Receiver<RenderRequest>) {
    // Live as long as there are requests to serve
    while let Some(recv) = recv.recv().await {
        // Placeholder - fail out every request x3
        let RenderRequest::CreatePicker { picker, .. } = recv;

        match picker {
            PickerRequest::Composited(response) | PickerRequest::Rendered(_, response) => {
                let _ = response.send(Err(CreatePickerError::Uninhabited));
            }
        }
    }
}
