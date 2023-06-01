/// Wrapper around vulkano errors, to be returned by APIs that work
/// directly with the GPU. Indicates a raw source, possible solution, and whether
/// the error destroyed the resource it was returned by.
#[derive(thiserror::Error, Debug)]
#[error("{} GpuError: {source:?}", if *.resource_lost {"[Lost]"} else {"[Recoverable]"})]
pub struct GpuError<Source : std::error::Error> {
    /// Possible actions needed to fix this error state.
    pub remedy : GpuRemedy,
    /// Whether the resource that returned this error died as a result of the error.
    pub resource_lost : bool,
    /// The underlying source of the error
    pub source : Source,
}
pub trait GpuResult {
    type OkTy;
    type ErrTy : std::error::Error;
    fn fatal(self) -> Result<Self::OkTy, GpuError<Self::ErrTy>>;
}

impl<OkTy> GpuResult for Result<OkTy, vulkano::OomError> {
    type OkTy = OkTy;
    type ErrTy = vulkano::OomError;
    fn fatal(self) -> Result<Self::OkTy, GpuError<Self::ErrTy>> {
        self.map_err(
            |err| GpuError{
                remedy: GpuRemedy::Oom(err.clone()),
                resource_lost: true,
                source: err
            }
        )
    }
}
/// The steps needed to recover from a catastrophic failure.
/// There is an implicit hierarchy here, higher enums imply the ones below.
#[derive(Debug)]
pub enum GpuRemedy {
    /// The previous device is no longer compatible. Perhaps physical monitor configuration changed.
    ReevaluateDevices,
    /// The device was lost.
    RecreateDevice,
    /// The surface is no longer compatible with the device.
    RecreateSurface,
    /// The swapchain was out-of-date or otherwise unusable.
    RecreateSwapchain,

    /// An impossible operation was attempted, like bad shader. Nothing can be done but blame me.
    None,
    /// Out of memory.
    Oom(vulkano::OomError),
}