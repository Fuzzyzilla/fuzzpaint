#[derive(thiserror::Error, Debug)]
#[error("{}GpuError: {source:?}", if *.resource_lost {"[Lost] "} else {""})]
pub struct GpuError {
    //The actions needed to fix this error state.
    pub remedy : GpuRemedy,
    //Whether the resource that returned this error died as a result of the error.
    pub resource_lost : bool,
    //The underlying source of the error
    pub source : vulkano::VulkanError,
}
// The steps needed to recover from a catastrophic failure.
// There is an implicit hierarchy here, higher enums imply the ones below.
#[derive(Debug)]
pub enum GpuRemedy {
    ReevaluateDevices,
    RecreateDevice,
    RecreateSurface,
    RecreateSwapchain,
}