///Implement some kind of builder pattern around GPU errors, to allow concise
/// inline error coersion into a more useful form than a big enum of doom.

/// Wrapper around vulkano errors, to be returned by APIs that work
/// directly with the GPU. Indicates a raw source, possible solution, and whether
/// the error destroyed the resource it was returned by.
#[derive(thiserror::Error, Debug)]
#[error("{} GpuError: {source:?}", if *.fatal {"[Lost]"} else {"[Recoverable]"})]
pub struct GpuError {
    /// Whether the resource that returned this error died as a result of the error.
    pub fatal : bool,
    /// The underlying source of the error
    pub source : GpuErrorSource,
}

pub type GpuResult<OkTy> = ::std::result::Result<OkTy, GpuError>;

#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum GpuErrorSource {
    #[error("Connection to device was lost.")]
    DeviceLost,
    #[error("{0:?}")]
    OomError(vulkano::OomError),
    #[error("{0:?}")]
    //Some variants of this enum mirror the variants of GpuErrorSource. Those
    //will never be reported.
    FenceError(vulkano::sync::fence::FenceError),
    #[error("{0:?}")]
    SemaphoreError(vulkano::sync::semaphore::SemaphoreError),
    #[error("Requirement not met: {requires_one_of:?} required for {required_for}")]
    RequirementNotMet{required_for : &'static str, requires_one_of: vulkano::RequiresOneOf},
    #[error("Wait Timeout")]
    Timeout,
    #[error("Requested resource already in use")]
    ResourceInUse,
}

//Blanket implementations, re-use the matching logic below to
//report Oom and DeviceLoss errors for all vulkano GPU error types.
//May incur a small extra cost for round-trip conversion? Especially
//given that some of the types can *never* report Oom or Device loss,
//so implementing these traits on them seems a bit redundant. Oh well, lazy.
impl<ErrTy> HasDeviceLoss for ErrTy
    where ErrTy : Into<GpuErrorSource> {
    fn device_lost(&self) -> bool {
        let err_source = self.into();

        GpuErrorSource::DeviceLost == err_source
    }
}
impl<ErrTy> HasOom for ErrTy
    where ErrTy : Into<GpuErrorSource>  {
    fn oom(&self) -> Option<vulkano::OomError> {
        let err_source = self.into();

        if let GpuErrorSource::OomError(oom) = err_source {
            Some(*oom)
        } else {
            None
        }
    }
}

// Traits for all GPU errors, to detect major faults.
///Trait for errors which may include DeviceLost.
pub trait HasDeviceLoss {
    fn device_lost(&self) -> bool;
}
///Trait for errors which may include device or host memory exhaustion. This
/// is true of most errors under vulkan.
pub trait HasOom {
    fn oom(&self) -> Option<vulkano::OomError>;
    fn is_oom(&self) -> bool {
        self.oom().is_some()
    }
    fn is_host_oom(&self) -> bool {
        self.oom()
            .is_some_and(|oom| oom == vulkano::OomError::OutOfHostMemory)
    }
    fn is_device_oom(&self) -> bool {
        self.oom()
            .is_some_and(|oom| oom == vulkano::OomError::OutOfDeviceMemory)
    }
}

impl From<vulkano::OomError> for GpuErrorSource{
    fn from(value: vulkano::OomError) -> Self {
        Self::OomError(value)
    }
}
impl From<vulkano::sync::fence::FenceError> for GpuErrorSource{
    fn from(value: vulkano::sync::fence::FenceError) -> Self {
        use vulkano::sync::fence::FenceError;
        match value {
            FenceError::DeviceLost => Self::DeviceLost,
            FenceError::OomError(oom) => oom.into(),
            FenceError::RequirementNotMet { required_for, requires_one_of }
                => Self::RequirementNotMet { required_for, requires_one_of },
            FenceError::Timeout => Self::Timeout,
            FenceError::InQueue => Self::ResourceInUse,
            _ => {
                Self::FenceError(value)
            }
        }
    }
}
impl From<vulkano::sync::semaphore::SemaphoreError> for GpuErrorSource{
    fn from(value: vulkano::sync::semaphore::SemaphoreError) -> Self {
        use vulkano::sync::semaphore::SemaphoreError;
        match value {
            SemaphoreError::DeviceLost => Self::DeviceLost,
            SemaphoreError::OomError(oom) => oom.into(),
            SemaphoreError::RequirementNotMet { required_for, requires_one_of }
                => Self::RequirementNotMet { required_for, requires_one_of },
                SemaphoreError::Timeout => Self::Timeout,
            SemaphoreError::InQueue => Self::ResourceInUse,
            SemaphoreError::QueueIsWaiting => Self::ResourceInUse,
            _ => {
                Self::SemaphoreError(value)
            }
        }
    }
}

impl<OkTy, ErrTy> From<::std::result::Result<OkTy, ErrTy>> for GpuResult<OkTy>
    where ErrTy : Into<GpuErrorSource>
{
    fn from(value: ::std::result::Result<OkTy, ErrTy>) -> Self {
        value
            .map_err(|err| err.into())
    }
}



impl<OkTy, ErrTy : HasDeviceLoss> HasDeviceLoss for ::std::result::Result<OkTy, ErrTy> {
    fn device_lost(&self) -> bool {
        self.as_ref()
            .err()
            .is_some_and(|err| err.device_lost())
    }
}
impl<OkTy, ErrTy : HasOom> HasOom for ::std::result::Result<OkTy, ErrTy> {
    fn oom(&self) -> Option<vulkano::OomError> {
        self.as_ref()
            .err()
            .map(|err| err.oom())
            .unwrap_or(None)
    }
}

trait OnDeviceLoss {
    type OkTy;
    type ErrTy;
    fn die_on_device_loss(self) -> ::std::result::Result<Self::OkTy, GpuError>;
}
impl<OkTy, ErrTy : HasDeviceLoss> OnDeviceLoss for ::std::result::Result<OkTy, ErrTy> {
    type OkTy = OkTy;
    type ErrTy = ErrTy;
    fn die_on_device_loss(self) -> ::std::result::Result<Self::OkTy, GpuError> {
        self.map_err(|err| {

        })
    }
}
