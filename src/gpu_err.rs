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

#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum GpuErrorSource {
    #[error("Connection to device was lost.")]
    DeviceLost,
    #[error("{0:?}")]
    OomError(vulkano::OomError),
    #[error("{0:?}")]
    FenceError(vulkano::sync::fence::FenceError),
    #[error("{0:?}")]
    SemaphoreError(vulkano::sync::semaphore::SemaphoreError),
}

impl HasDeviceLoss for GpuErrorSource {
    fn is_device_lost(&self) -> bool {
        GpuErrorSource::DeviceLost == *self
    }
}
impl HasOom for GpuErrorSource {
    fn oom(&self) -> Option<vulkano::OomError> {
        if let GpuErrorSource::OomError(oom) = self {
            Some(*oom)
        } else {
            None
        }
    }
}

// Traits for all GPU errors, to detect major faults.
///Trait for errors which may include DeviceLost.
pub trait HasDeviceLoss {
    fn is_device_lost(&self) -> bool;
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
            .map(|oom| oom == vulkano::OomError::OutOfHostMemory)
            .unwrap_or(false)
    }
    fn is_device_oom(&self) -> bool {
        self.oom()
            .map(|oom| oom == vulkano::OomError::OutOfDeviceMemory)
            .unwrap_or(false)
    }
}

impl<OkTy, ErrTy : HasDeviceLoss> HasDeviceLoss for ::std::result::Result<OkTy, ErrTy> {
    fn is_device_lost(&self) -> bool {
        self.as_ref()
            .err()
            .map(|err| err.is_device_lost())
            .unwrap_or(false)
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
