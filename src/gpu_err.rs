///Implement some kind of builder pattern around GPU errors, to allow concise
/// inline error coersion into a more useful form than a big enum of doom.

/// Wrapper around vulkano errors, to be returned by APIs that work
/// directly with the GPU. Indicates a raw source, possible solution, and whether
/// the error destroyed the resource it was returned by.
#[derive(thiserror::Error, Debug, Clone)]
#[error("{} GpuError: {source:?}", if *.fatal {"[Lost]"} else {"[Recoverable]"})]
pub struct GpuError {
    /// Whether the resource that returned this error died as a result of the error.
    pub fatal : bool,
    /// The underlying source of the error
    pub source : GpuErrorSource,
}
//Private API for construction by conversions below
impl GpuError {
    fn fatal_with_source(source : GpuErrorSource) -> Self {
        Self {
            source,
            fatal: true,
        }
    }
    fn recoverable_with_source(source : GpuErrorSource) -> Self {
        Self {
            source,
            fatal: false,
        }
    }
}
// Public API
pub trait GpuResultInspection {
    type ErrTy : Clone;
    fn unless(self, p : impl FnOnce(Self::ErrTy) -> bool) -> Self;
    fn recoverable_if_not_lost(self) -> Self;
    fn recoverable_if(self, p: impl FnOnce(Self::ErrTy) -> bool) -> Self;
}
impl GpuResultInspection for GpuError {
    type ErrTy = Self;
    fn unless(self, p : impl FnOnce(Self::ErrTy) -> bool) -> Self {
        let predicate_value = p(self.clone());
        //XOR - toggle fatal if predicate passed
        let new_fatal = self.fatal ^ predicate_value;

        Self {
            fatal: new_fatal,
            ..self
        }

    }
    fn recoverable_if_not_lost(self) -> Self {
        let is_lost = self.source.device_lost();

        Self {
            fatal: is_lost,
            ..self
        }
    }
    fn recoverable_if(self, p: impl FnOnce(Self::ErrTy) -> bool) -> Self {
        let recoverable = p(self);

        Self {
            fatal: !recoverable,
            ..self
        }
    }
}

pub type GpuResult<OkTy> = ::std::result::Result<OkTy, GpuError>;

impl<OkTy> GpuResultInspection for GpuResult<OkTy> {
    type ErrTy = GpuError;
    fn recoverable_if(self, p: impl FnOnce(GpuError) -> bool) -> Self {
        match self {
            Ok(ok) => Ok(ok),
            Err(err) => Err(err.recoverable_if(p))
        }
    }
    fn recoverable_if_not_lost(self) -> Self {
        match self {
            Ok(ok) => Ok(ok),
            Err(err) => Err(err.recoverable_if_not_lost())
        }
    }
    fn unless(self, p : impl FnOnce(Self::ErrTy) -> bool) -> Self {
        match self {
            Ok(ok) => Ok(ok),
            Err(err) => Err(err.unless(p))
        }
    }
}

#[derive(thiserror::Error, Debug, Eq, PartialEq, Clone)]
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


//Blanket implementations, re-use the matching logic below to
//report Oom and DeviceLoss errors for all vulkano GPU error types.
//May incur a small extra cost for round-trip conversion? Especially
//given that some of the types can *never* report Oom or Device loss,
//so implementing these traits on them seems a bit redundant. Oh well, lazy.
impl<ErrTy> HasDeviceLoss for ErrTy
    where ErrTy : Into<GpuErrorSource> {
    fn device_lost(&self) -> bool {
        let err_source = Into::<GpuErrorSource>::into(*self);

        GpuErrorSource::DeviceLost == err_source
    }
}
impl<ErrTy> HasOom for ErrTy
    where ErrTy : Into<GpuErrorSource>  {
    fn oom(&self) -> Option<vulkano::OomError> {
        let err_source = Into::<GpuErrorSource>::into(*self);

        if let GpuErrorSource::OomError(oom) = err_source {
            Some(oom)
        } else {
            None
        }
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
            SemaphoreError::OomError(oom) => oom.into(),
            SemaphoreError::RequirementNotMet { required_for, requires_one_of }
                => Self::RequirementNotMet { required_for, requires_one_of },
            SemaphoreError::InQueue => Self::ResourceInUse,
            SemaphoreError::QueueIsWaiting => Self::ResourceInUse,
            _ => {
                Self::SemaphoreError(value)
            }
        }
    }
}

pub trait IntoGpuResult {
    type OkTy;
    fn fatal(self) -> GpuResult<Self::OkTy>;
    fn recoverable(self) -> GpuResult<Self::OkTy>;
}

impl<OkTy, ErrTy> IntoGpuResult for ::std::result::Result<OkTy, ErrTy>
    where ErrTy : Into<GpuErrorSource>
{
    type OkTy = OkTy;
    fn fatal(self) -> GpuResult<OkTy> {
        self
            .map_err(|err|
                GpuError::fatal_with_source(err.into())
            )
    }
    fn recoverable(self) -> GpuResult<OkTy> {
        self
            .map_err(|err|
                GpuError::recoverable_with_source(err.into())
            )
    }
}
impl<OkTy> IntoGpuResult for GpuResult<OkTy> {
    type OkTy = OkTy;
    fn fatal(self) -> GpuResult<Self::OkTy> {
        self.map_err(
            |err| GpuError { fatal: true, ..err }
        )
    }
    fn recoverable(self) -> GpuResult<Self::OkTy> {
        self.map_err(
            |err| GpuError { fatal: false, ..err }
        )
    }
}
impl IntoGpuResult for GpuErrorSource {
    type OkTy = std::convert::Infallible; // Maybe one day never-type will stabilize ;P
    fn fatal(self) -> GpuResult<Self::OkTy> {
        Err(
            GpuError::fatal_with_source(self)
        )
    }
    fn recoverable(self) -> GpuResult<Self::OkTy> {
        Err(
            GpuError::recoverable_with_source(self)
        )
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