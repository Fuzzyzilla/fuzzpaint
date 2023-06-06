///Implement some kind of builder pattern around GPU errors, to allow concise
/// inline error coersion into a more useful form than a big enum of doom.

pub type GpuResult<OkTy> = ::std::result::Result<OkTy, GpuError>;

/// Wrapper around vulkano errors, to be returned by APIs that work
/// directly with the GPU. Indicates a raw source, possible solution, and whether
/// the error destroyed the resource it was returned by.
#[derive(thiserror::Error, Debug, Clone)]
#[error("{} GpuError: {source:?}", if *.fatal {"[Lost]"} else {"[Recoverable]"})]
//Interestingly this is needed for unused Result lints, but the text is not used x3
#[must_use = "GpuError may indicate fatal errors and should not be ignored."]
pub struct GpuError {
    /// Whether the resource that returned this error died as a result of the error.
    pub fatal: bool,
    /// The underlying source of the error
    pub source: GpuErrorSource,
}
//Private API for construction by conversions below
impl GpuError {
    fn fatal_with_source(source: GpuErrorSource) -> Self {
        Self {
            source,
            fatal: true,
        }
    }
    fn recoverable_with_source(source: GpuErrorSource) -> Self {
        Self {
            source,
            fatal: false,
        }
    }
}
// Public API
pub trait GpuResultInspection {
    type ErrTy: Clone;
    fn unless(self, p: impl FnOnce(Self::ErrTy) -> bool) -> Self;
}
impl GpuResultInspection for GpuError {
    type ErrTy = Self;
    fn unless(self, p: impl FnOnce(Self::ErrTy) -> bool) -> Self {
        let predicate_value = p(self.clone());
        //XOR - toggle fatal if predicate passed
        let new_fatal = self.fatal ^ predicate_value;

        Self {
            fatal: new_fatal,
            ..self
        }
    }
}

#[derive(thiserror::Error, Debug, Clone)]
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
    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: vulkano::RequiresOneOf,
    },
    #[error("Wait Timeout")]
    Timeout,
    #[error("Requested resource already in use")]
    ResourceInUse,
    #[error("{0:?}")]
    RenderpassCreationError(vulkano::render_pass::RenderPassCreationError),
    #[error("{0:?}")]
    ShaderCreationError(vulkano::shader::ShaderCreationError),
    #[error("{0:?}")]
    GraphicsPipelineCreationError(vulkano::pipeline::graphics::GraphicsPipelineCreationError),
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

impl HasDeviceLoss for GpuError {
    fn device_lost(&self) -> bool {
        if let GpuErrorSource::DeviceLost = self.source {
            true
        } else {
            false
        }
    }
}
impl HasOom for GpuError {
    fn oom(&self) -> Option<vulkano::OomError> {
        if let GpuErrorSource::OomError(oom) = self.source {
            Some(oom)
        } else {
            None
        }
    }
}
pub trait IntoGpuResultBuilder {
    type OkTy;
    type ErrTy: Into<GpuErrorSource>;
    fn fatal(self) -> super::GpuResultBuilder<Self::OkTy, Self::ErrTy>;
    fn recoverable(self) -> super::GpuResultBuilder<Self::OkTy, Self::ErrTy>;
}

impl<OkTy, ErrTy> IntoGpuResultBuilder for ::std::result::Result<OkTy, ErrTy>
where
    ErrTy: Into<GpuErrorSource>,
{
    type OkTy = OkTy;
    type ErrTy = ErrTy;
    fn fatal(self) -> super::GpuResultBuilder<OkTy, ErrTy> {
        super::GpuResultBuilder::new_fatal(self)
    }
    fn recoverable(self) -> super::GpuResultBuilder<OkTy, ErrTy> {
        super::GpuResultBuilder::new_recoverable(self)
    }
}

impl<OkTy, ErrTy: HasDeviceLoss> HasDeviceLoss for ::std::result::Result<OkTy, ErrTy> {
    fn device_lost(&self) -> bool {
        self.as_ref().err().is_some_and(|err| err.device_lost())
    }
}
impl<OkTy, ErrTy: HasOom> HasOom for ::std::result::Result<OkTy, ErrTy> {
    fn oom(&self) -> Option<vulkano::OomError> {
        self.as_ref().err().map(|err| err.oom()).unwrap_or(None)
    }
}
