/// Wrapper around vulkano errors, to be returned by APIs that work
/// directly with the GPU. Indicates a raw source, possible solution, and whether
/// the error destroyed the resource it was returned by.
#[derive(thiserror::Error, Debug)]
#[error("{} GpuError: {source:?}", if *.resource_lost {"[Lost]"} else {"[Recoverable]"})]
pub struct GpuError {
    /// Possible actions needed to fix this error state.
    pub remedy : GpuRemedy,
    /// Whether the resource that returned this error died as a result of the error.
    pub resource_lost : bool,
    /// The underlying source of the error
    pub source : Option<Box<dyn std::error::Error>>,
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

    /// A required feature/extension/version/etc was expected but not met.
    RequirementNotMet(vulkano::RequiresOneOf),

    /// An impossible operation was attempted, like bad shader, or double-used sync primitive. Nothing can be done but blame me.
    /// Really means that debugging should be performed, and in practice the renderer should be restarted.
    BlameTheDev,
    /// Out of memory.
    Oom(vulkano::OomError),
}

pub trait DefaultRemedy: std::error::Error {
    fn default_remedy(&self) -> GpuRemedy;
}
impl DefaultRemedy for vulkano::OomError {
    fn default_remedy(&self) -> GpuRemedy {
        GpuRemedy::Oom(self.clone())
    }
}
impl DefaultRemedy for vulkano::swapchain::AcquireError {
    fn default_remedy(&self) -> GpuRemedy {
        match self {
            Self::DeviceLost => GpuRemedy::RecreateDevice,
            //Todo: Speculative solution here
            Self::FullScreenExclusiveModeLost => GpuRemedy::RecreateSurface,
            Self::SurfaceLost => GpuRemedy::RecreateSurface,
            Self::OutOfDate => GpuRemedy::RecreateSurface,

            //Less of an error and more of a notification. Use map_gpu_err to deal with this weirdness :3
            Self::Timeout => GpuRemedy::BlameTheDev,

            Self::FenceError(fence_err) => fence_err.default_remedy(),
            Self::SemaphoreError(sem_err) => sem_err.default_remedy(),

            Self::OomError(oom) => oom.default_remedy(),
        }
    }
}
impl DefaultRemedy for vulkano::sync::fence::FenceError {
    fn default_remedy(&self) -> GpuRemedy {
        match self {
            Self::DeviceLost => GpuRemedy::RecreateDevice,
            Self::InQueue => GpuRemedy::BlameTheDev,
            Self::ImportedForSwapchainAcquire => GpuRemedy::BlameTheDev,
            Self::OomError(oom) => oom.default_remedy(),
            Self::RequirementNotMet { requires_one_of, ..} => GpuRemedy::RequirementNotMet(requires_one_of.clone()),

            _ => {
                //Platform specific operations which may fail. Not using any in this codebase :3
                eprintln!("Err {self:?} encountered without a default remedy.");
                GpuRemedy::BlameTheDev
            }
        }
    }
}
impl DefaultRemedy for vulkano::sync::semaphore::SemaphoreError {
    fn default_remedy(&self) -> GpuRemedy {
        match self {
            Self::InQueue => GpuRemedy::BlameTheDev,
            Self::ImportedForSwapchainAcquire => GpuRemedy::BlameTheDev,
            Self::OomError(oom) => oom.default_remedy(),
            Self::RequirementNotMet { requires_one_of, ..} => GpuRemedy::RequirementNotMet(requires_one_of.clone()),
            Self::QueueIsWaiting => GpuRemedy::BlameTheDev,
            _ => {
                //Platform specific operations which may fail. Not using any in this codebase :3
                eprintln!("Err {self:?} encountered without a default remedy.");
                GpuRemedy::BlameTheDev
            }
        }
    }
}
impl DefaultRemedy for vulkano::shader::ShaderCreationError {
    fn default_remedy(&self) -> GpuRemedy {
        match self {
            Self::OomError(oom) => oom.default_remedy(),
            //Todo!
            _ => GpuRemedy::BlameTheDev,
        }
    }
}
impl DefaultRemedy for vulkano::render_pass::RenderPassCreationError {
    fn default_remedy(&self) -> GpuRemedy {
        match self {
            Self::OomError(oom) => oom.default_remedy(),
            //Todo!
            _ => GpuRemedy::BlameTheDev
        }
    }
}
impl DefaultRemedy for vulkano::pipeline::graphics::GraphicsPipelineCreationError {
    fn default_remedy(&self) -> GpuRemedy {
        match self {
            Self::OomError(oom) => oom.default_remedy(),
            Self::RequirementNotMet { requires_one_of, .. } => GpuRemedy::RequirementNotMet(requires_one_of.clone()),
            //Todo! Many of these errors are regarding exceeded device limits.
            _ => GpuRemedy::BlameTheDev
        }
    }
}
impl DefaultRemedy for vulkano::command_buffer::BuildError {
    fn default_remedy(&self) -> GpuRemedy {
        match self {
            Self::OomError(oom) => oom.default_remedy(),
            //The syntax on the builder was wrong...
            Self::QueryActive | Self::RenderPassActive => {
                GpuRemedy::BlameTheDev
            }
        }
    }
}
impl DefaultRemedy for vulkano::command_buffer::CommandBufferBeginError {
    fn default_remedy(&self) -> GpuRemedy {
        match self {
            Self::OomError(oom) => oom.default_remedy(),
            Self::RequirementNotMet { requires_one_of, ..} => GpuRemedy::RequirementNotMet(requires_one_of.clone()),
            //Todo. This is starting to feel like a bad solution x,3
            _ => GpuRemedy::BlameTheDev,
        }
    }
}
pub type GpuResult<T> = Result<T, GpuError>;

pub trait MapGpuErrOrDefault {
    type OkTy;
    type ErrTy : DefaultRemedy;
    /// Provide a function to inspect the underlying error, and determine 
    /// if it's fatal and what remedy should be taken (or None for default remedy for that error)
    fn map_gpu_err_or_default(self, f: impl FnOnce(&Self::ErrTy) -> (bool, Option<GpuRemedy>)) -> GpuResult<Self::OkTy>;
}
pub trait MapGpuErr {
    type OkTy;
    type ErrTy;
    /// Provide a function to inspect the underlying error, and determine 
    /// if it's fatal and what remedy should be taken (or None for default remedy for that error)
    fn map_gpu_err(self, f: impl FnOnce(&Self::ErrTy) -> (bool, GpuRemedy)) -> GpuResult<Self::OkTy>;
}

impl<T: MapGpuErrOrDefault> MapGpuErr for T {
    type OkTy = T::OkTy;
    type ErrTy = T::ErrTy;
    fn map_gpu_err(self, f: impl FnOnce(&Self::ErrTy) -> (bool, GpuRemedy)) -> GpuResult<Self::OkTy> {
        self.map_gpu_err_or_default(|err| {
            let (fatal, remedy) = f(err);
            (fatal, Some(remedy))
        })
    }
}

///This blanket Impl doesn't work in practice. Not sure what type ErrTy should be to allow this to work for
///all Result. (Maybe it cant?)
impl<OkTy, ErrTy> MapGpuErr for std::result::Result<OkTy, ErrTy> {
    type OkTy = OkTy;
    type ErrTy = ErrTy;
    fn map_gpu_err(self, f: impl FnOnce(&Self::ErrTy) -> (bool, GpuRemedy)) -> Result<Self::OkTy, GpuError> {
        self.map_err(|err| {
            let (fatal, remedy) = f(&err);

            GpuError {
                remedy,
                resource_lost: fatal,
                source: None
            }
        })
    }
}

/// Trait to add functionality to vulkano errors, allowing marking as fatal/recoverable
/// and deriving a default plan-of-action for recovery
pub trait VulkanoResult: Sized {
    type OkTy;
    type ErrTy : std::error::Error;

    /// Provide a boolean indicating if an Err result is fatal or not
    fn with_fatality(self, fatal: bool) -> Result<Self::OkTy, GpuError> {
        if fatal {
            VulkanoResult::fatal(self)
        } else {
            VulkanoResult::recoverable(self)
        }
    }
    /// Convert this error to one indicating resource loss. Use this when a failure
    /// means loss of the calling resource.
    fn fatal(self) -> Result<Self::OkTy, GpuError>;

    /// Convert this error to one indicating the resource is not lost.
    fn recoverable(self) -> Result<Self::OkTy, GpuError> {
        VulkanoResult::fatal(self).map_err(
            |err| {
                GpuError {
                    resource_lost: false,
                    ..err
                }
            }
        )
    }
}

/// Implement these functions on all errors for which a default remedy is provided.
impl<OkTy, ErrTy : DefaultRemedy + 'static> VulkanoResult for Result<OkTy, ErrTy> {
    type OkTy = OkTy;
    type ErrTy = ErrTy;
    fn fatal(self) -> Result<Self::OkTy, GpuError> {
        self.map_err(
            |err| GpuError{
                remedy: err.default_remedy(),
                resource_lost: true,
                source: Some(Box::new(err))
            }
        )
    }
}