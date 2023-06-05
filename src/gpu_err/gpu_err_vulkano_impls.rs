use super::gpu_err::*;

impl From<vulkano::render_pass::RenderPassCreationError> for GpuErrorSource {
    fn from(value: vulkano::render_pass::RenderPassCreationError) -> Self {
        use vulkano::render_pass::RenderPassCreationError;
        match value {
            RenderPassCreationError::OomError(oom) => oom.into(),
            RenderPassCreationError::RequirementNotMet { required_for, requires_one_of } =>
                Self::RequirementNotMet { required_for, requires_one_of },
            _ => Self::RenderpassCreationError(value),
        }
    }
}
impl HasOom for vulkano::render_pass::RenderPassCreationError {
    fn oom(&self) -> Option<vulkano::OomError> {
        use vulkano::render_pass::RenderPassCreationError;
        match self {
            RenderPassCreationError::OomError(oom) => Some(*oom),
            _ => None
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