use super::gpu_err::*;

impl From<vulkano::OomError> for GpuErrorSource{
    fn from(value: vulkano::OomError) -> Self {
        Self::OomError(value)
    }
}
impl HasOom for vulkano::OomError {
    fn oom(&self) -> Option<vulkano::OomError> {
        Some(*self)
    }
}

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
impl HasDeviceLoss for vulkano::sync::fence::FenceError {
    fn device_lost(&self) -> bool {
        *self == vulkano::sync::fence::FenceError::DeviceLost
    }
}
impl HasOom for vulkano::sync::fence::FenceError {
    fn oom(&self) -> Option<vulkano::OomError> {
        use vulkano::sync::fence::FenceError;
        match self {
            FenceError::OomError(oom) => Some(*oom),
            _ => None
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
impl HasOom for vulkano::sync::semaphore::SemaphoreError {
    fn oom(&self) -> Option<vulkano::OomError> {
        use vulkano::sync::semaphore::SemaphoreError;
        match self {
            SemaphoreError::OomError(oom) => Some(*oom),
            _ => None
        }
    }
}

impl From<vulkano::shader::ShaderCreationError> for GpuErrorSource {
    fn from(value: vulkano::shader::ShaderCreationError) -> Self {
        use vulkano::shader::ShaderCreationError;
        match value {
            ShaderCreationError::OomError(oom) => oom.into(),
            _ => Self::ShaderCreationError(value)
        }
        
    }
}
impl HasOom for vulkano::shader::ShaderCreationError {
    fn oom(&self) -> Option<vulkano::OomError> {
        use vulkano::shader::ShaderCreationError;
        match self {
            ShaderCreationError::OomError(oom) => Some(*oom),
            _ => None
        }
    }
}

impl From<vulkano::pipeline::graphics::GraphicsPipelineCreationError> for GpuErrorSource {
    fn from(value: vulkano::pipeline::graphics::GraphicsPipelineCreationError) -> Self {
        use vulkano::pipeline::graphics::GraphicsPipelineCreationError;
        match value {
            GraphicsPipelineCreationError::OomError(oom) => oom.into(),
            _ => Self::GraphicsPipelineCreationError(value)
        }
        
    }
}
impl HasOom for vulkano::pipeline::graphics::GraphicsPipelineCreationError {
    fn oom(&self) -> Option<vulkano::OomError> {
        use vulkano::pipeline::graphics::GraphicsPipelineCreationError;
        match self {
            GraphicsPipelineCreationError::OomError(oom) => Some(*oom),
            _ => None
        }
    }
}

impl From<vulkano::command_buffer::CommandBufferBeginError> for GpuErrorSource{
    fn from(value: vulkano::command_buffer::CommandBufferBeginError) -> Self {
        use vulkano::command_buffer::CommandBufferBeginError;
        match value {
            CommandBufferBeginError::OomError(oom) => Self::OomError(oom),
            CommandBufferBeginError::RequirementNotMet { required_for, requires_one_of } =>
                GpuErrorSource::RequirementNotMet { required_for, requires_one_of},
            _ => {
                todo!("CommandBufferBeginError source");
            }
        }
    }
}
impl HasOom for vulkano::command_buffer::CommandBufferBeginError {
    fn oom(&self) -> Option<vulkano::OomError> {
        use vulkano::command_buffer::CommandBufferBeginError;
        match self {
            CommandBufferBeginError::OomError(oom) => Some(*oom),
            _ => None
        }
    }
}

impl From<vulkano::command_buffer::BuildError> for GpuErrorSource{
    fn from(value: vulkano::command_buffer::BuildError) -> Self {
        use vulkano::command_buffer::BuildError;
        match value {
            _ => {
                todo!("BuildError source");
            }
        }
    }
}

impl From<vulkano::buffer::BufferError> for GpuErrorSource{
    fn from(value: vulkano::buffer::BufferError) -> Self {
        use vulkano::buffer::BufferError;
        match value {
            _ => {
                todo!("BufferError source")
            }
        }
    }
}

impl From<vulkano::command_buffer::RenderPassError> for GpuErrorSource{
    fn from(value: vulkano::command_buffer::RenderPassError) -> Self {
        use vulkano::command_buffer::RenderPassError;
        match value {
            RenderPassError::RequirementNotMet { required_for, requires_one_of } =>
                Self::RequirementNotMet { required_for, requires_one_of },
            _ => {
                todo!("RenderPassError source")
            }
        }
    }
}


impl From<vulkano::command_buffer::PipelineExecutionError> for GpuErrorSource{
    fn from(value: vulkano::command_buffer::PipelineExecutionError) -> Self {
        use vulkano::command_buffer::PipelineExecutionError;
        match value {
            PipelineExecutionError::RequirementNotMet { required_for, requires_one_of } =>
                Self::RequirementNotMet { required_for, requires_one_of },
            _ => {
                todo!("PipelineExecutionError source")
            }
        }
    }
}

impl From<vulkano::image::ImageError> for GpuErrorSource{
    fn from(value: vulkano::image::ImageError) -> Self {
        use vulkano::image::ImageError;
        match value {
            ImageError::RequirementNotMet { required_for, requires_one_of } =>
                Self::RequirementNotMet { required_for, requires_one_of },
            _ => {
                todo!("ImageError source")
            }
        }
    }
}

impl From<vulkano::image::view::ImageViewCreationError> for GpuErrorSource{
    fn from(value: vulkano::image::view::ImageViewCreationError) -> Self {
        use vulkano::image::view::ImageViewCreationError;
        match value {
            ImageViewCreationError::OomError(oom) => Self::OomError(oom),
            ImageViewCreationError::RequirementNotMet { required_for, requires_one_of } =>
                Self::RequirementNotMet { required_for, requires_one_of },
            _ => {
                todo!("ImageViewCreationError source")
            }
        }
    }
}
impl HasOom for vulkano::image::view::ImageViewCreationError {
    fn oom(&self) -> Option<vulkano::OomError> {
        if let vulkano::image::view::ImageViewCreationError::OomError(oom) = self {
            Some(*oom)
        } else {
            None
        }
    }

}

impl From<vulkano::sampler::SamplerCreationError> for GpuErrorSource{
    fn from(value: vulkano::sampler::SamplerCreationError) -> Self {
        use vulkano::sampler::SamplerCreationError;
        match value {
            SamplerCreationError::OomError(oom) => Self::OomError(oom),
            SamplerCreationError::RequirementNotMet { required_for, requires_one_of } =>
                Self::RequirementNotMet { required_for, requires_one_of },
            _ => {
                todo!("SamplerCreationError source")
            }
        }
    }
}
impl HasOom for vulkano::sampler::SamplerCreationError {
    fn oom(&self) -> Option<vulkano::OomError> {
        if let vulkano::sampler::SamplerCreationError::OomError(oom) = self {
            Some(*oom)
        } else {
            None
        }
    }
}

impl From<vulkano::descriptor_set::DescriptorSetCreationError> for GpuErrorSource{
    fn from(value: vulkano::descriptor_set::DescriptorSetCreationError) -> Self {
        use vulkano::descriptor_set::DescriptorSetCreationError;
        match value {
            DescriptorSetCreationError::OomError(oom) => Self::OomError(oom),
            _ => {
                todo!("DescriptorSetCreationError source")
            }
        }
    }
}
impl HasOom for vulkano::descriptor_set::DescriptorSetCreationError {
    fn oom(&self) -> Option<vulkano::OomError> {
        if let vulkano::descriptor_set::DescriptorSetCreationError::OomError(oom) = self {
            Some(*oom)
        } else {
            None
        }
    }
}

impl From<vulkano::command_buffer::CopyError> for GpuErrorSource{
    fn from(value: vulkano::command_buffer::CopyError) -> Self {
        use vulkano::command_buffer::CopyError;
        match value {
            CopyError::RequirementNotMet { required_for, requires_one_of } =>
                Self::RequirementNotMet { required_for, requires_one_of },
            _ => {
                todo!("CopyError source")
            }
        }
    }
} 