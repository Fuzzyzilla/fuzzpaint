use crate::vulkano_prelude::*;
use anyhow::Result as AnyResult;
use std::sync::Arc;

/// Check that every index is less than the number of vertices in debug builds, nop in release mode.
/// # Panics
/// on failure.
pub fn debug_assert_indices_safe<Vertex>(vertices: &[Vertex], indices: &[u16]) {
    #[cfg(debug_assertions)]
    if indices
        .iter()
        .any(|&index| usize::from(index) >= vertices.len())
    {
        panic!("index buffer includes invalid indices");
    }
}

/// High level description of how physical device limits translate into
/// limits for fuzzpaint. Note that limits may still be larger than reasonable.
pub struct HighLevelLimits {
    pub max_document_size: [u32; 2],
}
impl HighLevelLimits {
    fn from_device(device: &vk::Device) -> Self {
        let p = device.physical_device().properties();
        // May be greater for specific formats, but this is the guaranteed minimum for all.
        // For now, just use this. (avoids nasty bugs if I forget to update this when I
        // change a format somewhere.. lol)
        let dimension = p.max_image_dimension2_d;
        Self {
            // Don't need to care about max_viewport* as the spec says they are >= max_framebuffer*.
            max_document_size: [
                dimension.min(p.max_framebuffer_width),
                dimension.min(p.max_framebuffer_height),
            ],
        }
    }
}

pub struct Queue {
    queue: Arc<vk::Queue>,
    family_idx: u32,
}
impl Queue {
    #[must_use]
    pub fn idx(&self) -> u32 {
        self.family_idx
    }
    #[must_use]
    pub fn queue(&self) -> &Arc<vk::Queue> {
        &self.queue
    }
}

enum QueueSrc {
    UseGraphics,
    Queue(Queue),
}

#[derive(Clone, Copy, Debug)]
struct QueueIndices {
    /// Also present, if that was required.
    graphics: u32,
    graphics_can_present: bool,
    compute: u32,
}
pub struct Queues {
    graphics_queue: Queue,
    /// Always the same as graphics if so (required by vulkano during present).
    has_present: bool,
    compute_queue: QueueSrc,
}
impl Queues {
    #[must_use]
    pub fn present(&self) -> Option<&Queue> {
        self.has_present.then(|| self.graphics())
    }
    #[must_use]
    pub fn graphics(&self) -> &Queue {
        &self.graphics_queue
    }
    #[must_use]
    pub fn compute(&self) -> &Queue {
        match &self.compute_queue {
            QueueSrc::UseGraphics => self.graphics(),
            QueueSrc::Queue(q) => q,
        }
    }
    //No transfer queues yet, just use Graphics.
    #[must_use]
    pub fn transfer(&self) -> &Queue {
        self.graphics()
    }
    #[must_use]
    pub fn has_unique_compute(&self) -> bool {
        match &self.compute_queue {
            QueueSrc::UseGraphics => false,
            QueueSrc::Queue(..) => true,
        }
    }
    /// Create a sharing object for compute and graphics. If compute and graphics
    /// are the same queue, returns exclusive sharing.
    #[must_use]
    pub fn sharing_compute_graphics<C>(&self) -> vk::Sharing<C>
    where
        C: FromIterator<u32> + IntoIterator<Item = u32>,
    {
        if self.has_unique_compute() {
            vk::Sharing::Concurrent(
                [self.graphics().idx(), self.compute().idx()]
                    .into_iter()
                    .collect(),
            )
        } else {
            vk::Sharing::Exclusive
        }
    }
}

pub struct RenderSurface {
    context: Arc<RenderContext>,
    swapchain: Arc<vk::Swapchain>,
    _surface: Arc<vk::Surface>,
    swapchain_images: Vec<Arc<vk::Image>>,

    swapchain_create_info: vk::SwapchainCreateInfo,
}
impl RenderSurface {
    #[must_use]
    pub fn extent(&self) -> [u32; 2] {
        self.swapchain_create_info.image_extent
    }
    #[must_use]
    pub fn format(&self) -> vk::Format {
        self.swapchain_create_info.image_format
    }
    #[must_use]
    pub fn swapchain(&self) -> &Arc<vk::Swapchain> {
        &self.swapchain
    }
    #[must_use]
    pub fn swapchain_images(&self) -> &[Arc<vk::Image>] {
        &self.swapchain_images
    }
    #[must_use]
    pub fn context(&self) -> &Arc<RenderContext> {
        &self.context
    }
    fn new(
        context: Arc<RenderContext>,
        surface: Arc<vk::Surface>,
        size: [u32; 2],
    ) -> AnyResult<Self> {
        let physical_device = context.physical_device();

        let surface_info = vk::SurfaceInfo::default();
        let capabilies = physical_device.surface_capabilities(&surface, surface_info.clone())?;

        let Some(&(format, color_space)) = physical_device
            .surface_formats(&surface, surface_info)?
            .iter()
            // FIXME!! Find the highest BGRA swapchain format.
            // Used to make bad assumptions about Framebuffer formats later in the code :V
            // What really needs to happen is *whatever* format is chosen (we don't care)
            // needs to be broadcast out and pipelines need to be remade if incompatible.
            // No infrastructure for that at this time.
            .find(|(format, _)| *format == vk::Format::B8G8R8A8_SRGB)
        else {
            return Err(anyhow::anyhow!(
                "Device reported no supported surface formats."
            ));
        };

        //Use mailbox for low-latency, if supported. Otherwise, FIFO is always supported.
        let present_mode = physical_device
            .surface_present_modes(&surface, vulkano::swapchain::SurfaceInfo::default())
            .map(|mut modes| {
                if modes.any(|mode| mode == vk::PresentMode::Mailbox) {
                    vk::PresentMode::Mailbox
                } else {
                    vk::PresentMode::Fifo
                }
            })
            .unwrap_or(vk::PresentMode::Fifo);

        // Use the minimum - Only one frame will be rendered at once.
        let image_count = capabilies.min_image_count;

        // We don't care!
        let alpha_mode = capabilies
            .supported_composite_alpha
            .into_iter()
            .next()
            .expect("Device provided no alpha modes");

        let swapchain_create_info = vk::SwapchainCreateInfo {
            min_image_count: image_count,
            image_format: format,
            image_color_space: color_space,
            image_extent: size,
            image_usage: vk::ImageUsage::COLOR_ATTACHMENT | vk::ImageUsage::TRANSFER_DST,
            composite_alpha: alpha_mode,
            present_mode,
            clipped: true, // We wont read the framebuffer.
            ..Default::default()
        };

        let (swapchain, images) = vk::Swapchain::new(
            context.device().clone(),
            surface.clone(),
            swapchain_create_info.clone(),
        )?;

        Ok(Self {
            context,
            swapchain,
            _surface: surface,
            swapchain_images: images,
            swapchain_create_info,
        })
    }
    pub fn recreate(self, new_size: Option<[u32; 2]>) -> AnyResult<Self> {
        let mut new_info = self.swapchain_create_info;
        if let Some(new_size) = new_size {
            new_info.image_extent = new_size;
        }
        let (swapchain, swapchain_images) = self.swapchain.recreate(new_info.clone())?;

        Ok(Self {
            swapchain,
            swapchain_images,
            swapchain_create_info: new_info,
            ..self
        })
    }
}

pub struct Allocators {
    command_buffer_alloc: vk::StandardCommandBufferAllocator,
    memory_alloc: Arc<dyn vulkano::memory::allocator::MemoryAllocator>,
    descriptor_set_alloc: vk::StandardDescriptorSetAllocator,
}

impl Allocators {
    pub fn command_buffer(&self) -> &vk::StandardCommandBufferAllocator {
        &self.command_buffer_alloc
    }
    pub fn memory(&self) -> &Arc<dyn vulkano::memory::allocator::MemoryAllocator> {
        &self.memory_alloc
    }
    pub fn descriptor_set(&self) -> &vk::StandardDescriptorSetAllocator {
        &self.descriptor_set_alloc
    }
}

pub struct RenderContext {
    _library: Arc<vk::VulkanLibrary>,
    _instance: Arc<vk::Instance>,
    physical_device: Arc<vk::PhysicalDevice>,
    high_level_limits: HighLevelLimits,
    device: Arc<vk::Device>,
    queues: Queues,

    _debugger: Option<vulkano::instance::debug::DebugUtilsMessenger>,

    allocators: Allocators,
}

impl RenderContext {
    pub fn new_headless() -> AnyResult<Self> {
        unimplemented!()
    }
    pub fn new_with_window_surface(
        win: &crate::window::Surface,
    ) -> AnyResult<(Arc<Self>, RenderSurface)> {
        use vulkano::instance::debug as vkDebug;

        let library = vk::VulkanLibrary::new()?;

        let mut required_instance_extensions = vk::Surface::required_extensions(win.event_loop());
        required_instance_extensions.ext_debug_utils = true;

        let instance = vk::Instance::new(
            library.clone(),
            vk::InstanceCreateInfo {
                application_name: Some(option_env!("CARGO_PKG_NAME").unwrap_or("").to_string()),
                application_version: vk::Version {
                    major: option_env!("CARGO_PKG_VERSION_MAJOR")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0),
                    minor: option_env!("CARGO_PKG_VERSION_MINOR")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0),
                    patch: option_env!("CARGO_PKG_VERSION_PATCH")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0),
                },
                enabled_extensions: required_instance_extensions,
                ..Default::default()
            },
        )?;

        let debugger = vkDebug::DebugUtilsMessenger::new(
            instance.clone(),
            vkDebug::DebugUtilsMessengerCreateInfo {
                message_severity: vkDebug::DebugUtilsMessageSeverity::ERROR
                    | vkDebug::DebugUtilsMessageSeverity::WARNING
                    | vkDebug::DebugUtilsMessageSeverity::INFO
                    | vkDebug::DebugUtilsMessageSeverity::VERBOSE,
                message_type: vkDebug::DebugUtilsMessageType::GENERAL
                    | vkDebug::DebugUtilsMessageType::PERFORMANCE
                    | vkDebug::DebugUtilsMessageType::VALIDATION,
                ..vkDebug::DebugUtilsMessengerCreateInfo::user_callback(
                    // SAFETY: the closure must not access vulkan API in any way.
                    // Not a problem, as it simply logs to console or file, depending on log target.
                    // In the future when this prints to an internal log however, I must keep
                    // this in mind!
                    unsafe {
                        vulkano::instance::debug::DebugUtilsMessengerCallback::new(
                            |severity, ty, data| {
                                #[allow(clippy::wildcard_in_or_patterns)]
                                let level = match severity {
                                    vkDebug::DebugUtilsMessageSeverity::ERROR => log::Level::Error,
                                    vkDebug::DebugUtilsMessageSeverity::WARNING => log::Level::Warn,
                                    vkDebug::DebugUtilsMessageSeverity::VERBOSE => {
                                        log::Level::Trace
                                    }
                                    vkDebug::DebugUtilsMessageSeverity::INFO | _ => {
                                        log::Level::Info
                                    }
                                };
                                let ty = match ty {
                                    vkDebug::DebugUtilsMessageType::GENERAL => "GENERAL",
                                    vkDebug::DebugUtilsMessageType::PERFORMANCE => "PERFORMANCE",
                                    vkDebug::DebugUtilsMessageType::VALIDATION => "VALIDATION",
                                    _ => "UNKNOWN",
                                };
                                let layer = data.message_id_name.unwrap_or("");

                                log::log!(target: "vulkan", level, "[{ty}] {layer} - {}", data.message);
                            },
                        )
                    },
                )
            },
        )?;

        let surface = vk::Surface::from_window(instance.clone(), win.window())?;
        let required_device_extensions = vk::DeviceExtensions {
            khr_swapchain: true,
            ext_line_rasterization: true,
            ..Default::default()
        };
        // Extensions needed when the physical device supports less than vulkan 1.3
        let required_device_extensions_lt_1_3 = vk::DeviceExtensions {
            // Promoted to core in 1.3.
            khr_dynamic_rendering: true,
            ..Default::default()
        };

        let Some((physical_device, queue_indices)) = Self::choose_physical_device(
            &instance,
            &required_device_extensions,
            &required_device_extensions_lt_1_3,
            Some(&surface),
        )?
        else {
            return Err(anyhow::anyhow!("Failed to find a suitable Vulkan device."));
        };

        log::info!(
            "Chose physical device {} ({:?})",
            physical_device.properties().device_name,
            physical_device.properties().driver_info
        );

        let (device, queues) = Self::create_device(
            physical_device.clone(),
            queue_indices,
            &required_device_extensions,
            &required_device_extensions_lt_1_3,
        )?;

        // We have a device! Now to create the swapchain..
        let image_size = win.window().inner_size();

        let context = Arc::new(Self {
            allocators: Allocators {
                command_buffer_alloc: vk::StandardCommandBufferAllocator::new(
                    device.clone(),
                    Default::default(),
                ),
                memory_alloc: Arc::new(vk::StandardMemoryAllocator::new_default(device.clone())),
                descriptor_set_alloc: vk::StandardDescriptorSetAllocator::new(
                    device.clone(),
                    vulkano::descriptor_set::allocator::StandardDescriptorSetAllocatorCreateInfo {
                        update_after_bind: false,
                        ..Default::default()
                    },
                ),
            },
            high_level_limits: HighLevelLimits::from_device(&device),
            _library: library,
            _instance: instance,
            device,
            physical_device,
            queues,

            _debugger: Some(debugger),
        });
        let render_surface =
            RenderSurface::new(context.clone(), surface.clone(), image_size.into())?;

        Ok((context, render_surface))
    }
    fn create_device(
        physical_device: Arc<vk::PhysicalDevice>,
        queue_indices: QueueIndices,
        extensions: &vk::DeviceExtensions,
        extensions_lt_1_3: &vk::DeviceExtensions,
    ) -> AnyResult<(Arc<vk::Device>, Queues)> {
        //Need a graphics queue.
        let mut graphics_queue_info = vk::QueueCreateInfo {
            queue_family_index: queue_indices.graphics,
            queues: vec![0.5],
            ..Default::default()
        };

        //Todo: what if compute and present end up in the same family, aside from graphics? Unlikely :)
        // ^^ Pfft, tell that to the future asp who spent two hours debugging a similar situation
        let (has_unique_compute_queue, compute_queue_info) = {
            if queue_indices.compute == queue_indices.graphics {
                let capacity = physical_device.queue_family_properties()
                    [graphics_queue_info.queue_family_index as usize]
                    .queue_count;
                //Is there room for another queue?
                if capacity as usize > graphics_queue_info.queues.len() {
                    graphics_queue_info.queues.push(0.5);

                    //In the graphics queue family. No queue info.
                    (true, None)
                } else {
                    //Share a queue with graphics.
                    (false, None)
                }
            } else {
                (
                    true,
                    Some(vk::QueueCreateInfo {
                        queue_family_index: queue_indices.compute,
                        queues: vec![0.5],
                        ..Default::default()
                    }),
                )
            }
        };

        let mut create_infos = vec![graphics_queue_info];

        if let Some(compute_create_info) = compute_queue_info {
            create_infos.push(compute_create_info);
        }

        let enabled_extensions = if physical_device.api_version() < vk::Version::V1_3 {
            extensions.union(extensions_lt_1_3)
        } else {
            *extensions
        };

        let (device, mut queues) = vk::Device::new(
            physical_device,
            vk::DeviceCreateInfo {
                enabled_extensions,
                enabled_features: vk::Features {
                    dual_src_blend: true,
                    dynamic_rendering: true,
                    multi_draw_indirect: true,
                    maintenance4: true,
                    geometry_shader: true,
                    ..vk::Features::empty()
                },
                queue_create_infos: create_infos,
                ..Default::default()
            },
        )?;

        // Queues will be in order of create info, so this depends on the layout created by the `if` statements above.
        let graphics_queue = Queue {
            queue: queues.next().unwrap(),
            family_idx: queue_indices.graphics,
        };
        let compute_queue = if has_unique_compute_queue {
            QueueSrc::Queue(Queue {
                queue: queues.next().unwrap(),
                family_idx: queue_indices.compute,
            })
        } else {
            QueueSrc::UseGraphics
        };

        assert!(queues.next().is_none());

        Ok((
            device,
            Queues {
                graphics_queue,
                has_present: queue_indices.graphics_can_present,
                compute_queue,
            },
        ))
    }
    /// Find a device that fits our needs, including the ability to present to the surface if in non-headless mode.
    /// Horrible signature - Returns Ok(None) if no device found, Ok(Some((device, queue indices))) if suitable device found.
    fn choose_physical_device(
        instance: &Arc<vk::Instance>,
        required_extensions: &vk::DeviceExtensions,
        required_extensions_lt_1_3: &vk::DeviceExtensions,
        compatible_surface: Option<&vk::Surface>,
    ) -> AnyResult<Option<(Arc<vk::PhysicalDevice>, QueueIndices)>> {
        //TODO: does not respect queue family max queue counts. This will need to be redone in some sort of
        //multi-pass shenanigan to properly find a good queue setup. Also requires that graphics and compute queues be transfer as well.
        let res = instance
            .enumerate_physical_devices()?
            .filter_map(|device| {
                use vk::QueueFlags;
                let required_extensions = if device.api_version() < vk::Version::V1_3 {
                    required_extensions.union(required_extensions_lt_1_3)
                } else {
                    *required_extensions
                };
                //Make sure it has what we need
                if !device.supported_extensions().contains(&required_extensions) {
                    return None;
                }

                let families = device.queue_family_properties();

                //Find a queue that supports the requested surface, if any
                let present_queue = compatible_surface.and_then(|surface| {
                    families.iter().enumerate().find(|(family_idx, _)| {
                        //Assume error is false. Todo?
                        device
                            .surface_support(*family_idx as u32, surface)
                            .unwrap_or(false)
                    })
                });

                //We needed a present queue, but none was found. Disqualify this device!
                if compatible_surface.is_some() && present_queue.is_none() {
                    return None;
                }

                // We need a graphics queue, always! Otherwise, disqualify.
                // If we require present to the surface, ensure that this queue can also do it.
                let graphics_queue = families.iter().enumerate().find(|q| {
                    let is_graphics = q.1.queue_flags.contains(QueueFlags::GRAPHICS);
                    let can_present = compatible_surface.is_none()
                        || device
                            .surface_support(q.0 as u32, compatible_surface.unwrap())
                            .unwrap_or(false);
                    is_graphics && can_present
                })?;

                //We need a compute queue. This can be the same as graphics, but preferably not.
                let graphics_supports_compute =
                    graphics_queue.1.queue_flags.contains(QueueFlags::COMPUTE);

                //Find a different queue that supports compute
                let compute_queue = families
                    .iter()
                    .enumerate()
                    //Ignore the family we chose for graphics
                    .filter(|&(idx, _)| idx != graphics_queue.0)
                    .find(|q| q.1.queue_flags.contains(QueueFlags::COMPUTE));

                //Failed to find compute queue, shared or otherwise. Disqualify!
                if !graphics_supports_compute && compute_queue.is_none() {
                    return None;
                }

                Some((
                    device.clone(),
                    QueueIndices {
                        compute: compute_queue.unwrap_or(graphics_queue).0 as u32,
                        // Would have bailed if not!
                        graphics_can_present: compatible_surface.is_some(),
                        graphics: graphics_queue.0 as u32,
                    },
                ))
            })
            .min_by_key(|(device, _)| {
                use vk::PhysicalDeviceType;
                match device.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,

                    _ => 3,
                }
            });

        Ok(res)
    }
    pub fn now(&self) -> vk::NowFuture {
        vk::sync::now(self.device.clone())
    }
    pub fn physical_device(&self) -> &Arc<vk::PhysicalDevice> {
        &self.physical_device
    }
    pub fn queues(&self) -> &Queues {
        &self.queues
    }
    pub fn device(&self) -> &Arc<vk::Device> {
        &self.device
    }
    pub fn allocators(&self) -> &Allocators {
        &self.allocators
    }
    pub fn high_level_limits(&self) -> &HighLevelLimits {
        &self.high_level_limits
    }
}
