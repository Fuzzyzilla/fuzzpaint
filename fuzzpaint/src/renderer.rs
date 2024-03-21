mod gpu_tess;
pub mod picker;
pub mod requests;
mod stroke_batcher;

use std::sync::Arc;

use crate::vulkano_prelude::*;

struct PerDocumentData {
    listener: crate::commands::queue::DocumentCommandListener,
    /// Cached images of each of the nodes of the graph.
    graph_render_data: hashbrown::HashMap<crate::state::graph::AnyID, RenderData>,
}
#[derive(thiserror::Error, Debug)]
enum IncrementalDrawErr {
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
    /// State was not usable for incremental draw.
    /// Draw from scratch instead!
    #[error("state mismatch")]
    StateMismatch,
}
#[allow(dead_code)]
enum CachedImage<'data> {
    /// The data is ready for use immediately.
    Ready(&'data RenderData),
    /// The data is currently in-flight on the graphics device, and will become ready once the provided
    /// `fence` becomes signalled.
    ReadyAfter {
        image: &'data RenderData,
        fence: &'data vk::sync::future::FenceSignalFuture<Box<dyn vk::sync::GpuFuture>>,
    },
}
impl<'data> CachedImage<'data> {
    #[allow(dead_code)]
    fn data(&self) -> &'data RenderData {
        match self {
            CachedImage::Ready(data) => data,
            CachedImage::ReadyAfter { image, .. } => image,
        }
    }
}
struct Renderer {
    context: Arc<crate::render_device::RenderContext>,
    stroke_renderer: stroke_renderer::StrokeLayerRenderer,
    text_builder: crate::text::TextBuilder,
    text_renderer: crate::text::renderer::monochrome::Renderer,
    blend_engine: crate::blend::BlendEngine,
    data: hashbrown::HashMap<crate::state::DocumentID, PerDocumentData>,
}
impl Renderer {
    fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
        Ok(Self {
            context: context.clone(),
            blend_engine: crate::blend::BlendEngine::new(context.device())?,
            text_builder: crate::text::TextBuilder::allocate_new(
                context.allocators().memory().clone(),
            )?,
            text_renderer: crate::text::renderer::monochrome::Renderer::new(context.clone())?,
            stroke_renderer: stroke_renderer::StrokeLayerRenderer::new(context)?,
            data: hashbrown::HashMap::new(),
        })
    }
    fn render_one(
        &mut self,
        id: crate::state::DocumentID,
        into: &Arc<vk::ImageView>,
    ) -> anyhow::Result<()> {
        let data = self.data.entry(id);
        // Get the document data, and a flag for if we need to initialize that data.
        let (is_new, data) = match data {
            hashbrown::hash_map::Entry::Occupied(o) => (false, o.into_mut()),
            hashbrown::hash_map::Entry::Vacant(v) => {
                let Some(listener) = crate::default_provider().inspect(
                    id,
                    crate::commands::queue::DocumentCommandQueue::listen_from_now,
                ) else {
                    // Deleted before we could do anything.
                    anyhow::bail!("Document deleted before render worker reached it");
                };
                (
                    true,
                    v.insert(PerDocumentData {
                        listener,
                        graph_render_data: hashbrown::HashMap::new(),
                    }),
                )
            }
        };
        // Forward the listener state.
        let changes = match data.listener.forward_clone_state() {
            Ok(changes) => changes,
            Err(e) => {
                // Destroy the render data, report the error.
                // Could be closed, or a thrashed document state D:
                self.data.remove(&id);
                return Err(e.into());
            }
        };
        // Render from scratch if we just created the data,
        // otherwise update from previous state.
        if is_new {
            Self::draw_from_scratch(
                &self.context,
                &self.blend_engine,
                &self.stroke_renderer,
                &mut self.text_builder,
                &self.text_renderer,
                data,
                &changes,
                into,
            )
        } else {
            // Try to draw incrementally. If that reports it's impossible, try
            // to draw from scratch.
            match Self::draw_incremental(
                &self.context,
                &self.blend_engine,
                &self.stroke_renderer,
                data,
                &changes,
                into,
            ) {
                Err(IncrementalDrawErr::StateMismatch) => {
                    log::info!("Incremental draw failed! Retrying from scratch...");
                    Self::draw_from_scratch(
                        &self.context,
                        &self.blend_engine,
                        &self.stroke_renderer,
                        &mut self.text_builder,
                        &self.text_renderer,
                        data,
                        &changes,
                        into,
                    )
                }
                Err(IncrementalDrawErr::Anyhow(anyhow)) => Err(anyhow),
                Ok(()) => Ok(()),
            }
        }
    }
    /// Draws the entire state from the beginning, ignoring the diff.
    /// Reuses allocated images, but ignores their contents!
    fn draw_from_scratch(
        context: &Arc<crate::render_device::RenderContext>,
        blend_engine: &crate::blend::BlendEngine,
        renderer: &stroke_renderer::StrokeLayerRenderer,
        text_builder: &mut crate::text::TextBuilder,
        text_renderer: &crate::text::renderer::monochrome::Renderer,
        document_data: &mut PerDocumentData,
        state: &impl crate::commands::queue::state_reader::CommandQueueStateReader,
        into: &Arc<vk::ImageView>,
    ) -> anyhow::Result<()> {
        use crate::state::graph::{LeafType, NodeID, NodeType};

        /// Insert a single node (possibly recursing) into the builder.
        fn insert_blend(
            blend_engine: &crate::blend::BlendEngine,
            builder: &mut crate::blend::BlendInvocationBuilder,
            document_data: &PerDocumentData,
            graph: &crate::state::graph::BlendGraph,

            id: crate::state::graph::AnyID,
            data: &crate::state::graph::NodeData,
        ) -> anyhow::Result<()> {
            match (data.leaf(), data.node()) {
                // Pre-rendered leaves
                (
                    Some(
                        LeafType::SolidColor { blend, .. }
                        | LeafType::StrokeLayer { blend, .. }
                        | LeafType::Text { blend, .. },
                    ),
                    None,
                ) => {
                    let view = document_data
                        .graph_render_data
                        .get(&id)
                        .unwrap()
                        .view
                        .clone();
                    builder.then_blend(crate::blend::BlendImageSource::Immediate(view), *blend)?;
                }
                (Some(LeafType::Note), None) => (),
                // Passthrough - add children directly without grouped blend
                (None, Some(NodeType::Passthrough)) => {
                    blend_for_passthrough(
                        blend_engine,
                        builder,
                        document_data,
                        graph,
                        id.try_into().unwrap(),
                    )?;
                }
                // Grouped blend - add children to a new blend worker.
                (None, Some(NodeType::GroupedBlend(blend))) => {
                    let handle = blend_for_node(
                        blend_engine,
                        document_data,
                        graph,
                        id.try_into().unwrap(),
                        document_data
                            .graph_render_data
                            .get(&id)
                            .unwrap()
                            .view
                            .clone(),
                        true,
                    )?;
                    builder.then_blend(handle.into(), *blend)?;
                }
                // Invalid states
                (Some(_), Some(_)) | (None, None) => unreachable!(),
            }
            Ok(())
        }

        /// Recursively add children into existing blend builder.
        fn blend_for_passthrough(
            blend_engine: &crate::blend::BlendEngine,
            builder: &mut crate::blend::BlendInvocationBuilder,
            document_data: &PerDocumentData,
            graph: &crate::state::graph::BlendGraph,
            node: NodeID,
        ) -> anyhow::Result<()> {
            let iter = graph
                .iter_node(node)
                .ok_or_else(|| anyhow::anyhow!("Passthrough node not found"))?;
            for (id, data) in iter {
                insert_blend(blend_engine, builder, document_data, graph, id, data)?;
            }
            Ok(())
        }

        /// Recursively build a blend invocation for the node, or None for root.
        fn blend_for_node(
            blend_engine: &crate::blend::BlendEngine,
            document_data: &PerDocumentData,
            graph: &crate::state::graph::BlendGraph,
            node: NodeID,

            into_image: Arc<vk::ImageView>,
            clear_image: bool,
        ) -> anyhow::Result<crate::blend::BlendInvocationHandle> {
            let iter = graph
                .iter_node(node)
                .ok_or_else(|| anyhow::anyhow!("Node not found"))?;
            let mut builder = blend_engine.start(into_image, clear_image);

            for (id, data) in iter {
                insert_blend(blend_engine, &mut builder, document_data, graph, id, data)?;
            }

            // We traverse top-down, we need to blend bottom-up
            builder.reverse();
            Ok(builder.build())
        }
        // Create/discard images
        Self::allocate_prune_graph(
            renderer,
            &mut document_data.graph_render_data,
            state.graph(),
        )?;

        let mut fences = Vec::<vk::FenceSignalFuture<Box<dyn GpuFuture>>>::new();

        // Walk the tree in arbitrary order, rendering all as needed and collecting their futures.
        for (id, data) in state.graph().iter() {
            match data.leaf() {
                Some(LeafType::SolidColor { source, .. }) => {
                    let image = document_data.graph_render_data.get(&id).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Expected image to be created by allocate_prune_graph for {id:?}"
                        )
                    })?;
                    // Fill image, store semaphore.
                    fences.push(Self::render_color(context.as_ref(), image, *source)?);
                }
                // Render stroke image
                Some(LeafType::StrokeLayer { collection, .. }) => {
                    let image = document_data.graph_render_data.get(&id).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Expected image to be created by allocate_prune_graph for {id:?}"
                        )
                    })?;
                    let strokes = state.stroke_collections().get(*collection).ok_or_else(|| {
                        anyhow::anyhow!("Missing stroke collection {collection:?}")
                    })?;
                    let strokes: Vec<_> = strokes.iter_active().collect();
                    if strokes.is_empty() {
                        //FIXME: Renderer doesn't know how to handle zero strokes.
                        fences.push(Self::render_color(
                            context.as_ref(),
                            image,
                            [0.0, 0.0, 0.0, 0.0],
                        )?);
                    } else {
                        renderer.draw(strokes.as_ref(), image, true)?;
                    }
                }
                // Render stroke image
                Some(LeafType::Text {
                    text, px_per_em, ..
                }) => {
                    let image = document_data.graph_render_data.get(&id).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Expected image to be created by allocate_prune_graph for {id:?}"
                        )
                    })?;
                    fences.push(Self::render_text(
                        context.as_ref(),
                        text_builder,
                        text_renderer,
                        image,
                        *px_per_em,
                        text,
                    )?);
                }
                Some(LeafType::Note) | None => (),
            }
        }

        let mut top_level_blend = blend_engine.start(into.clone(), true);
        let graph = state.graph();
        // Walk the tree in tree-order, building up a blend operation.
        for (id, data) in graph.iter_top_level() {
            insert_blend(
                blend_engine,
                &mut top_level_blend,
                document_data,
                graph,
                id,
                data,
            )?;
        }
        // We traverse top-down, we need to blend bottom-up
        top_level_blend.reverse();
        let top_level_blend = top_level_blend.build();

        // Wait for every fence. Terrible, but vulkano semaphores don't seem to be working currently.
        // Note to self: see commit fuzzpaint @ d435ca7c29cf045be413c9849be928693a2de458 for a time when this worked.
        // Iunno what changed :<
        for fence in fences {
            fence.wait(None)?;
        }

        // Execute blend after the images are ready
        blend_engine.submit(context.as_ref(), top_level_blend)
    }
    /// Assumes the existence of a previous `draw_from_scratch`, applying only the diff.
    fn draw_incremental(
        _context: &Arc<crate::render_device::RenderContext>,
        _blend_engine: &crate::blend::BlendEngine,
        _renderer: &stroke_renderer::StrokeLayerRenderer,
        _document_data: &mut PerDocumentData,
        state: &impl crate::commands::queue::state_reader::CommandQueueStateReader,
        _into: &Arc<vk::ImageView>,
    ) -> Result<(), IncrementalDrawErr> {
        if state.has_changes() {
            // State is dirty!
            // Lol, just defer to draw_from_scratch until that works.
            Err(IncrementalDrawErr::StateMismatch)
        } else {
            // Nothing to do
            Ok(())
        }
    }
    fn render_color(
        context: &crate::render_device::RenderContext,
        image: &RenderData,
        color: [f32; 4],
    ) -> anyhow::Result<vk::FenceSignalFuture<Box<dyn GpuFuture>>> {
        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            context.allocators().command_buffer(),
            // Unfortunately transfer queue cannot clear images TwT
            // Here, graphics is more likely to be idle than compute.
            context.queues().graphics().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer.clear_color_image(vk::ClearColorImageInfo {
            clear_value: color.into(),
            regions: smallvec::smallvec![image.view.subresource_range().clone()],
            image_layout: vk::ImageLayout::General,
            ..vk::ClearColorImageInfo::image(image.image.clone())
        })?;

        let command_buffer = command_buffer.build()?;

        Ok(vk::sync::now(context.device().clone())
            .then_execute(context.queues().graphics().queue().clone(), command_buffer)?
            .boxed()
            .then_signal_fence_and_flush()?)
    }
    fn render_text(
        context: &crate::render_device::RenderContext,
        builder: &mut crate::text::TextBuilder,
        renderer: &crate::text::renderer::monochrome::Renderer,
        image: &RenderData,
        px_per_em: f32,
        text: &str,
    ) -> anyhow::Result<vk::FenceSignalFuture<Box<dyn GpuFuture>>> {
        static FACE: std::sync::OnceLock<(rustybuzz::Face<'static>, rustybuzz::ShapePlan)> =
            std::sync::OnceLock::new();
        // There is currently no font server.
        let face_data: &'static [u8] = unimplemented!();
        // const FACE_DATA: &[u8] = include_bytes!("/usr/share/fonts/open-sans/OpenSans-Regular.ttf");

        let (face, plan) = FACE.get_or_init(|| {
            let face = rustybuzz::Face::from_slice(face_data, 0).expect("bad face");
            println!(
                "{:#?}",
                face.variation_axes().into_iter().collect::<Vec<_>>()
            );
            let plan = rustybuzz::ShapePlan::new(
                &face,
                rustybuzz::Direction::LeftToRight,
                Some(rustybuzz::script::LATIN),
                None,
                &[],
            );

            (face, plan)
        });
        let units_per_em = f32::from(face.as_ref().units_per_em());
        let px_per_unit = px_per_em / units_per_em;
        let size_class = crate::text::SizeClass::from_scale_factor(px_per_unit)
            .unwrap_or(crate::text::SizeClass::ONE)
            .saturating_mul(renderer.internal_size_class());
        let mut xform = ultraviolet::Similarity2 {
            scale: px_per_unit,
            ..Default::default()
        };
        let proj = ultraviolet::Similarity2 {
            // map 0..DOCUMENT_DIMENSION to 0.0..2.0
            scale: 2.0 / (crate::DOCUMENT_DIMENSION as f32),
            // map 0.0..2.0 to -1.0..1.0 (NDC)
            translation: ultraviolet::Vec2 { x: -1.0, y: -1.0 },
            ..Default::default()
        };
        // This is not behaving. Dont wanna fix it right now. grr.
        xform.append_similarity(proj);

        let output = builder.tess_draw_multiline(
            face,
            plan,
            size_class,
            &crate::text::MultilineInfo {
                text,
                language: None,
                script: Some(rustybuzz::script::LATIN),
                main_direction: rustybuzz::Direction::LeftToRight,
                line_spacing_mul: 1.0,
                main_align: crate::text::Align::Center,
                cross_direction: rustybuzz::Direction::TopToBottom,
            },
            [0.0, 0.0, 0.0, 1.0],
        )?;
        let commands = renderer.draw(
            xform.into_homogeneous_matrix().into_homogeneous(),
            image.view.clone(),
            &output,
        )?;
        context
            .now()
            .then_execute(context.queues().graphics().queue().clone(), commands)?
            .boxed()
            .then_signal_fence_and_flush()
            .map_err(Into::into)
    }
    /// Creates images for all nodes which require rendering, drops node images that are deleted, etc.
    /// Only fails when graphics device is out-of-memory
    fn allocate_prune_graph(
        renderer: &stroke_renderer::StrokeLayerRenderer,
        graph_render_data: &mut hashbrown::HashMap<crate::state::graph::AnyID, RenderData>,
        graph: &crate::state::graph::BlendGraph,
    ) -> anyhow::Result<()> {
        let mut retain_data = hashbrown::HashSet::new();
        for (id, node) in graph.iter() {
            #[allow(clippy::match_same_arms)]
            let has_graphics = match (node.leaf(), node.node()) {
                // We expect it to be a node xor leaf!
                // This is an api issue ;w;
                (Some(..), Some(..)) | (None, None) => unreachable!(),
                // Color and Stroke have images.
                // Color needing a whole image is a big ol inefficiency but that's todo :P
                (
                    Some(
                        crate::state::graph::LeafType::SolidColor { .. }
                        | crate::state::graph::LeafType::StrokeLayer { .. }
                        | crate::state::graph::LeafType::Text { .. },
                    ),
                    None,
                ) => true,
                // Blend groups need an image.
                (None, Some(crate::state::graph::NodeType::GroupedBlend(..))) => true,
                // Every other type has no graphic.
                _ => false,
            };
            if has_graphics {
                // Mark this data as needed
                retain_data.insert(id);
                // Allocate new image, if none allocated already.
                if let hashbrown::hash_map::Entry::Vacant(v) = graph_render_data.entry(id) {
                    v.insert(renderer.uninit_render_data()?);
                }
            }
        }

        // Drop all images that are no longer needed
        graph_render_data.retain(|id, _| retain_data.contains(id));

        Ok(())
    }
}
async fn render_changes(
    renderer: Arc<crate::render_device::RenderContext>,
    document_preview: Arc<crate::document_viewport_proxy::DocumentViewportPreviewProxy>,
) -> anyhow::Result<()> {
    let mut change_notifier = crate::default_provider().change_notifier();
    let mut changed: Vec<_> = crate::default_provider().document_iter().collect();
    let mut renderer = Renderer::new(renderer)?;
    // Initialize renderer with all documents.
    // let _ = renderer.render(&changed);
    loop {
        use tokio::sync::broadcast::error::RecvError;
        let first_msg = change_notifier.recv().await;
        match first_msg {
            // Got message. Collect as many as are available, then go render.
            Ok(msg) => {
                changed.clear();
                changed.push(msg.id());
                while let Ok(msg) = change_notifier.try_recv() {
                    // Handle lagged? That'd be a weird failure case...
                    changed.push(msg.id());
                }
                // Implicitly handles deletion - when the renderer goes to fetch changes,
                // it will see that the document has closed.
                //renderer.render(&changed)?;
                // No current doc, skip rendering.
                let Some(selections) = crate::AdHocGlobals::read_clone() else {
                    continue;
                };
                // Rerender, if requested
                if changed.contains(&selections.document) {
                    let write = document_preview.write().await;
                    renderer.render_one(selections.document, &write)?;
                    // We sync with renderer, oofs :V
                    write.submit_now();
                }
            }
            // Messages lost. Resubscrive and check all documents for changes, to be safe.
            Err(RecvError::Lagged(..)) => {
                // Discard messages.
                change_notifier = change_notifier.resubscribe();
                // Replace with every document ID. Doing this after the
                // resubscribe is important, such that no new docs are missed!
                changed.clear();
                changed.extend(crate::default_provider().document_iter());
                // Retain here. This is a list of all docs, so any not listed
                // are therefore deleted.
                tokio::task::yield_now().await;
                //renderer.render_retain(&changed)?;
            }
            // Work here is done!
            Err(RecvError::Closed) => return Ok(()),
        }
    }
}
pub async fn render_worker(
    renderer: Arc<crate::render_device::RenderContext>,
    request_reciever: tokio::sync::mpsc::Receiver<requests::RenderRequest>,
    document_preview: Arc<crate::document_viewport_proxy::DocumentViewportPreviewProxy>,
) -> anyhow::Result<()> {
    tokio::try_join!(
        async {
            requests::handler(request_reciever).await;
            Ok(())
        },
        render_changes(renderer, document_preview),
    )
    .map(|_| ())
}

/// The data managed by the renderer.
/// For now, in persuit of actually getting a working product one day,
/// this is a very coarse caching sceme. In the future, perhaps a bit more granular
/// control can occur, should performance become an issue:
///  * Caching images of incrementally older states, reducing work to get to any given state (performant undo)
///  * Caching tesselation output
pub struct RenderData {
    image: Arc<vk::Image>,
    pub view: Arc<vk::ImageView>,
}
mod stroke_renderer {

    use crate::vulkano_prelude::*;
    use anyhow::Result as AnyResult;
    use std::sync::Arc;
    mod vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/stamp.vert",
        }
    }
    mod frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/stamp.frag",
        }
    }

    pub struct StrokeLayerRenderer {
        context: Arc<crate::render_device::RenderContext>,
        texture_descriptor: Arc<vk::PersistentDescriptorSet>,
        gpu_tess: super::gpu_tess::GpuStampTess,
        pipeline: Arc<vk::GraphicsPipeline>,
    }
    impl StrokeLayerRenderer {
        pub fn new(context: Arc<crate::render_device::RenderContext>) -> AnyResult<Self> {
            // Begin uploading a brush image in the background while we continue setup
            let (image, sampler, _defer) = {
                let brush_image =
                    image::load_from_memory(include_bytes!("../brushes/splotch.png"))?
                        .into_luma_alpha8();

                // Iter over opacities. Weird/inefficient way to do this hehe
                // Idealy it'd just be an Alpha image but nothing suports that file layout :V
                let brush_image_alphas = brush_image.iter().skip(1).step_by(2);

                let device_image = vk::Image::new(
                    context.allocators().memory().clone(),
                    vk::ImageCreateInfo {
                        extent: [brush_image.width(), brush_image.height(), 1],
                        array_layers: 1,
                        format: vk::Format::R8_UNORM,
                        usage: vk::ImageUsage::SAMPLED | vk::ImageUsage::TRANSFER_DST,

                        ..Default::default()
                    },
                    vk::AllocationCreateInfo {
                        memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                )?;
                let image_stage = vk::Buffer::new_slice::<u8>(
                    context.allocators().memory().clone(),
                    vk::BufferCreateInfo {
                        usage: vk::BufferUsage::TRANSFER_SRC,
                        ..Default::default()
                    },
                    vk::AllocationCreateInfo {
                        memory_type_filter: vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    vk::DeviceSize::from(brush_image.width())
                        * vk::DeviceSize::from(brush_image.height()),
                )?;
                // Write image into buffer
                {
                    // Unwrap ok - the device can't possibly be using it,
                    // and we don't read from it from host.
                    let mut write = image_stage.write().unwrap();
                    // Copy pixel-by-pixel
                    write
                        .iter_mut()
                        .zip(brush_image_alphas)
                        .for_each(|(into, from)| *into = *from);
                }
                let mut cb = vk::AutoCommandBufferBuilder::primary(
                    context.allocators().command_buffer(),
                    context.queues().transfer().idx(),
                    vk::CommandBufferUsage::OneTimeSubmit,
                )?;
                let region = vk::BufferImageCopy {
                    image_extent: device_image.extent(),
                    image_subresource: vk::ImageSubresourceLayers {
                        array_layers: 0..1,
                        aspects: vk::ImageAspects::COLOR,
                        mip_level: 0,
                    },
                    // Buffer is tightly-packed, same size as device_image.
                    ..Default::default()
                };
                cb.copy_buffer_to_image(vk::CopyBufferToImageInfo {
                    regions: smallvec::smallvec![region],
                    ..vk::CopyBufferToImageInfo::buffer_image(image_stage, device_image.clone())
                })?;
                let fence = context
                    .now()
                    .then_execute(context.queues().transfer().queue().clone(), cb.build()?)?
                    .then_signal_fence_and_flush()?;

                let view = vk::ImageView::new(
                    device_image.clone(),
                    vk::ImageViewCreateInfo {
                        component_mapping: vk::ComponentMapping {
                            //Red is coverage of white, with premul.
                            a: vk::ComponentSwizzle::Red,
                            r: vk::ComponentSwizzle::Red,
                            b: vk::ComponentSwizzle::Red,
                            g: vk::ComponentSwizzle::Red,
                        },
                        ..vk::ImageViewCreateInfo::from_image(&device_image)
                    },
                )?;

                let sampler = vk::Sampler::new(
                    context.device().clone(),
                    vk::SamplerCreateInfo {
                        min_filter: vk::Filter::Linear,
                        mag_filter: vk::Filter::Linear,
                        ..Default::default()
                    },
                )?;

                (
                    view,
                    sampler,
                    // synchronizing at the end of init so other setup can happen in parallel.
                    defer::defer(move || fence.wait(None).unwrap()),
                )
            };

            let frag = frag::load(context.device().clone())?;
            let vert = vert::load(context.device().clone())?;
            // Unwraps ok here, using GLSL where "main" is the only allowed entry point.
            let frag = frag.entry_point("main").unwrap();
            let vert = vert.entry_point("main").unwrap();

            let frag_stage = vk::PipelineShaderStageCreateInfo::new(frag);
            let vert_stage = vk::PipelineShaderStageCreateInfo::new(vert.clone());
            // DualSrcBlend (~75% coverage) is used to control whether to erase or draw on a per-fragment basis
            // [1.0; 4] = draw, [0.0; 4] = erase.
            let premul_dyn_constants = {
                let blend = vk::AttachmentBlend {
                    src_alpha_blend_factor: vk::BlendFactor::Src1Alpha,
                    src_color_blend_factor: vk::BlendFactor::Src1Color,
                    dst_alpha_blend_factor: vk::BlendFactor::OneMinusSrcAlpha,
                    dst_color_blend_factor: vk::BlendFactor::OneMinusSrcAlpha,
                    alpha_blend_op: vk::BlendOp::Add,
                    color_blend_op: vk::BlendOp::Add,
                };
                let blend_states = vk::ColorBlendAttachmentState {
                    blend: Some(blend),
                    ..Default::default()
                };
                vk::ColorBlendState::with_attachment_states(1, blend_states)
            };

            let matrix_push_constant = vk::PushConstantRange {
                offset: 0,
                stages: vk::ShaderStages::VERTEX,
                size: std::mem::size_of::<vert::Matrix>() as u32,
            };

            let image_sampler_layout = vk::DescriptorSetLayout::new(
                context.device().clone(),
                vk::DescriptorSetLayoutCreateInfo {
                    bindings: [(
                        0,
                        vk::DescriptorSetLayoutBinding {
                            descriptor_count: 1,
                            stages: vk::ShaderStages::FRAGMENT,
                            ..vk::DescriptorSetLayoutBinding::descriptor_type(
                                vk::DescriptorType::CombinedImageSampler,
                            )
                        },
                    )]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                },
            )?;

            let layout = vk::PipelineLayout::new(
                context.device().clone(),
                vk::PipelineLayoutCreateInfo {
                    push_constant_ranges: vec![matrix_push_constant],
                    set_layouts: vec![image_sampler_layout],
                    ..Default::default()
                },
            )?;

            let pipeline = vk::GraphicsPipeline::new(
                context.device().clone(),
                None,
                vk::GraphicsPipelineCreateInfo {
                    color_blend_state: Some(premul_dyn_constants),
                    input_assembly_state: Some(vk::InputAssemblyState {
                        topology: vk::PrimitiveTopology::TriangleList,
                        primitive_restart_enable: false,
                        ..Default::default()
                    }),
                    multisample_state: Some(vk::MultisampleState::default()),
                    rasterization_state: Some(vk::RasterizationState {
                        cull_mode: vk::CullMode::None,
                        ..Default::default()
                    }),
                    vertex_input_state: Some(
                        super::gpu_tess::interface::OutputStrokeVertex::per_vertex()
                            .definition(&vert.info().input_interface)?,
                    ),
                    viewport_state: Some(vk::ViewportState {
                        viewports: smallvec::smallvec![vk::Viewport {
                            depth_range: 0.0..=1.0,
                            extent: [crate::DOCUMENT_DIMENSION as f32; 2],
                            offset: [0.0; 2],
                        }],
                        ..Default::default()
                    }),
                    subpass: Some(vk::PipelineSubpassType::BeginRendering(
                        vk::PipelineRenderingCreateInfo {
                            color_attachment_formats: vec![Some(crate::DOCUMENT_FORMAT)],
                            ..Default::default()
                        },
                    )),
                    stages: smallvec::smallvec![vert_stage, frag_stage,],
                    ..vk::GraphicsPipelineCreateInfo::layout(layout)
                },
            )?;
            let descriptor_set = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                pipeline.layout().set_layouts()[0].clone(),
                [vk::WriteDescriptorSet::image_view_sampler(
                    0, image, sampler,
                )],
                [],
            )?;

            let tess = super::gpu_tess::GpuStampTess::new(context.clone())?;

            Ok(Self {
                context,
                pipeline,
                gpu_tess: tess,
                texture_descriptor: descriptor_set,
            })
        }
        /// Allocate a new `RenderData` object. Initial contents are undefined!
        pub fn uninit_render_data(&self) -> anyhow::Result<super::RenderData> {
            use vulkano::VulkanObject;

            let image = vk::Image::new(
                self.context.allocators().memory().clone(),
                vk::ImageCreateInfo {
                    usage: vk::ImageUsage::COLOR_ATTACHMENT
                        | vk::ImageUsage::STORAGE
                        // For color clearing
                        | vk::ImageUsage::TRANSFER_DST,
                    extent: [crate::DOCUMENT_DIMENSION, crate::DOCUMENT_DIMENSION, 1],
                    array_layers: 1,
                    mip_levels: 1,
                    sharing: self.context.queues().sharing_compute_graphics(),
                    format: crate::DOCUMENT_FORMAT,
                    ..Default::default()
                },
                vk::AllocationCreateInfo {
                    memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )?;
            let view = vk::ImageView::new_default(image.clone())?;

            log::info!("Made render data at id{:?}", view.handle());

            Ok(super::RenderData { image, view })
        }
        pub fn draw(
            &self,
            strokes: &[&crate::state::stroke_collection::ImmutableStroke],
            renderbuf: &super::RenderData,
            clear: bool,
        ) -> AnyResult<()> {
            let mut matrix = cgmath::Matrix4::from_scale(2.0 / crate::DOCUMENT_DIMENSION as f32);
            matrix.y *= -1.0;
            matrix.w.x -= 1.0;
            matrix.w.y += 1.0;

            let mut batch = super::stroke_batcher::StrokeBatcher::new(
                self.context.allocators().memory().clone(),
                65536,
                vk::BufferUsage::STORAGE_BUFFER,
                vulkano::sync::Sharing::Exclusive,
            )?;
            batch.batch(strokes.iter().map(|&&i| i), |batch| -> AnyResult<_> {
                let (future, vertices, indirects) = self.gpu_tess.tess_batch(batch, true)?;

                let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
                    self.context.allocators().command_buffer(),
                    self.context.queues().graphics().idx(),
                    vk::CommandBufferUsage::OneTimeSubmit,
                )?;

                log::trace!(
                    "Drawing {} vertices from {} indirects",
                    vertices.len(),
                    indirects.len()
                );

                command_buffer
                    .begin_rendering(vk::RenderingInfo {
                        color_attachments: vec![Some(vk::RenderingAttachmentInfo {
                            clear_value: if clear {
                                Some([0.0, 0.0, 0.0, 0.0].into())
                            } else {
                                None
                            },
                            load_op: if clear {
                                vk::AttachmentLoadOp::Clear
                            } else {
                                vk::AttachmentLoadOp::Load
                            },
                            store_op: vk::AttachmentStoreOp::Store,
                            ..vk::RenderingAttachmentInfo::image_view(renderbuf.view.clone())
                        })],
                        contents: vk::SubpassContents::Inline,
                        depth_attachment: None,
                        ..Default::default()
                    })?
                    .bind_pipeline_graphics(self.pipeline.clone())?
                    .push_constants(
                        self.pipeline.layout().clone(),
                        0,
                        Into::<[[f32; 4]; 4]>::into(matrix),
                    )?
                    .bind_descriptor_sets(
                        vk::PipelineBindPoint::Graphics,
                        self.pipeline.layout().clone(),
                        0,
                        self.texture_descriptor.clone(),
                    )?
                    .bind_vertex_buffers(0, vertices)?
                    .draw_indirect(indirects)?
                    .end_rendering()?;

                let command_buffer = command_buffer.build()?;

                let fence = future
                    .then_execute(
                        self.context.queues().graphics().queue().clone(),
                        command_buffer,
                    )?
                    .then_signal_fence_and_flush()?;

                // After tessellation finishes, render.
                Ok(super::stroke_batcher::SyncOutput::Fence(fence))
            })?;

            Ok(())
        }
    }
}
