mod blender;
mod gpu_tess;
pub mod picker;
pub mod requests;
mod stroke_batcher;

use fuzzpaint_core::{
    queue::{self, state_reader::CommandQueueStateReader},
    state,
};
use std::sync::Arc;
use vulkano::command_buffer::{CopyImageInfo, ImageCopy};

use crate::vulkano_prelude::*;

struct PerDocumentData {
    listener: queue::DocumentCommandListener,
    /// Cached images of each of the nodes of the graph.
    graph_render_data: hashbrown::HashMap<state::graph::AnyID, RenderData>,
    /// precompiled blend operations, invalided when the graph changes.
    compiled_blend: Option<blender::BlendInvocation>,
    render_target: RenderData,
}

/// Dispatches render work to engines to create document images.
/// Maintains a cache of document render data.
struct Renderer {
    engines: Engines,
    data: hashbrown::HashMap<state::DocumentID, PerDocumentData>,
}
impl Renderer {
    fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
        Ok(Self {
            engines: Engines::new(context)?,
            data: hashbrown::HashMap::new(),
        })
    }
    fn render_one(
        &mut self,
        id: state::DocumentID,
        into: &Arc<vk::ImageView>,
    ) -> anyhow::Result<vk::FenceSignalFuture<Box<dyn vk::sync::GpuFuture + Send>>> {
        let data = self.data.entry(id);
        // Get the document data to update.
        let data = match data {
            hashbrown::hash_map::Entry::Occupied(o) => o.into_mut(),
            hashbrown::hash_map::Entry::Vacant(v) => {
                // Special case - new render! Build it + draw it from scratch.

                let Some(listener) = crate::global::provider()
                    .inspect(id, queue::DocumentCommandQueue::listen_from_now)
                else {
                    // Deleted before we could do anything.
                    anyhow::bail!("Document deleted before render worker reached it");
                };

                let data = v.insert(self.engines.new_render_from_scrach(listener)?);

                // Then copy to the preview.
                return self
                    .engines
                    .copy_document_to_preview_proxy(data, into)
                    .map_err(Into::into);
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

        // Draw just the changes!
        enum StrokeChanges {
            // Strokes were added
            Add(Vec<state::stroke_collection::ImmutableStrokeID>),
            // Big change, redraw from scratch.
            Invalidated,
        }

        let mut stroke_changes =
            hashbrown::HashMap::<state::stroke_collection::StrokeCollectionID, StrokeChanges>::new(
            );
        let mut graph_invalidated = false;

        let mut analyze_change = |change| -> std::ops::ControlFlow<()> {
            use fuzzpaint_core::commands::{Command, DoUndo, MetaCommand, StrokeCollectionCommand};
            use state::stroke_collection::commands::StrokeCommand;
            match change {
                // An added stroke can be executed as a delta.
                DoUndo::Do(Command::StrokeCollection(StrokeCollectionCommand::Stroke {
                    target: stroke_collection,
                    command:
                        StrokeCommand::Created {
                            target: stroke_id, ..
                        },
                })) => {
                    let changes = stroke_changes
                        .entry(*stroke_collection)
                        .or_insert(StrokeChanges::Add(vec![]));
                    match changes {
                        StrokeChanges::Add(add) => add.push(*stroke_id),
                        // Already invalidated, can't do a delta.
                        StrokeChanges::Invalidated => (),
                    }
                }
                // All other stroke commands invalidate the data and need full layer redraw.
                DoUndo::Do(Command::StrokeCollection(c))
                | DoUndo::Undo(Command::StrokeCollection(c)) => match *c {
                    StrokeCollectionCommand::Created(id)
                    | StrokeCollectionCommand::Stroke { target: id, .. } => {
                        let _ = stroke_changes.insert(id, StrokeChanges::Invalidated);
                    }
                },
                // Any modifications to the blend graph require a rebuild.
                // (text can be delta'd but that's not really implemented at all yet.)
                DoUndo::Do(Command::Graph(_)) | DoUndo::Undo(Command::Graph(_)) => {
                    graph_invalidated = true
                }
                // Palettes influence the blend graph and possibly every stroke layer. Uh oh.
                // Invalidate everything, and make this better future me!!!
                DoUndo::Do(Command::Palette(_)) | DoUndo::Undo(Command::Palette(_)) => {
                    for &key in changes.stroke_collections().0.keys() {
                        let _ = stroke_changes.insert(key, StrokeChanges::Invalidated);
                    }
                    graph_invalidated = true;
                    // Invalidated literally everything lmao, no need to keep looking at deltas.
                    return std::ops::ControlFlow::Break(());
                }
                // Commands must be externally flattened.
                DoUndo::Do(Command::Meta(MetaCommand::Scope(..)))
                | DoUndo::Undo(Command::Meta(MetaCommand::Scope(..))) => unreachable!(),
                // No influence on rendering.
                DoUndo::Do(Command::Meta(_) | Command::Dummy)
                | DoUndo::Undo(Command::Meta(_) | Command::Dummy) => (),
            }
            std::ops::ControlFlow::Continue(())
        };

        for change in changes.changes() {
            use fuzzpaint_core::commands::{Command, DoUndo, MetaCommand};
            // Flatten and analyze changes!
            match change {
                DoUndo::Do(Command::Meta(MetaCommand::Scope(_, s))) => {
                    // This should be recursive. I don't want to. BLegh.
                    for change in s {
                        if analyze_change(DoUndo::Do(change)).is_break() {
                            break;
                        }
                    }
                }
                DoUndo::Undo(Command::Meta(MetaCommand::Scope(_, s))) => {
                    // This should be recursive. I don't want to. BLegh.
                    for change in s.iter().rev() {
                        if analyze_change(DoUndo::Undo(change)).is_break() {
                            break;
                        }
                    }
                }
                _ => {
                    if analyze_change(change).is_break() {
                        break;
                    }
                }
            }
        }

        let mut fences = vec![];

        if graph_invalidated {
            log::trace!("Scouring allocations");
            // Needs recompile.
            let _ = data.compiled_blend.take();
            self.engines
                .allocate_prune_graph(&mut data.graph_render_data, changes.graph())?;
        }

        for (collection, stroke_changes) in stroke_changes {
            let graph_id = changes
                .graph()
                .iter()
                .find_map(|(id, data)| {
                    // If this node is a stroke layer with our same collection ID, then we found it!
                    data.leaf()
                        .is_some_and(|leaf| match leaf {
                            state::graph::LeafType::StrokeLayer {
                                collection: this_leaf,
                                ..
                            } => collection == *this_leaf,
                            _ => false,
                        })
                        .then_some(id)
                })
                .ok_or_else(|| anyhow::anyhow!("delta references non-existent node"))?;

            let render_data = data
                .graph_render_data
                .get(&graph_id)
                .ok_or_else(|| anyhow::anyhow!("missing render data for delta"))?;
            let collection = changes
                .stroke_collections()
                .get(collection)
                .ok_or_else(|| anyhow::anyhow!("delta references non-existent collection"))?;

            let which = match &stroke_changes {
                StrokeChanges::Add(which) => {
                    // Draw selected.
                    Some(which.as_slice())
                }
                StrokeChanges::Invalidated => {
                    // Draw all.
                    None
                }
            };

            if let Some(fence) =
                self.engines
                    .stroke_layer(collection, changes.palette(), render_data, which)?
            {
                fences.push(fence);
            }
        }

        for fence in fences {
            // Blegh. No way to do better express this with current vulkano sync.
            fence.wait(None)?;
        }

        // This has to be *after* stroke render, for some reason, or the layers don't show up at all.
        // Probably something wrong with the internal layout transitions. ;;;w;;;
        // *screaming*
        let compiled_blend = match &mut data.compiled_blend {
            Some(c) => c,
            None => {
                log::trace!("Recompiling blend graph");
                // Drop old one before building anew, to conserve mem. This could be delta'd instead to re-use old work, todo.
                let _ = data.compiled_blend.take();
                let invocation = self.engines.compile_blend_graph(
                    changes.graph(),
                    &data.graph_render_data,
                    changes.palette(),
                    &data.render_target,
                )?;

                data.compiled_blend.insert(invocation)
            }
        };

        compiled_blend.execute()?;

        self.engines.copy_document_to_preview_proxy(data, into)
    }
}
/// Struct that contains all the compiled GPU logic.
struct Engines {
    context: Arc<crate::render_device::RenderContext>,
    strokes: stroke_renderer::StrokeLayerRenderer,
    text_builder: crate::text::Builder,
    text: crate::text::renderer::monochrome::Renderer,
    blend: Arc<blender::BlendEngine>,
}
impl Engines {
    fn new(context: Arc<crate::render_device::RenderContext>) -> anyhow::Result<Self> {
        Ok(Self {
            context: context.clone(),
            blend: blender::BlendEngine::new(context.clone())?,
            text_builder: crate::text::Builder::allocate_new(
                context.allocators().memory().clone(),
            )?,
            text: crate::text::renderer::monochrome::Renderer::new(context.clone())?,
            strokes: stroke_renderer::StrokeLayerRenderer::new(context)?,
        })
    }
    /// Compile a GPU blend invocation for blending a document into an image.
    /// The `graph_render_data` should be fully populated with allocated images for any nodes or leaves that make use of images.
    ///
    /// Reuse this invocation as much as possible!
    fn compile_blend_graph(
        &self,
        graph: &state::graph::BlendGraph,
        graph_render_data: &hashbrown::HashMap<state::graph::AnyID, RenderData>,
        palette: &state::palette::Palette,
        into: &RenderData,
    ) -> anyhow::Result<blender::BlendInvocation> {
        use state::graph::{LeafType, NodeID, NodeType};
        /// Insert a single node (possibly recursing) into the builder.
        fn insert_blend(
            blend_engine: &Arc<blender::BlendEngine>,
            builder: &mut blender::BlendInvocationBuilder,
            graph_render_data: &hashbrown::HashMap<state::graph::AnyID, RenderData>,
            graph: &state::graph::BlendGraph,
            palette: &state::palette::Palette,

            id: state::graph::AnyID,
            data: &state::graph::NodeData,
        ) -> anyhow::Result<()> {
            match (data.leaf(), data.node()) {
                // Pre-rendered leaves
                (
                    Some(LeafType::StrokeLayer { blend, .. } | LeafType::Text { blend, .. }),
                    None,
                ) => {
                    let view = graph_render_data
                        .get(&id)
                        .ok_or_else(|| anyhow::anyhow!("blend data not found for leaf {id:?}"))?
                        .view
                        .clone();
                    builder.then_blend(blender::BlendImageSource::Immediate(view), *blend)?;
                }
                // Lazily rendered leaves
                (Some(LeafType::SolidColor { blend, source }), None) => {
                    // Dereference possibly paletted color
                    let color = source.get().left_or_else(|pal_idx| {
                        palette
                            .get(pal_idx)
                            .unwrap_or(fuzzpaint_core::color::Color::TRANSPARENT)
                    });
                    builder.then_blend(blender::BlendImageSource::SolidColor(color), *blend)?;
                }
                (Some(LeafType::Note), None) => (),
                // Passthrough - add children directly without grouped blend
                (None, Some(NodeType::Passthrough)) => {
                    blend_for_passthrough(
                        blend_engine,
                        builder,
                        graph_render_data,
                        graph,
                        palette,
                        id.try_into().unwrap(),
                    )?;
                }
                // Grouped blend - add children to a new blend worker.
                (None, Some(NodeType::GroupedBlend(blend))) => {
                    let handle = blend_for_node(
                        blend_engine,
                        graph_render_data,
                        graph,
                        palette,
                        id.try_into().unwrap(),
                        graph_render_data
                            .get(&id)
                            .ok_or_else(|| anyhow::anyhow!("blend data not found for group {id:?}"))
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
            blend_engine: &Arc<blender::BlendEngine>,
            builder: &mut blender::BlendInvocationBuilder,
            graph_render_data: &hashbrown::HashMap<state::graph::AnyID, RenderData>,
            graph: &state::graph::BlendGraph,
            palette: &state::palette::Palette,
            node: NodeID,
        ) -> anyhow::Result<()> {
            let iter = graph
                .iter_node(node)
                .ok_or_else(|| anyhow::anyhow!("Passthrough node not found"))?;
            for (id, data) in iter {
                insert_blend(
                    blend_engine,
                    builder,
                    graph_render_data,
                    graph,
                    palette,
                    id,
                    data,
                )?;
            }
            Ok(())
        }

        /// Recursively build a blend invocation for the node, or None for root.
        fn blend_for_node(
            blend_engine: &Arc<blender::BlendEngine>,
            graph_render_data: &hashbrown::HashMap<state::graph::AnyID, RenderData>,
            graph: &state::graph::BlendGraph,
            palette: &state::palette::Palette,
            node: NodeID,

            into_image: Arc<vk::ImageView>,
            clear_image: bool,
        ) -> anyhow::Result<blender::NestedBlendInvocation> {
            let iter = graph
                .iter_node(node)
                .ok_or_else(|| anyhow::anyhow!("Node not found"))?;
            let mut builder = blend_engine.clone().start(into_image, clear_image);

            for (id, data) in iter {
                insert_blend(
                    blend_engine,
                    &mut builder,
                    graph_render_data,
                    graph,
                    palette,
                    id,
                    data,
                )?;
            }

            // We traverse top-down, we need to blend bottom-up
            builder.reverse();
            Ok(builder.nest())
        }

        let mut top_level_blend = self.blend.clone().start(into.view.clone(), true);
        // Walk the tree in tree-order, building up a blend operation.
        for (id, data) in graph.iter_top_level() {
            insert_blend(
                &self.blend,
                &mut top_level_blend,
                graph_render_data,
                graph,
                palette,
                id,
                data,
            )?;
        }
        // We traverse top-down, we need to blend bottom-up
        top_level_blend.reverse();

        top_level_blend.build()
    }
    /// Render a document from scratch into a newly allocated document data.
    fn new_render_from_scrach(
        &self,
        listener: queue::DocumentCommandListener,
    ) -> anyhow::Result<PerDocumentData> {
        let mut data = PerDocumentData {
            listener,
            compiled_blend: None,
            graph_render_data: hashbrown::HashMap::new(),
            render_target: self.strokes.uninit_render_data()?,
        };

        // Observe concrete document state.
        let reader = data.listener.forward_clone_state()?;

        // Allocate blend and leaf images.
        self.allocate_prune_graph(&mut data.graph_render_data, reader.graph())?;

        // Draw leaves.
        self.leaves_from_scratch(&data, &reader)?;

        // Compile blending logic on the GPU.
        let invocation = self.compile_blend_graph(
            reader.graph(),
            &data.graph_render_data,
            reader.palette(),
            &data.render_target,
        )?;

        // Execute blending!
        data.compiled_blend.insert(invocation).execute()?;

        // Woohoo!
        Ok(data)
    }
    /// Render a stroke layer. If `which` is `Some`, this defines an update operation, where each stroke in `which` is drawn into the existing buffer.
    /// Otherwise, the buffer is cleared and all active (not undone) strokes from the collection are drawn.
    ///
    /// Ok(None) represents a success that does not need host-side waiting, `Ok(Some(fence))` requires synchronization before the operation is complete.
    fn stroke_layer(
        &self,
        collection: &state::stroke_collection::StrokeCollection,
        palette: &state::palette::Palette,
        data: &RenderData,
        which: Option<&[state::stroke_collection::ImmutableStrokeID]>,
    ) -> anyhow::Result<Option<vk::FenceSignalFuture<Box<dyn vk::sync::GpuFuture>>>> {
        enum EitherIter<
            'a,
            A: Iterator<Item = &'a state::stroke_collection::ImmutableStroke>,
            B: Iterator<Item = &'a state::stroke_collection::ImmutableStroke>,
        > {
            Active(A),
            Which(B),
        }
        impl<
                'a,
                A: Iterator<Item = &'a state::stroke_collection::ImmutableStroke>,
                B: Iterator<Item = &'a state::stroke_collection::ImmutableStroke>,
            > Iterator for EitherIter<'a, A, B>
        {
            type Item = &'a state::stroke_collection::ImmutableStroke;
            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    EitherIter::Active(a) => a.next(),
                    EitherIter::Which(b) => b.next(),
                }
            }
        }

        let clear = which.is_none();

        let strokes: Vec<_> = match which {
            Some(which) => EitherIter::Which(which.iter().filter_map(|&id| collection.get(id))),
            None => EitherIter::Active(collection.iter_active()),
        }
        .map(|stroke| {
            // Convert paletted color into concrete color.
            // THIS LOGIC SHOULD NOT BE HERE X3
            let color_modulate = stroke.brush.color_modulate.get().left_or_else(|idx| {
                palette
                    .get(idx)
                    .unwrap_or(fuzzpaint_core::color::Color::BLACK)
            });
            fuzzpaint_core::state::stroke_collection::ImmutableStroke {
                brush: state::StrokeBrushSettings {
                    color_modulate: color_modulate.into(),
                    ..stroke.brush
                },
                ..*stroke
            }
        })
        .collect();

        if strokes.is_empty() {
            if clear {
                //FIXME: Renderer doesn't know how to handle zero strokes.
                Self::clear(
                    &self.context,
                    data,
                    fuzzpaint_core::color::Color::TRANSPARENT,
                )
                .map(Some)
            } else {
                Ok(None)
            }
        } else {
            self.strokes
                .draw(strokes.as_ref(), data, clear)
                .map(|()| None)
        }
    }
    fn text_layer(
        &self,
        text: &str,
        pix_per_em: f32,
        data: &RenderData,
    ) -> anyhow::Result<vk::FenceSignalFuture<Box<dyn vk::sync::GpuFuture>>> {
        todo!();
        // Fixme: text builder needs inner mutability.
        // Self::render_text(&self.context, &mut self.text_builder, renderer, image, px_per_em, text)
    }
    fn copy_document_to_preview_proxy(
        &self,
        document_data: &PerDocumentData,
        into: &Arc<vk::ImageView>,
    ) -> anyhow::Result<vk::FenceSignalFuture<Box<dyn vk::sync::GpuFuture + Send>>> {
        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            self.context.allocators().command_buffer(),
            self.context.queues().graphics().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;
        let region = ImageCopy {
            dst_offset: [0; 3],
            src_offset: [0; 3],
            extent: [crate::DOCUMENT_DIMENSION, crate::DOCUMENT_DIMENSION, 1],
            src_subresource: vk::ImageSubresourceLayers {
                array_layers: 0..1,
                mip_level: 0,
                aspects: vk::ImageAspects::COLOR,
            },
            dst_subresource: vk::ImageSubresourceLayers {
                array_layers: into.subresource_range().array_layers.clone(),
                aspects: vk::ImageAspects::COLOR,
                mip_level: 0,
            },
            ..Default::default()
        };
        command_buffer.copy_image(CopyImageInfo {
            regions: smallvec::smallvec![region],
            ..CopyImageInfo::images(
                document_data.render_target.image.clone(),
                into.image().clone(),
            )
        })?;

        let command_buffer = command_buffer.build()?;

        Ok(vk::sync::now(self.context.device().clone())
            .then_execute(
                self.context.queues().graphics().queue().clone(),
                command_buffer,
            )?
            .boxed_send()
            .then_signal_fence_and_flush()?)
    }
    /// Renders every leaf, does not execute blend.
    fn leaves_from_scratch(
        &self,
        document_data: &PerDocumentData,
        reader: impl queue::state_reader::CommandQueueStateReader,
    ) -> anyhow::Result<()> {
        use state::graph::LeafType;
        let mut fences = vec![];

        // Walk the tree in arbitrary order, rendering all as needed and collecting their futures.
        for (id, data) in reader.graph().iter() {
            match data.leaf() {
                // Render stroke image
                Some(LeafType::StrokeLayer { collection, .. }) => {
                    let data = document_data.graph_render_data.get(&id).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Expected image to be created by allocate_prune_graph for {id:?}"
                        )
                    })?;
                    let strokes =
                        reader
                            .stroke_collections()
                            .get(*collection)
                            .ok_or_else(|| {
                                anyhow::anyhow!("Missing stroke collection {collection:?}")
                            })?;
                    if let Some(fence) = self.stroke_layer(strokes, reader.palette(), data, None)? {
                        fences.push(fence);
                    };
                }
                // Render stroke image
                Some(LeafType::Text {
                    text, px_per_em, ..
                }) => {
                    let data = document_data.graph_render_data.get(&id).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Expected image to be created by allocate_prune_graph for {id:?}"
                        )
                    })?;
                    fences.push(self.text_layer(text, *px_per_em, data)?);
                }
                // No rendering or lazily rendered.
                Some(LeafType::SolidColor { .. } | LeafType::Note) | None => (),
            }
        }

        // Wait for every fence. Terrible, but vulkano semaphores don't seem to be working currently.
        // Note to self: see commit fuzzpaint @ d435ca7c29cf045be413c9849be928693a2de458 for a time when this worked.
        // Iunno what changed :<
        for fence in fences {
            fence.wait(None)?;
        }

        Ok(())
    }
    fn clear(
        context: &crate::render_device::RenderContext,
        image: &RenderData,
        color: fuzzpaint_core::color::Color,
    ) -> anyhow::Result<vk::FenceSignalFuture<Box<dyn GpuFuture>>> {
        let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
            context.allocators().command_buffer(),
            // Unfortunately transfer queue cannot clear images TwT
            // Here, graphics is more likely to be idle than compute.
            context.queues().graphics().idx(),
            vk::CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer.clear_color_image(vk::ClearColorImageInfo {
            clear_value: color.as_array().into(),
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
        builder: &mut crate::text::Builder,
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
        &self,
        graph_render_data: &mut hashbrown::HashMap<state::graph::AnyID, RenderData>,
        graph: &state::graph::BlendGraph,
    ) -> anyhow::Result<()> {
        let mut retain_data = hashbrown::HashSet::new();
        for (id, node) in graph.iter() {
            #[allow(clippy::match_same_arms)]
            let has_graphics = match (node.leaf(), node.node()) {
                // We expect it to be a node xor leaf!
                // This is an api issue ;w;
                (Some(..), Some(..)) | (None, None) => unreachable!(),
                // Stroke and text have images.
                (
                    Some(
                        state::graph::LeafType::StrokeLayer { .. }
                        | state::graph::LeafType::Text { .. },
                    ),
                    None,
                ) => true,
                // Blend groups need an image.
                (None, Some(state::graph::NodeType::GroupedBlend(..))) => true,
                // Every other type has no graphic.
                _ => false,
            };
            if has_graphics {
                // Mark this data as needed
                retain_data.insert(id);
                // Allocate new image, if none allocated already.
                if let hashbrown::hash_map::Entry::Vacant(v) = graph_render_data.entry(id) {
                    v.insert(self.strokes.uninit_render_data()?);
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
    document_preview: Arc<crate::document_viewport_proxy::Proxy>,
) -> anyhow::Result<()> {
    // Sync -> Async bridge for change notification. Bleh..
    let (send, mut changes_recv) = tokio::sync::mpsc::unbounded_channel();
    let exit_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let exit_flag_move = exit_flag.clone();
    let _thread = std::thread::spawn(move || {
        let mut change_listener = crate::global::provider().change_listener();
        loop {
            // Parent requested child exit.
            if exit_flag_move.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            // Poll every so often, so an assertion of the exit flag is not missed.
            match change_listener.recv_timeout(std::time::Duration::from_millis(250)) {
                Ok(change) => {
                    // Got a change. Broadcast this one (and all others that are ready now)
                    if send.send(change.id()).is_err() {
                        // Disconnected!
                        return;
                    }
                    while let Ok(change) = change_listener.try_recv() {
                        if send.send(change.id()).is_err() {
                            // Disconnected!
                            return;
                        }
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => (),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
            }
        }
    });
    // Drop order - this will run before thread is joined, otherwise deadlock occurs!
    defer::defer!(exit_flag.store(true, std::sync::atomic::Ordering::Relaxed));

    let mut changes: Vec<_> = crate::global::provider().document_iter().collect();
    let mut renderer = Renderer::new(renderer)?;

    loop {
        let changes = async {
            // Already has some! Report immediately.
            if !changes.is_empty() {
                return Some(&mut changes);
            }
            let first = changes_recv.recv().await?;
            changes.push(first);
            // Collect all others that are available without blocking as well:
            while let Ok(next) = changes_recv.try_recv() {
                changes.push(next);
            }
            Some(&mut changes)
        };

        let Some(changes) = changes.await else {
            // Channel closed
            return Ok(());
        };
        // Implicitly handles deletion - when the renderer goes to fetch changes,
        // it will see that the document has closed.
        //renderer.render(&changed)?;
        // No current doc, skip rendering.
        let Some(selections) = crate::AdHocGlobals::read_clone() else {
            changes.clear();
            continue;
        };
        // Rerender, if requested
        if changes.contains(&selections.document) {
            let write = document_preview.write().await;

            let fence = renderer.render_one(selections.document, &write)?;

            write.submit_with_fence(fence);
        }
        changes.clear();
    }
}
pub async fn render_worker(
    renderer: Arc<crate::render_device::RenderContext>,
    request_reciever: tokio::sync::mpsc::Receiver<requests::RenderRequest>,
    document_preview: Arc<crate::document_viewport_proxy::Proxy>,
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

    use crate::{renderer::gpu_tess, vulkano_prelude::*};
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
        texture_descriptors: fuzzpaint_core::brush::UniqueIDMap<Arc<vk::PersistentDescriptorSet>>,
        gpu_tess: super::gpu_tess::GpuStampTess,
        pipeline: Arc<vk::GraphicsPipeline>,
    }
    impl StrokeLayerRenderer {
        pub fn new(context: Arc<crate::render_device::RenderContext>) -> AnyResult<Self> {
            // Begin uploading a brush image in the background while we continue setup
            let (image_a, image_b, sampler, _defer) = {
                let brush_a = image::load_from_memory(include_bytes!("../../brushes/splotch.png"))?
                    .into_luma8();
                let mut brush_b = image::load_from_memory(include_bytes!(
                    "../../../fuzzpaint-core/default/circle.png"
                ))?
                .into_luma8();

                brush_b.iter_mut().for_each(|l| *l = 255 - *l);
                assert_eq!(brush_a.width(), brush_b.width());
                assert_eq!(brush_a.height(), brush_b.height());
                let mips = brush_a.width().max(brush_a.height()).ilog2() + 1;

                let device_image = vk::Image::new(
                    context.allocators().memory().clone(),
                    vk::ImageCreateInfo {
                        extent: [brush_a.width(), brush_a.height(), 1],
                        array_layers: 2,
                        mip_levels: mips,
                        format: vk::Format::R8_UNORM,
                        usage: vk::ImageUsage::SAMPLED
                            | vk::ImageUsage::TRANSFER_DST
                            | vk::ImageUsage::TRANSFER_SRC,

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
                    vk::DeviceSize::from(brush_a.width())
                        * vk::DeviceSize::from(brush_a.height())
                        * 2,
                )?;
                // Write image into buffer
                {
                    // Unwrap ok - the device can't possibly be using it,
                    // and we don't read from it from host.
                    let mut write = image_stage.write().unwrap();

                    let partition = brush_a.width() as usize * brush_a.height() as usize;
                    // Copy pixel-by-pixel
                    write[..partition].copy_from_slice(&brush_a);
                    write[partition..].copy_from_slice(&brush_b);
                }
                let mut cb = vk::AutoCommandBufferBuilder::primary(
                    context.allocators().command_buffer(),
                    context.queues().transfer().idx(),
                    vk::CommandBufferUsage::OneTimeSubmit,
                )?;
                let region = vk::BufferImageCopy {
                    image_extent: device_image.extent(),
                    image_subresource: vk::ImageSubresourceLayers {
                        array_layers: 0..2,
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
                // Generate mips.
                {
                    let mut src_width = brush_a.width();
                    let mut src_height = brush_a.height();
                    for src_mip in 0..mips - 1 {
                        let dst_mip = src_mip + 1;
                        let dst_width = src_width / 2;
                        let dst_height = src_height / 2;

                        let blit = vk::ImageBlit {
                            src_subresource: vk::ImageSubresourceLayers {
                                array_layers: 0..2,
                                aspects: vk::ImageAspects::COLOR,
                                mip_level: src_mip,
                            },
                            dst_subresource: vk::ImageSubresourceLayers {
                                array_layers: 0..2,
                                aspects: vk::ImageAspects::COLOR,
                                mip_level: dst_mip,
                            },
                            src_offsets: [[0, 0, 0], [src_width, src_height, 1]],
                            dst_offsets: [[0, 0, 0], [dst_width, dst_height, 1]],
                            ..Default::default()
                        };

                        cb.blit_image(vk::BlitImageInfo {
                            filter: vk::Filter::Linear,
                            regions: smallvec::smallvec![blit,],
                            ..vk::BlitImageInfo::images(device_image.clone(), device_image.clone())
                        })?;

                        src_width = dst_width;
                        src_height = dst_height;
                    }
                }
                let fence = context
                    .now()
                    .then_execute(context.queues().transfer().queue().clone(), cb.build()?)?
                    .then_signal_fence_and_flush()?;

                let view_a = vk::ImageView::new(
                    device_image.clone(),
                    vk::ImageViewCreateInfo {
                        component_mapping: vk::ComponentMapping {
                            //Red is coverage of white, with premul.
                            a: vk::ComponentSwizzle::Red,
                            r: vk::ComponentSwizzle::Red,
                            b: vk::ComponentSwizzle::Red,
                            g: vk::ComponentSwizzle::Red,
                        },
                        subresource_range: vk::ImageSubresourceRange {
                            array_layers: 0..1,
                            aspects: vk::ImageAspects::COLOR,
                            mip_levels: 0..mips,
                        },
                        ..vk::ImageViewCreateInfo::from_image(&device_image)
                    },
                )?;
                let view_b = vk::ImageView::new(
                    device_image.clone(),
                    vk::ImageViewCreateInfo {
                        component_mapping: vk::ComponentMapping {
                            //Red is coverage of white, with premul.
                            a: vk::ComponentSwizzle::Red,
                            r: vk::ComponentSwizzle::Red,
                            b: vk::ComponentSwizzle::Red,
                            g: vk::ComponentSwizzle::Red,
                        },
                        subresource_range: vk::ImageSubresourceRange {
                            array_layers: 1..2,
                            aspects: vk::ImageAspects::COLOR,
                            mip_levels: 0..mips,
                        },
                        ..vk::ImageViewCreateInfo::from_image(&device_image)
                    },
                )?;

                let sampler = vk::Sampler::new(
                    context.device().clone(),
                    vk::SamplerCreateInfo {
                        min_filter: vk::Filter::Linear,
                        mag_filter: vk::Filter::Linear,
                        mipmap_mode: vulkano::image::sampler::SamplerMipmapMode::Linear,
                        ..Default::default()
                    },
                )?;

                (
                    view_a,
                    view_b,
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
            let descriptor_set_a = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                pipeline.layout().set_layouts()[0].clone(),
                [vk::WriteDescriptorSet::image_view_sampler(
                    0,
                    image_a,
                    sampler.clone(),
                )],
                [],
            )?;
            let descriptor_set_b = vk::PersistentDescriptorSet::new(
                context.allocators().descriptor_set(),
                pipeline.layout().set_layouts()[0].clone(),
                [vk::WriteDescriptorSet::image_view_sampler(
                    0, image_b, sampler,
                )],
                [],
            )?;

            let tess = super::gpu_tess::GpuStampTess::new(context.clone())?;

            Ok(Self {
                context,
                pipeline,
                gpu_tess: tess,
                texture_descriptors: [
                    (fuzzpaint_core::brush::UniqueID([0; 32]), descriptor_set_a),
                    (
                        fuzzpaint_core::brush::UniqueID([
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                        ]),
                        descriptor_set_b,
                    ),
                ]
                .into_iter()
                .collect(),
            })
        }
        /// Allocate a new `RenderData` object. Initial contents are undefined!
        pub fn uninit_render_data(&self) -> anyhow::Result<super::RenderData> {
            use vulkano::VulkanObject;

            let image = vk::Image::new(
                self.context.allocators().memory().clone(),
                vk::ImageCreateInfo {
                    // Too many !!
                    usage:
                    // Feedback loop for blending into..
                    vk::ImageUsage::COLOR_ATTACHMENT
                        | vk::ImageUsage::INPUT_ATTACHMENT
                        // Source for blending from..
                        | vk::ImageUsage::SAMPLED
                        // For color clearing..
                        | vk::ImageUsage::TRANSFER_DST
                        // For blitting to preview proxy...
                        | vk::ImageUsage::TRANSFER_SRC,
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
            strokes: &[fuzzpaint_core::state::stroke_collection::ImmutableStroke],
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
            batch.batch(strokes.iter().copied(), |batch| -> AnyResult<_> {
                let gpu_tess::TessResult {
                    ready_after,
                    vertices,
                    mut indirects,
                    sources,
                } = self.gpu_tess.tess_batch(batch, true)?;

                let mut sources = &sources[..];
                let mut next_indirects_by_brush_id = || -> Option<(fuzzpaint_core::brush::UniqueID, vk::Subbuffer<[vulkano::command_buffer::DrawIndirectCommand]>)> {
                    let id = sources.first()?.brush.brush;
                    let first_differ = sources[1..].iter().position(|source| source.brush.brush != id);

                    if let Some(idx) = first_differ {
                        // Position refers to index in 1..
                        // Convert to index in 0..
                        let idx = idx + 1;

                        sources = &sources[idx..];
                        let (taken_indirects, left_indirects) = indirects.clone().split_at(idx as u64);
                        indirects = left_indirects;

                        Some((id, taken_indirects))
                    } else {
                        sources = &[];
                        // Take the rest.
                        Some((id, indirects.clone()))
                    }
                };

                let mut command_buffer = vk::AutoCommandBufferBuilder::primary(
                    self.context.allocators().command_buffer(),
                    self.context.queues().graphics().idx(),
                    vk::CommandBufferUsage::OneTimeSubmit,
                )?;

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
                    .bind_vertex_buffers(0, vertices)?;

                // Group together commands by brush ID and draw them!
                while let Some((brush_id, indirects)) = next_indirects_by_brush_id() {
                    let Some(descriptor) = self.texture_descriptors
                        .get(&brush_id)
                        .cloned() else {
                            continue
                        };
                    command_buffer
                    .bind_descriptor_sets(
                        vk::PipelineBindPoint::Graphics,
                        self.pipeline.layout().clone(),
                        0,
                        descriptor,
                    )?
                    .draw_indirect(indirects)?;
                }

                command_buffer.end_rendering()?;

                let command_buffer = command_buffer.build()?;

                // After tessellation finishes, render.
                let fence = ready_after
                    .then_execute(
                        self.context.queues().graphics().queue().clone(),
                        command_buffer,
                    )?
                    .then_signal_fence_and_flush()?;

                // Let the batcher know when we're done using the stage.
                // (In reality, the stage is done after `ready_after` but vulkano sync currently lacks a way to represent this)
                Ok(super::stroke_batcher::SyncOutput::Fence(fence))
            })?;

            Ok(())
        }
    }
}
