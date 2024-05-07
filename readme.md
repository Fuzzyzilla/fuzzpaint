<h1 style="text-align:center;"><a href="#" onclick="return false;"><img src="https://raw.githubusercontent.com/googlefonts/noto-emoji/main/svg/emoji_u1f411.svg" alt="Baa" title="Baa" style="position:relative;bottom: -0.2em;width:1em;"/></a> Fuzzpaint</h1>

![a screenshot of the app, with a sheep mascot drawn](demo-images/full.png)

## What's all this?
A graphics accelerated paint program combining the simplicity of raster with all the lossless quality of vector!

Fuzzpaint is the culmination of many years of passive frustration at the state of digital drawing applications. Inspired by the limitless rescalability of vector formats like *SVG* and the
lossless experimentation afforded by compositing software, Fuzzpaint is built from the ground up to provide both of these workflows. In contrast to the methodical precision of these, however,
Fuzzpaint is designed combine these benifits with the ease-of-use of freeform raster-based digital art programs.

The primary idea at the core of this project is to store the brush input verbatim, interpreting this path on-the-fly based on brush settings to render it to the output image. By working with brush strokes as path
data instead of flattened rasterized images, additional editing options are exposed such as the ability to modify brush settings after-the-fact and freely transform parts of a drawing without loss of quality.
One might think that this would be prohibitively slow, but to my great amazement such an image can be fully rendered in a few milliseconds without breaking a sweat on a decade old iGPU!
In addition, individual drawing actions use so little data that several novel niceties arise: infinite undo+redo is implemented and could even be saved within the document for little extra cost,
and the tininess of individual actions inspires ideas of peer-to-peer collaborative drawing (but I'm getting ahead of myself!)

Finally, it aims to allow me to play with digital art in ways that *I just think is neat!* Ideas include arbitrary shader effects, procedural lighting for 2D scenes, etc - but that's a long ways off for now.

**This project is in heavy development**, many features are in-progress and semver is ignored. Notably, **drawings cannot yet be saved** - but you can doodle to your heart's content!

### What this isn't
* An image editor - though this project aims to provide my ideal digital art creation environment over traditional raster software, it does not aim to implement the other functions of a raster image editor.
Some features are planned for including raster images within an artwork, however!
* A graphic/vector design software - Despite it's vector nature, Fuzzpaint does not aim to provide precise control over curves and is instead intended for freeform hand-drawn artwork - though, like above,
some features will be available for brush stroke editting for touching up drawings.

## Building
Requires the [most recent Rust *nightly* toolchain](https://www.rust-lang.org/tools/install). Clone and execute `cargo +nightly run --release` from within the root directory of this repo!

## Platform Support
This app is cross platform and should run on any device that meets the [current vulkan requirements](assumptions.md).
(if your device doesnt work - even if it's because of these requirements - please file an issue!).

For tablet support, see the [octotablet](https://github.com/Fuzzyzilla/octotablet) sister project.

# Road to **0.2.0**
To declare **0.2.0**, I would like to be able to freely doodle a thing and save it to disk. We're getting dangerously close!

 - [ ] File I/O
   - [ ] Read/Write [custom vector image format](fileschema.md)
   - [ ] Write file history
   - [ ] Export common image formats
     - via image-rs/image
   - [X] [Shell integration](https://github.com/Fuzzyzilla/fuzzpaint-thumbnailer)
 - [ ] Brushes
   - [ ] Make and manage textured brushes from inside fuzzpaint
   - [ ] Save brushes to file
     - To document or to global repo? - resolved: both, with ~~UUIDs~~ cryptographic hashes! That way, files can be freely shared, and brushes can be easily re-used.
   - [X] Tesselation engine capable of mixed brushes
   - [X] Stamped brushes
   - [X] Efficient erasers
 - [ ] Layers
   - [X] Simple UI to manage layer creation, order, modes, etc.
   - [X] Blending compute shaders
     - take advantage of associativity and commutativity of blend modes to reduce number of distinct dispatches?
     - fall forward on `EXT_blend_operation_advanced` or `EXT_fragment_shader_interlock`?
   - [X] Passthrough, grouped-blend layers
   - [X] Color fill layers
   - [ ] Text layers
   - FX + other self-populating layers (clone, gradients) come later
 - [ ] UI
   - [X] Initial layout
   - [X] A ~~simple~~ **✨robust and rebindable✨** hotkey system, with support for
         pen and pad buttons (although, pen+pad events are not yet wired up)
   - [ ] Application settings menu
   - [X] Infinite Undo/Redo
   - [X] Pan, Zoom, Scroll, Rotate
   - [ ] Mirror
   - [X] Fit, 100% modes
 - [ ] Editor
   - [ ] Color, Object Pickers
   - [ ] Select strokes
     - [ ] Select parts of strokes
   - [ ] Transform existing layers and strokes
   - [ ] Cut, Copy, Paste strokes (within program only)

# Long-term Goals
 * Support the "95%" GPU - For accessibility, this should work on low-end GPUs. Don't rely on overly niche vulkan extensions, but fall-forward on them if they provide sufficient benifit (eg, `EXT_mesh_shader`)
    * Progress towards this goal is tracked in [assumptions.md](assumptions.md).
 * Survive extreme errors - recovery from swapchain, surface, device, or even window loss.
   * *No* important data should ever exist on the GPU alone - all buffers and images should be derived data.
 * Be able to run without a surface at all. For exporting files from command line.
 * File saving should contain no fallible operations (aside from IO)

