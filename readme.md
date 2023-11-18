# <a href="#" onclick="return false;"><img src="https://raw.githubusercontent.com/googlefonts/noto-emoji/main/svg/emoji_u1f411.svg" alt="Baa" title="Baa" style="position:relative;bottom: -0.2em;width:1em;"/></a> fuzzpaint-vk

Graphics accelerated vector-based paint program for people who like compositing software :3

In heavy development, many features are in-progress and semver is ignored. But, you can doodle to your heart's content!

## Building
Requires the [most recent Rust nightly toolchain](https://www.rust-lang.org/tools/install). Clone this repo and execute `cargo run --release` from within the root directory of this repo!

**Note the platform support below.** The app is cross platform and should run on any device (if not, please report) but tablet input is very much unfinished.

| **Platform**      | Pen input   | Pad input   | Notes                                                   |
|-------------------|-------------|-------------|---------------------------------------------------------|
| Windows (Ink)     |None         |None         |                                                         |
| Unix (Xorg)       |Pressure only|None         |`WINIT_UNIX_BACKEND=x11` to activate using xwayland      |
| ~~Unix (Wayland)~~|None         |None         |No universally supported tablet API. [Currently disabled!](https://github.com/Fuzzyzilla/fuzzpaint-vk/issues/21#issue-1953431137)|
| Windows (wintab)  |None         |None         |Documentation tough to come accross                      |

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
     - To document or to global repo? - resolved: both, with UUIDs! That way, files can be freely shared, and brushes can be easily re-used.
   - [X] Tesselation engine capable of mixed brushes
   - [X] Stamped brushes
   - [X] Efficient erasers
 - [X] Layers
   - [X] Simple UI to manage layer creation, order, modes, etc.
   - [X] Blending compute shaders
     - take advantage of associativity and commutativity of blend modes to reduce number of distinct dispatches?
     - fall forward on `EXT_blend_operation_advanced` or `EXT_fragment_shader_interlock`?
   - [X] Passthrough, grouped-blend layers
   - [X] Color fill layers
   - FX + other self-populating layers (clone, gradients) come later
 - [ ] UI
   - [X] Initial layout
   - [X] A ~~simple~~ **✨robust and rebindable✨** hotkey system, with support for
         pen and pad buttons (although, pen+pad events are not yet reported)
   - [X] Infinite Undo/Redo
   - [X] Pan, Zoom, Scroll, Rotate, Mirror viewport
   - [ ] Fit, 100% modes
 - [ ] Editor
   - [ ] Select strokes
     - [ ] Select parts of strokes
   - [ ] Transform existing layers and strokes
   - [ ] Cut, Copy, Paste strokes (within program only)

## Long-term Goals
 * Support the "95%" GPU - For accessibility, this should work on low-end GPUs. Don't rely on overly niche vulkan extensions, but fall-forward on them if they provide sufficient benifit (eg, `EXT_mesh_shader`)
    * Progress towards this goal is tracked in [assumptions.md](assumptions.md).
 * Survive extreme errors - recovery from swapchain, surface, device, or even window loss.
   * *No* important data should ever exist on the GPU alone - all buffers and images should be derived data.
 * Be able to run without a surface at all. For exporting files from command line.
 * File saving should contain no fallible operations (aside from IO)