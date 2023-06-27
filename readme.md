# fuzzpaint-vk :sheep:

Graphics accelerated vector-based paint program for people who like compositing software :3

It doesn't do much more than let you scribble yet.

Platform support, in order of development priority:
| **Platform**      | Pen input          | Pad input          |
|-------------------|--------------------|--------------------|
| Unix (Wayland)    |:white_large_square:|:white_large_square:|
| Unix (Xorg)       |Partial             |:x:                 |
| Windows (Ink)     |:white_large_square:|:white_large_square:|
| Windows (wintab)  |:white_large_square:|:white_large_square:|

# Road to **0.2.0**
To declare **0.2.0**, I would like to be able to somewhat freely draw a thing and save said thing to disk. Note-to-self: do so in a sustainable manner, such that completing 0.2.0 does not prevent me from developing 1.0.0, in both the burnout and technical debt sense :P

 - [ ] File I/O
   - [ ] Read/Write custom vector image format
   - [ ] File history (linear)
   - [ ] Write common image formats (via image-rs/image)
 - [ ] Brushes
   - [ ] Make and manage textured brushes from inside fuzzpaint
   - [ ] Save brushes to file (to document or to global repo?)
   - [ ] Tesselation engine capable of mixed brushes
   - [ ] Roller brushes
   - [ ] Stamped brushes
   - [ ] Efficient erasers
 - [ ] Layers
   - [ ] Simple UI to manage layer creation, order, modes, etc.
   - [ ] Blending Über-compute-shader 
     - In the future, compile a document-specific compute shader to do the blending in a more optimizer-friendly manner.
     - fall forward on `EXT_blend_operation_advanced`?
   - Groups + FX + self-populating layers (clone, fill) come later
 - [ ] UI
   - [ ] A simple hotkey system, for common actions like Ctrl+Z or Ctrl+Space
   - [ ] Infinite Undo/Redo
   - [ ] Pan, Zoom, Scroll, Rotate viewport
   - [ ] Fit, 100% modes

## Long-term Goals
 * Support the "95%" GPU - For accessibility, this should work on low-end GPUs. Don't rely on overly niche vulkan extensions, but fall-forward on them if they provide sufficient benifit (eg, `EXT_mesh_shader`)
 * Survive extreme errors - recovery from swapchain, surface, device, or even window loss.
   * *No* important data should ever exist on the GPU alone - all buffers and images should be derived data.
 * Be able to run without a surface at all. For exporting files from command line.
 * File saving should contain no fallible operations (aside from IO)