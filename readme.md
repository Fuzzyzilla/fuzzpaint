# fuzzpaint-vk

Graphics accelerated vector-based paint program for people who like compositing software :3

(It doesn't do anything yet)

## Goals
 * Survive extreme errors - recovery from swapchain, surface, device, or even window loss.
   * *No* important data should ever exist on the GPU alone - all buffers and images should be derived data.
 * Be able to run without a surface at all. For exporting files from command line.
 * File saving should contain no fallible operations (aside from IO)

## Features (one day)
 * Maintain entire file history inside saves.
    * Infinite undo and editable past actions.
    * Branching histories with tags? (Low priority)
 * No destructive editting :3