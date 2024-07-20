#version 460

// Blends color from src on top of the color from dst, using non-clip logic.

// Expects a #DEFINE of "EXPR" that `return`s the premultiplied RGBA of the result, given
// `vec4 c_dst` and `vec4 c_src` representing the premultiplied RGBA of the destination and source,
// respectively. The logic need not handle the fact that transparent dst should return src unchanged,
// instead only the opaque-dst logic should be implemented.
// Example:

// #DEFINE EXPR return c_src * 0.2;

#ifndef EXPR
# error "You must #DEFINE an EXPR to perform blending."
#endif

layout(set = 0, binding = 0) uniform sampler2D src;
layout(input_attachment_index = 0, set = 1, binding = 0) uniform subpassInput dst;

layout(push_constant) uniform Constants {
    // Solid color constant, otherwise just the alpha is used as global multiplier.
    vec4 solid_color;
    // True if the shader should 'sample' from `solid_color` instead of the image.
    // UB to read image if this is set.
    bool is_solid;
};

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

vec4 blend(in vec4 c_src, in vec4 c_dst) {
    EXPR
}

void main() {
    vec4 c_dst = subpassLoad(dst);
    vec4 c_src = is_solid ? solid_color : (texture(src, uv) * solid_color.a);

    // Apply user logic, but we need to apply logic so that a transparent dst
    // gives src directly, ignoring mode. Yes, this is the expected behavior, lol.
    vec4 result = blend(c_src, c_dst);

    // To handle transparent dst, do a 'normal' blend of result, clpped by dst, on top of src.
    // Effectively makes it so that transparent dst shows src as "normal" blended.
    color = result + c_src * (1.0 - c_dst.a);
}
