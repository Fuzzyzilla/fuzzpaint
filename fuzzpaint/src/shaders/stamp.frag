#version 460
layout(set = 0, binding = 0) uniform sampler2DArray brush_tex;

layout(location = 0) in vec4 color;
layout(location = 1) in vec4 blend_constants;
layout(location = 2) in vec2 uv;

// Output color
layout(location = 0, index = 0) out vec4 out_color;
// Blend constants - set up such that [0.0; 4] = eraser
layout(location = 0, index = 1) out vec4 out_constants;

void main() {
    out_color = color * texture(brush_tex, vec3(uv, 0.0));
    out_constants = blend_constants;
}