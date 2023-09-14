#version 460

layout(push_constant) uniform Matrix {
    mat4 mvp;
} push_matrix;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;
layout(location = 3) in vec4 secondary_color;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 blend_constants;
layout(location = 2) out vec2 out_uv;

void main() {
    out_color = color;
    blend_constants = secondary_color;
    out_uv = uv;
    
    vec4 position_2d = push_matrix.mvp * vec4(pos, 0.0, 1.0);
    gl_Position = vec4(position_2d.xy, 0.0, 1.0);
}