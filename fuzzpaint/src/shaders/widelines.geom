#version 460

// ========================= DEFINES ===================================
// optional unless otherwise noted.
//
// datatypes:
//  width: f32
//  in_u: f32
//  out_uv: vec2
//  all others: any
//
// WIDTH_LOCATION (required) - location of the `width` input variable.
// INPUTS - full GLSL definition of input arrays[4], excluding width, including U + layout
// OUTPUTS - full GLSL definition of output elements, including UV + layout
//
// COPY_B - snippit of GLSL to copy inputs to outputs for element index called "B", excluding UV
// COPY_C - snippit of GLSL to copy inputs to outputs for element index called "C", excluding UV
//
// IN_U_NAME, OUT_UV_NAME - INPUT and OUTPUT names for vertex 1d texcoord -> fragment 2d texcoord
// (the requirement that INPUT and OUTPUT define their own u/uv is to allow whatever qualifiers user wants)

#ifndef WIDTH_LOCATION
	#error "WIDTH_LOCATION must be provided"
#endif
// Mutually optional - if one is def but not the other, an error!
#if defined(IN_U_NAME) && !defined(OUT_UV_NAME) || !defined(IN_U_NAME) && defined(OUT_UV_NAME)
	#error "IN_U_NAME and OUT_UV_NAME must have both or neither defined"
#endif
// Missing, fill in with empty.
#ifndef INPUTS
	#define INPUTS
#endif
#ifndef OUTPUTS
	#define OUTPUTS
#endif
#ifndef COPY_B
	#define COPY_B
#endif
#ifndef COPY_C
	#define COPY_C
#endif

layout(std430, push_constant) uniform Push { mat4 transform; };

layout(lines_adjacency) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = WIDTH_LOCATION) in float in_width[4];

INPUTS

OUTPUTS

#define A 0
#define B 1
#define C 2
#define D 3
#define A_in gl_in[A]
#define B_in gl_in[B]
#define C_in gl_in[C]
#define D_in gl_in[D]

// positive if B is "to the left of" A.
// alternatively, sin of the counterclockwise angle between A and B, times A and
// B's lengths.
float cross2(in vec2 a, in vec2 b) { return a.x * b.y - a.y * b.x; }
// Rotates input clockwise 90deg
vec2 clockwise90(in vec2 a) { return vec2(a.y, -a.x); }

void main() {
	//         2\--------4
	//         |  \      |
	// A - - - B - -\- - C - - - D
	//         |      \  |
	//         1--------\3

	// --------- Calculate.....
	// Todo: what if smol delta?
	// This causes small visual breaks in the line.
	vec2 ba = normalize(A_in.gl_Position.xy - B_in.gl_Position.xy);
	vec2 bc = normalize(C_in.gl_Position.xy - B_in.gl_Position.xy);
	vec2 cd = normalize(D_in.gl_Position.xy - C_in.gl_Position.xy);
	vec2 cb = -bc;

	// If inner angle is acute, mirror it!
	// Makes sharp corners have a squared off appearance instead of a needle lol
	ba = dot(ba, bc) < 0 ? -ba : ba;
	cd = dot(cd, cb) < 0 ? -cd : cd;

	// Points from B to 1 or 2
	vec2 b_normal = (bc - ba) / 2.0;
	// If not sufficiently long to normalize, use rotated BC instead
	b_normal =
			dot(b_normal, b_normal) < 0.0001 ? clockwise90(bc) : normalize(b_normal);
	// Make the cross positive, now points from B to 1
	b_normal = cross2(ba, b_normal) < 0.0 ? -b_normal : b_normal;

	// Points from C to 3 or 4
	vec2 c_normal = (cb - cd) / 2.0;
	// If not sufficiently long to normalize, use rotated CB instead
	c_normal =
			dot(c_normal, c_normal) < 0.0001 ? clockwise90(cb) : normalize(c_normal);
	// Make the cross positive, now points from C to 4
	c_normal = cross2(cd, c_normal) < 0.0 ? -c_normal : c_normal;

	// Project normals (todo: is this right?)
	b_normal = (transform * vec4(b_normal, 0.0, 0.0)).xy;
	c_normal = (transform * vec4(c_normal, 0.0, 0.0)).xy;

	// --------- Do vertices!
	float b_half_width = abs(in_width[B] / 2.0);
	float c_half_width = abs(in_width[C] / 2.0);

	// 1
	gl_Position = B_in.gl_Position + vec4(b_normal * b_half_width, 0.0, 0.0);
	COPY_B
	#ifdef IN_U_NAME
	OUT_UV_NAME = vec2(IN_U_NAME[B], 0.0);
	#endif
	EmitVertex();

	// 2
	gl_Position = B_in.gl_Position - vec4(b_normal * b_half_width, 0.0, 0.0);
	COPY_B
	#ifdef IN_U_NAME
	OUT_UV_NAME = vec2(IN_U_NAME[B], 0.0);
	#endif
	EmitVertex();

	// 3
	gl_Position = C_in.gl_Position - vec4(c_normal * c_half_width, 0.0, 0.0);
	COPY_C
	#ifdef IN_U_NAME
	OUT_UV_NAME = vec2(IN_U_NAME[C], 0.0);
	#endif
	EmitVertex();

	// 4
	gl_Position = C_in.gl_Position + vec4(c_normal * c_half_width, 0.0, 0.0);
	COPY_C
	#ifdef IN_U_NAME
	OUT_UV_NAME = vec2(IN_U_NAME[C], 0.0);
	#endif
	EmitVertex();
}
