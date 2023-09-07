//! Collection of blending modes! Use relavent defines to select which
//! functions to produce, to maybe decrease build time.
//! Or, use BLEND_ALL to build everything
#ifdef BLEND_ALL
#define BLEND_NORMAL
#define BLEND_MULTIPLY
#define BLEND_ADD
#define BLEND_OVERLAY
#endif
// Helper macro - user specifies an expression to generate mixed RGB with no regard for alpha,
// and from.a and onto.a are handled automatically:
// * As onto.a decreases, from is treated as a normal blend on transparency.
// * As from.a decreases, the blended rgb is faded into unchaged value of onto.
// This simulates the behavior I observe in programs like krita, but may not be correct for all modes.
// From and Onto should be the name of a vec4, not an expression.
// expr should evaluate to vec3
// (as of right now, i'm fairly certain this is wrong, but it's in one place so can be easily
// modified later! :3 )
#define BLEND_RGB_AUTO_ALPHA(from, onto, expr)\
    mix(\
        /* Transparent bg = use from as-is.*/\
        from,\
        vec4(\
            /* keep alpha of onto*/\
            mix(\
                /* Fade between blended and original based on from.a*/\
                expr,\
                onto.rgb,\
                from.a\
            ),\
            onto.a\
        ),\
        onto.a\
    )\

#ifdef BLEND_NORMAL
vec4 blend_normal(vec4 from, vec4 onto) {
    return vec4(from.rgb * from.a, from.a) + onto * (1.0 - from.a);
}
#endif

#ifdef BLEND_MULTIPLY
vec4 blend_multiply(vec4 from, vec4 onto) {
    return mix(
        from,
        // Keep alpha of onto. Mix between rgb multiplied color and unchanged bg based on from.a
        vec4(mix(onto.rgb * from.rgb, onto.rgb, from.a), onto.a),
        // Transparent bg = use from as-is.
        onto.a
    );
}
#endif
/* TODO: unimplemented lol
#ifdef BLEND_SCREEN
vec4 blend_screen(vec4 from, vec4 onto) {
    return mix(
        from,
        // Keep alpha of onto. Mix between rgb multiplied color and unchanged bg based on from.a
        vec4(
            mix(
                todo!,
                onto.rgb,
                from.a
            ),
            onto.a
        ),
        // Transparent bg = use from as-is.
        onto.a
    );
}
#endif*/

#ifdef BLEND_ADD
vec4 blend_add(vec4 from, vec4 onto) {
    return mix(
        from,
        // Keep alpha of onto. Mix between rgb added color and unchanged bg based on from.a
        vec4(mix(onto.rgb + from.rgb, onto.rgb, from.a), onto.a),
        // Transparent bg = use from as-is.
        onto.a
    );
}
#endif

#ifdef BLEND_OVERLAY
// Vectorized ver of https://github.com/jamieowen/glsl-blend/tree/master
vec3 blend_overlay_rgb(vec3 from, vec3 onto) {
    bvec3 lt_half = lessThan(onto, vec3(0.5));
    return mix(
        1.0-2.0*(1.0-onto)*(1.0-from),
        2.0 * onto * from,
        lt_half
    );
}
vec4 blend_overlay(vec4 from, vec4 onto) {
    return BLEND_RGB_AUTO_ALPHA(from, onto, blend_overlay_rgb(from.rgb, onto.rgb));
}
#endif