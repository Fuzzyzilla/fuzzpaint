# Assumptions

To ease development, several assumptions are made about the graphics device. Of course, the plan will be to reduce reliance on these assumptions as development furthers. As I make these assumptions (and discover the ones I made prior to assembling this list :P ) I will notate them here in order to both serve as a todo list of blockers for running on any device and to figure out what's missing should I find a device which doesn't work.

* VK1.3 OR maintenance4 (workgroup size specialization)
* dualSrcBlend (erasers) (~100% on desktop)
* dynamicRendering (Pure laziness)
* multiDrawIndirect (tessellated stroke draw batching)
* geometry shading (WideLine gizmos) (~100% on desktop)
* B8G8R8A8_SRGB surface format (pure laziness, Fixme!!)
