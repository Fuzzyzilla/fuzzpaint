#!/bin/bash
# move to blend directory
cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null
# run a validator on the modes.glsl file (with dummy entry point from validate.comp)
glslangValidator -S comp -Od -DBLEND_ALL=1 -V -o VALIDATE_OUT.sprv validate.comp
# i am too smol brain to figure out how to disable output entirely :<
rm VALIDATE_OUT.sprv