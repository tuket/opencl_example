#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

layout (local_size_x = 256) in;
layout(set = 0, binding = 0) uniform Unifs {
    uniform int64_t numThreads;
    uniform int64_t n;
} unifs;

void main()
{
    uint64_t nyId = gl_GlobalInvocationID.x;
    
}