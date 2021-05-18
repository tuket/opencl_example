#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

layout (local_size_x = 128) in;
layout(set = 0, binding = 0) uniform Unifs {
    uniform int64_t n;
    uniform int64_t N;
} unifs;

layout(set = 1, binding = 0) buffer OutBuffer {
    uint64_t outBuf[];
};

void main()
{
    uint myId = gl_GlobalInvocationID.x;
    outBuf[myId] = 0;
    uint64_t start = myId * unifs.n;
    uint64_t end = start + unifs.n;
    for(uint64_t i = start; i < end; i++) {
        if((11 * i) % unifs.N == 1) {
            outBuf[myId] = i;
            break;
        }
    }
}