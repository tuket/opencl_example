#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

layout (local_size_x = 128) in;
layout(set = 0, binding = 0) uniform Unifs {
    uniform int64_t start;
    uniform int64_t n;
    uniform int64_t N;
} unifs;

layout(set = 1, binding = 0) buffer OutBuffer {
    uint64_t outBuf[];
};

int64_t multMod(int64_t a, int64_t b, int64_t m)
{
	//return (a * b) % m;
	int64_t res = 0;
	while (b > 0) {
		if (b % 2 == 1)
			res = (res + a) % m;
		a = (2*a) % m;
		b /= 2;
	}
	return res;
}

void main()
{
    uint myId = gl_GlobalInvocationID.x;
    outBuf[myId] = 0;
    int64_t start = 1 + unifs.start + myId * unifs.n;
    int64_t end = start + unifs.n;
    for(int64_t i = start; i < end; i++) {
        //if(multMod(11, i, unifs.N) == 1) {
		if(11 * i % unifs.N == 1) {
            outBuf[myId] = i;
            break;
        }
    }
}