// Consumer: calls externally-defined functions whose definitions live in
// `lib.hlsl`. With -fspv-allow-import these become OpDecorate ... Import.

float scale_by(float v, float k);
float add_offset(float v, float k);

RWStructuredBuffer<float> output : register(u0);

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    float v = (float)dtid.x;
    output[dtid.x] = add_offset(scale_by(v, 2.0), 10.0);
}
