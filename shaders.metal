#include <metal_stdlib>

using namespace metal;

kernel void read_buffer(
  device half4* input [[buffer(0)]],
  device half4* output [[buffer(1)]],
  uint3 id [[thread_position_in_grid]],
  uint3 gridsize [[threads_per_grid]])
{
  output[id.x + id.y*gridsize.x] = input[id.x + id.y*gridsize.x];
}

kernel void read_texture_float(
  texture2d<float, access::read> input [[texture(0)]],
  device half4* output [[buffer(0)]],
  uint3 id [[thread_position_in_grid]],
  uint3 gridsize [[threads_per_grid]])
{
  output[id.x + id.y*gridsize.x] = half4(input.read(id.xy));
}

kernel void read_texture_half(
  texture2d<half, access::read> input [[texture(0)]],
  device half4* output [[buffer(0)]],
  uint3 id [[thread_position_in_grid]],
  uint3 gridsize [[threads_per_grid]])
{
  output[id.x + id.y*gridsize.x] = input.read(id.xy);
}

kernel void write_buffer(
  constant half4& value [[buffer(0)]],
  device half4* output [[buffer(1)]],
  uint3 id [[thread_position_in_grid]],
  uint3 gridsize [[threads_per_grid]])
{
  output[id.x + id.y*gridsize.x] = value;
}

kernel void write_texture_float(
  constant half4& value [[buffer(0)]],
  texture2d<float, access::write> output [[texture(0)]],
  uint3 id [[thread_position_in_grid]])
{
  output.write(float4(value), id.xy);
}

kernel void write_texture_half(
  constant half4& value [[buffer(0)]],
  texture2d<half, access::write> output [[texture(0)]],
  uint3 id [[thread_position_in_grid]])
{
  output.write(value, id.xy);
}

kernel void copy_from_buffer(
  device half4* input [[buffer(0)]],
  device float4* output [[buffer(1)]],
  uint3 id [[thread_position_in_grid]],
  uint3 gridsize [[threads_per_grid]])
{
  output[id.x + id.y*gridsize.x] = float4(input[id.x + id.y*gridsize.x]);
}

kernel void copy_from_texture(
  texture2d<float, access::read> input [[texture(0)]],
  device float4* output [[buffer(0)]],
  uint3 id [[thread_position_in_grid]],
  uint3 gridsize [[threads_per_grid]])
{
  output[id.x + id.y*gridsize.x] = input.read(id.xy);
}
