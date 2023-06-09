#include "vec4.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

namespace Buffer {
    void* create(
        size_t                  size);
}

namespace Image{

    __global__ void clear(
        vec4*                   image,
        size_t                  width,
        size_t                  height);

    __global__ void combine(
        vec4*                   dst,
        vec4*                   src,
        size_t                  width,
        size_t                  height);

    __global__ void normalize(
        vec4*                   frameBuffer,
        size_t                  width,
        size_t                  height);

    void outPPM(
        vec4*                   frameBuffer,
        size_t                  width,
        size_t                  height,
        const std::string&      filename);

    void outPGM(
        vec4*                   frameBuffer,
        size_t                  width,
        size_t                  height,
        const std::string&      filename);

}