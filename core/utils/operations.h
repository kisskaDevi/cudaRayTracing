#include "vec4.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

namespace Buffer {
    vec4* create(
        size_t                  size);
}

namespace Image{

    void clear(
        vec4*                   image,
        size_t                  size);

    void combine(
        vec4*                   dst,
        vec4*                   src,
        size_t                  size);

    void normalize(
        vec4*                   frameBuffer,
        size_t                  size);

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