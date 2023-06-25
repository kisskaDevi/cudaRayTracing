#ifndef TRIANGLE
#define TRIANGLE

#include "hitable.h"
#include "buffer.h"
#include "material.h"

struct alignas(64) vertex {
    vec4 point{0.0f, 0.0f, 0.0f, 1.0f};
    vec4 normal{ 0.0f, 0.0f, 0.0f, 0.0f };
    vec4 color{ 0.0f, 0.0f, 0.0f, 0.0f };
    properties props;
    __host__ __device__ vertex() {}
    __host__ __device__ vertex(vec4 point, vec4 normal, vec4 color, const properties& props):
        point(point), normal(normal), color(color), props(props)
    {}
};

class triangle;
__global__ void createTriangle(triangle** tr, const size_t i0, const size_t i1, const size_t i2, vertex* vertexBuffer, material* matptr);

class alignas(64) triangle : public hitable {
private:
    size_t index0, index1, index2;
    vertex* vertexBuffer{ nullptr };
    material* matptr{ nullptr };

public:
    __host__ __device__ triangle() {}
    __host__ __device__ ~triangle() {}
    __host__ __device__ void destroy() {}

    __host__ __device__ triangle(const size_t& i0, const size_t& i1, const size_t& i2, vertex* vertexBuffer, material* matptr)
        : index0(i0), index1(i1), index2(i2), vertexBuffer(vertexBuffer), matptr(matptr) {};
    __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const override;

    static triangle* create(const size_t& i0, const size_t& i1, const size_t& i2, vertex* vertexBuffer, material* matptr) {
        triangle** tr;
        checkCudaErrors(cudaMalloc((void**)&tr, sizeof(triangle**)));

        createTriangle << <1, 1 >> > (tr, i0, i1, i2, vertexBuffer, matptr);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        triangle** hosttr = new triangle*;
        checkCudaErrors(cudaMemcpy(hosttr, tr, sizeof(triangle*), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(tr));

        return *hosttr;
    }
};

#endif