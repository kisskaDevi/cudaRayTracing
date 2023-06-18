#ifndef METALH
#define METALH

#include "material.h"

class metal;
__global__ void createMetal(metal** mat, float f, float angle);

class metal : public material {
private:
    float fuzz{ 0.0f };
    float angle{ 0.0f };
public:
    __host__ __device__ metal(float f, float angle) : fuzz(f), angle(angle){}
    __device__ vec4 scatter(const ray& r, const vec4& norm, curandState* local_rand_state) const override {
        vec4 reflect = normal(r.getDirection() + 2.0f * std::abs(dot(r.getDirection(), norm)) * norm);
        vec4 scattered = reflect + fuzz * random_in_unit_sphere(reflect, angle, local_rand_state);
        return (dot(scattered, norm) > 0.0f ? 1.0f : 0.0f) * scattered;
    }
    __device__ bool lightFound() const override {
        return false;
    }

    static metal* create(float f, float angle) {
        metal** mat;
        checkCudaErrors(cudaMalloc((void**)&mat, sizeof(metal**)));

        createMetal << <1, 1 >> > (mat, f, angle);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        metal** hostmat = new metal*;
        checkCudaErrors(cudaMemcpy(hostmat, mat, sizeof(metal*), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(mat));

        return *hostmat;
    }
};

__global__ void createMetal(metal** mat, float f, float angle) {
    *mat = new metal(f, angle);
}

#endif