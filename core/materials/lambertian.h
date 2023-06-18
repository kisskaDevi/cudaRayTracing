#ifndef LAMBERTIANH
#define LAMBERTIANH

#include "material.h"

class lambertian;
__global__ void createLambertian(lambertian** mat, float angle);

class lambertian : public material {
private:
    float angle{ pi };
public:
    __host__ __device__ lambertian(float angle) : angle(angle){}
    __device__ virtual vec4 scatter(const ray& r, const vec4& norm, curandState* local_rand_state) const override {
        vec4 scattered = random_in_unit_sphere(norm, angle, local_rand_state);
        return (dot(norm, scattered) > 0.0f ? 1.0f : -1.0f) * scattered;
    }
    __device__ bool lightFound() const override {
        return false;
    }

    static lambertian* create(float angle) {
        lambertian** mat;
        checkCudaErrors(cudaMalloc((void**)&mat, sizeof(lambertian**)));

        createLambertian << <1, 1 >> > (mat, angle);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        lambertian** hostmat = new lambertian*;
        checkCudaErrors(cudaMemcpy(hostmat, mat, sizeof(lambertian*), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(mat));

        return *hostmat;
    }
};

__global__ void createLambertian(lambertian** mat, float angle) {
    *mat = new lambertian(angle);
}

#endif