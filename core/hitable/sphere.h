#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include "operations.h"

class sphere;
__global__ void createSphere(sphere** sph, vec4 cen, float r, vec4 color, material* matptr);

class alignas(64) sphere : public hitable {
private:
    vec4 center{ 0.0f, 0.0f, 0.0f, 1.0f };
    vec4 color{ 0.0f,0.0f, 0.0f, 0.0f };
    float radius{ 0.0f };
    material* matptr{ nullptr };

public:
    __host__ __device__ sphere() {}
    __host__ __device__ ~sphere() {}
    __host__ __device__ void destroy() override {
        if (matptr) {
            delete matptr;
        }
    }
    __host__ __device__ sphere(vec4 cen, float r, vec4 color, material* matptr) : center(cen), radius(r), color(color), matptr(matptr) {};
    __host__ __device__ sphere(vec4 cen, float r, vec4 color) : center(cen), radius(r), color(color) {};
    __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const override;

    static sphere* create(vec4 cen, float r, vec4 color, material* matptr) {
        sphere** sph;
        checkCudaErrors(cudaMalloc((void**)&sph, sizeof(sphere**)));

        createSphere << <1, 1 >> > (sph, cen, r, color, matptr);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        sphere** hostsph = new sphere*;
        checkCudaErrors(cudaMemcpy(hostsph, sph, sizeof(sphere*), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(sph));

        return *hostsph;
    }
};

#endif