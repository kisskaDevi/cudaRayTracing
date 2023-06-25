#ifndef BASEMATERIALH
#define BASEMATERIALH

#include "material.h"
#include "operations.h"

class baseMaterial;
__global__ void createBaseMaterial(baseMaterial** mat);
__global__ void destroyBaseMaterial(baseMaterial* mat);

class baseMaterial : public material {
private:

public:
    __host__ __device__ baseMaterial() {}
    __device__ vec4 scatter(const ray& r, const vec4& norm, const properties& props, curandState* local_rand_state) const override
    {
        vec4 scattered = vec4(0.0f, 0.0f, 0.0f, 0.0f);
        const vec4& d = r.getDirection();

        if (curand_uniform(local_rand_state) <= props.refractProb) {
            const vec4 n = (dot(d, norm) <= 0.0f) ? norm : -norm;
            const float eta = (dot(d, norm) <= 0.0f) ? (1.0f / props.refractiveIndex) : props.refractiveIndex;

            float cosPhi = dot(d, n);
            float sinTheta = eta * std::sqrt(1.0f - cosPhi * cosPhi);
            if (std::abs(sinTheta) <= 1.0f) {
                float cosTheta = std::sqrt(1.0f - sinTheta * sinTheta);
                vec4 tau = normal(d - dot(d, n) * n);
                scattered = sinTheta * tau - cosTheta * n;
            }

            if (scattered.length2() == 0.0f) {
                vec4 reflect = normal(d + 2.0f * std::abs(dot(d, norm)) * norm);
                scattered = (dot(reflect, norm) > 0.0f ? 1.0f : 0.0f) * reflect;
            }
        } else {
            if (props.fuzz > 0.0f) {
                vec4 reflect = normal(d + 2.0f * std::abs(dot(d, norm)) * norm);
                scattered = reflect + props.fuzz * random_in_unit_sphere(reflect, props.angle, local_rand_state);
                scattered = (dot(norm, scattered) > 0.0f ? 1.0f : 0.0f) * scattered;
            } else {
                scattered = random_in_unit_sphere(norm, props.angle, local_rand_state);
                scattered = (dot(norm, scattered) > 0.0f ? 1.0f : -1.0f) * scattered;
            }
        }

        return scattered;
    }
    __device__ bool lightFound() const override {
        return false;
    }

    static baseMaterial* create() {
        baseMaterial** mat;
        checkCudaErrors(cudaMalloc((void**)&mat, sizeof(baseMaterial**)));

        createBaseMaterial << <1, 1 >> > (mat);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        baseMaterial** hostmat = new baseMaterial*;
        checkCudaErrors(cudaMemcpy(hostmat, mat, sizeof(baseMaterial*), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(mat));

        return *hostmat;
    }

    static void destroy(baseMaterial* mat) {
        destroyBaseMaterial << <1, 1 >> > (mat);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

    }
};

__global__ void createBaseMaterial(baseMaterial** mat) {
    *mat = new baseMaterial();
}

__global__ void destroyBaseMaterial(baseMaterial* mat) {
    delete mat;
}

#endif