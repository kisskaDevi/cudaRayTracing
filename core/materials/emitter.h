#ifndef EMITTERH
#define EMITTERH

#include "material.h"
#include "operations.h"

class emitter;
__global__ void createEmitter(emitter** mat);

class emitter : public material {
private:
public:
    __host__ __device__ emitter(){}
    __device__ vec4 scatter(const ray& r, const vec4& norm, curandState* local_rand_state) const override {
        return vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __device__ bool lightFound() const override {
        return true;
    }

    static emitter* create() {
        emitter** mat;
        checkCudaErrors(cudaMalloc((void**)&mat, sizeof(emitter**)));

        createEmitter << <1, 1 >> > (mat);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        emitter** hostmat = new emitter*;
        checkCudaErrors(cudaMemcpy(hostmat, mat, sizeof(emitter*), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(mat));

        return *hostmat;
    }
};

__global__ void createEmitter(emitter** mat) {
    *mat = new emitter;
}

#endif