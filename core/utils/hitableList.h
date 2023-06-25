#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

class hitableList {
private:
    hitable* head{ nullptr };
    hitable* tail{ nullptr };

    __host__ __device__ void addSingle(hitable* object);
public:
    __host__ __device__ hitableList() {}
    __host__ __device__ ~hitableList();

    __device__ bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const;

    template<class... T>
    __host__ __device__ void add(T... objects) {
        for (auto& object : { objects... }) {
            addSingle(object);
        }
    }

    static hitableList* create();
    static void destroy(hitableList* list);
};

__global__ void addSingleInList(hitableList* list, hitable* object);

template<class... T>
void addInList(hitableList* list, T... objects) {
    for (auto& object : { objects... }) {
        addSingleInList << <1, 1 >> > (list, object);
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

#endif