#include "hitableList.h"
#include "operations.h"

__host__ __device__ void destroy(hitable* object) {
    if (object->next) {
        destroy(object->next);
    }
    object->destroy();
    delete object;
}

__host__ __device__ hitableList::~hitableList() {
    destroy(head);
}

__host__ __device__ void hitableList::addSingle(hitable* object) {
    if (head) {
        tail->next = object;
    } else {
        head = object;
        head->next = object;
    }
    tail = object;
}

__device__ bool hitableList::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    float depth = tMax;
    for (hitable* object = head; object; object = object->next) {
        if (object->hit(r, tMin, depth, rec)) {
            depth = rec.t;
        }
    }
    return depth != tMax;
}

hitableList* hitableList::create() {
    hitableList* list;
    checkCudaErrors(cudaMalloc((void**)&list, sizeof(hitableList)));
    checkCudaErrors(cudaGetLastError());
    return list;
}

__global__ void addSingleInList(hitableList* list, hitable* object) {
    list->add(object);
}
