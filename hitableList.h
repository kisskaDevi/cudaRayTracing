#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

class hitableList {
private:
    hitable** list;
    size_t listSize{ 0 };

public:
    __device__ hitableList() {}
    __device__ ~hitableList();

    __device__ bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const;
    __device__ void add(hitable* object);
};

__device__ hitableList::~hitableList() {
    for (size_t i = 0; i < listSize; i++) {
        delete *(list + i);
    }
    delete[] list;
}

__device__ void hitableList::add(hitable* object) {
    hitable** newlist = new hitable*[listSize + 1];
    for (size_t i = 0; i < listSize; i++) {
        *(newlist + i) = *(list + i);
    }
    *(newlist + listSize) = object;
    delete[] list;
    list = newlist;
    listSize++;
}

__device__ bool hitableList::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    hitRecord tempRec;
    bool hitAnything = false;
    float depth = tMax;
    for (int i = 0; i < listSize; i++) {
        if (list[i]->hit(r, tMin, depth, tempRec)) {
            hitAnything = true;
            depth = tempRec.t;
            rec = tempRec;
        }
    }
    return hitAnything;
}

#endif