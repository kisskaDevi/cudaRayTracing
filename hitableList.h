#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

class hitableList {
private:
    hitable* head{ nullptr };
    hitable* tail{ nullptr };

    __device__ void addSingle(hitable* object);
public:
    __device__ hitableList() {}
    __device__ ~hitableList();

    __device__ bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const;
    template<class... T> __device__ void add(T... objects);
};

__device__ void destroy(hitable* object) {
    if (object->next) {
        destroy(object->next);
        delete object;
    }
}

__device__ hitableList::~hitableList() {
    destroy(head);
}

__device__ void hitableList::addSingle(hitable* object) {
    if (head) {
        tail->next = object;
        tail = object;
    } else {
        head = object;
        tail = object;
        head->next = tail;
    }
}

template<class... T>
__device__ void hitableList::add(T... objects) {
    for (auto& object : { objects... }) {
        addSingle(object);
    }
}

__device__ bool hitableList::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    hitRecord tempRec;
    bool hitAnything = false;
    float depth = tMax;
    for (hitable* object = head; object; object = object->next) {
        if (object->hit(r, tMin, depth, tempRec)) {
            hitAnything = true;
            depth = tempRec.t;
            rec = tempRec;
        }
    }
    return hitAnything;
}

#endif