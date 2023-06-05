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
};

#endif