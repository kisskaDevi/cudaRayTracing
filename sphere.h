#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere : public hitable {
private:
    vec4 center{ 0.0f,0.0f, 0.0f, 1.0f };
    float radius{ 0.0f };
    material* matptr{ nullptr };

public:
    __device__ sphere() {}
    __device__ ~sphere() {
        if (matptr) {
            delete matptr;
        }
    }
    __device__ sphere(vec4 cen, float r, material* matptr) : center(cen), radius(r), matptr(matptr) {};
    __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const;
};

__device__ bool sphere::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    vec4 oc = r.getOrigin() - center;

    float a = dot(r.getDirection(), r.getDirection());
    float b = dot(oc, r.getDirection());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;

    if (discriminant >= 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < 0.0f) {
            temp = (-b + sqrt(discriminant)) / a;
        }
        if (temp < tMax && temp > tMin) {
            rec.t = temp;
            rec.point = r.point(rec.t);
            rec.normal = (rec.point - center) / radius;
            rec.mat = matptr;
            return true;
        }
    }
    return false;
}


#endif