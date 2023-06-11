#ifndef GLASSH
#define GLASSH

#include "material.h"

class glass : public material {
private:
    float refractiveIndex{ 1.0f };
    float refractProb{ 1.0f };

public:
    __host__ __device__ glass(const float& refractiveIndex, const float& refractProb) : refractiveIndex(refractiveIndex), refractProb(refractProb) {}
    __device__ vec4 scatter(const ray& r, const vec4& norm, curandState* local_rand_state) const override {
        vec4 scattered = vec4(0.0f, 0.0f, 0.0f, 0.0f);
        if (curand_uniform(local_rand_state) <= refractProb) {
            const vec4& d = r.getDirection();
            const vec4 n = (dot(d, norm) <= 0.0f) ? norm : -norm;
            const float eta = (dot(d, norm) <= 0.0f) ? (1.0f / refractiveIndex) : refractiveIndex;

            float cosPhi = dot(d, n);
            float sinTheta = eta * std::sqrt(1.0f - cosPhi * cosPhi);
            if (std::abs(sinTheta) <= 1.0f) {
                float cosTheta = std::sqrt(1.0f - sinTheta * sinTheta);
                vec4 tau = normal(d - dot(d, n) * n);
                scattered = sinTheta * tau - cosTheta * n;
            }
        }
        if (scattered.length2() == 0.0f) {
            vec4 reflect = normal(r.getDirection() + 2.0f * std::abs(dot(r.getDirection(), norm)) * norm);
            scattered = (dot(reflect, norm) > 0.0f ? 1.0f : 0.0f) * reflect;
        }

        return scattered;
    }
    __device__ bool lightFound() const override {
        return false;
    }
};

#endif