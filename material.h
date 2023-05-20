#ifndef MATERIALH
#define MATERIALH

#include "ray.h"

#define pi 3.14159265358979323846

namespace {
    __device__ vec4 getHorizontal(const vec4& d) {
        float D = std::sqrt(d.x() * d.x() + d.y() * d.y());
        float x = d.y();
        float y = -d.x();
        return vec4(x, y, 0.0f, 0.0f) / D;
    }

    __device__ vec4 getVertical(const vec4& d) {
        float z = std::sqrt(d.x() * d.x() + d.y() * d.y());
        float x = -d.z() * d.x() / z;
        float y = -d.z() * d.y() / z;
        return vec4(x, y, z, 0.0f) / d.length();
    }

    __device__ vec4 random_in_unit_sphere(const vec4& norm, const vec4& direction, const float& angle, curandState* local_rand_state) {
        float phi = 2 * pi * curand_uniform(local_rand_state);
        float theta = angle * curand_uniform(local_rand_state);

        vec4 horizontal = getHorizontal(direction);
        vec4 vertical = getVertical(direction);
        float x = std::sin(theta) * std::cos(phi);
        float y = std::sin(theta) * std::sin(phi);
        float z = std::cos(theta);

        return normal(x * horizontal + y * vertical + z * direction);
    }
}

class material {
public:
	__device__ virtual vec4 scatter(const ray& r, const vec4& normal, curandState* local_rand_state) const = 0;
    __device__ virtual vec4 getAlbedo() const = 0;
    __device__ virtual bool lightFound() const = 0;
};

class lambertian : public material {
private:
    vec4 albedo{ 0.0f, 0.0f, 0.0f, 0.0f };
    float angle{ pi };
public:
    __device__ lambertian(const vec4& a) : albedo(a) {}
    __device__ virtual vec4 scatter(const ray& r, const vec4& norm, curandState* local_rand_state) const override {
        vec4 scattered = random_in_unit_sphere(norm, norm, angle, local_rand_state);
        return (dot(norm, scattered) > 0 ? 1.0f : -1.0f) * scattered;
    }
    __device__ virtual vec4 getAlbedo() const override {
        return albedo;
    }
    __device__ bool lightFound() const override {
        return false;
    }
};

class metal : public material {
private:
    vec4 albedo{ 0.0f, 0.0f, 0.0f, 0.0f };
    float fuzz{ 0.0f };
    float angle{ 0.3f * pi };
public:
    __device__ metal(const vec4& a, float f) : albedo(a), fuzz(f){}
    __device__ vec4 scatter(const ray& r, const vec4& norm, curandState* local_rand_state) const override {
        vec4 reflect = normal(r.getDirection() + 2.0f * std::abs(dot(r.getDirection(), norm)) * norm);
        vec4 scattered = reflect + fuzz * random_in_unit_sphere(norm, reflect, angle, local_rand_state);
        return (dot(scattered, norm) > 0.0f ? 1.0f : 0.0f) * scattered;
    }
    __device__ virtual vec4 getAlbedo() const override {
        return albedo;
    }
    __device__ bool lightFound() const override {
        return false;
    }
};

class emitter : public material {
private:
    vec4 albedo{ 0.0f, 0.0f, 0.0f, 0.0f };
public:
    __device__ emitter(const vec4& a) : albedo(a) {}
    __device__ vec4 scatter(const ray& r, const vec4& norm, curandState* local_rand_state) const override {
        return vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __device__ virtual vec4 getAlbedo() const override {
        return albedo;
    }
    __device__ bool lightFound() const override {
        return true;
    }
};

class glass : public material {
private:
    vec4 albedo{ 0.0f, 0.0f, 0.0f, 0.0f };
    float refractiveIndex{ 1.0f };
    float refractProb{ 1.0f };

public:
    __device__ glass(const vec4& a, const float& refractiveIndex, const float& refractProb) : albedo(a), refractiveIndex(refractiveIndex), refractProb(refractProb){}
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

                scattered = sinTheta * tau  - cosTheta * n;
            }
        }
        if(scattered.length2() == 0.0f){
            vec4 reflect = normal(r.getDirection() + 2.0f * std::abs(dot(r.getDirection(), norm)) * norm);
            scattered = reflect;
            scattered = (dot(scattered, norm) > 0.0f ? 1.0f : 0.0f) * scattered;
        }

        return scattered;
    }
    __device__ virtual vec4 getAlbedo() const override {
        return albedo;
    }
    __device__ bool lightFound() const override {
        return false;
    }
};

#endif
