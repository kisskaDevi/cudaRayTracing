#include "vec4.h"

__host__ __device__ vec4 normal(vec4 v) {
    return v / v.length();
}

__device__ vec4 random_in_unit_sphere(const vec4& direction, const float& angle, curandState* local_rand_state) {
    float phi = 2 * pi * curand_uniform(local_rand_state);
    float theta = angle * curand_uniform(local_rand_state);

    float x = std::sin(theta) * std::cos(phi);
    float y = std::sin(theta) * std::sin(phi);
    float z = std::cos(theta);

    return normal(x * vec4::getHorizontal(direction) + y * vec4::getVertical(direction) + z * direction);
}