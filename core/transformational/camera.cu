#include "camera.h"
#include "operations.h"

__host__ __device__ camera::camera(
    const ray viewRay,
    float aspect,
    float matrixScale,
    float matrixOffset,
    float focus,
    float apertura) :
    viewRay(viewRay),
    matrixScale(matrixScale),
    matrixOffset(matrixOffset),
    focus(focus),
    apertura(apertura)
{
    horizontal = aspect * vec4::getHorizontal(viewRay.getDirection());
    vertical = vec4::getVertical(viewRay.getDirection());
}

__host__ __device__ camera::camera(const ray viewRay, float aspect) : viewRay(viewRay)
{
    horizontal = aspect * vec4::getHorizontal(viewRay.getDirection());
    vertical = vec4::getVertical(viewRay.getDirection());
}

__device__ ray camera::getPixelRay(float u, float v, curandState* local_rand_state) {
    const float t = focus / (matrixOffset - focus);
    u = matrixScale * t * u + apertura * float(curand_uniform(local_rand_state));
    v = matrixScale * t * v + apertura * float(curand_uniform(local_rand_state));
    return ray(viewRay.point(matrixOffset), t * matrixOffset * viewRay.getDirection() - (u * horizontal + v * vertical));
}

camera* camera::create(const ray& viewRay, float aspect) {
    camera* cam;
    camera hostcam(viewRay, aspect);
    checkCudaErrors(cudaMalloc((void**)&cam, sizeof(camera)));
    cudaMemcpy(cam, &hostcam, sizeof(camera), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());
    return cam;
}

void camera::destroy(camera* cam) {
    checkCudaErrors(cudaFree(cam));
}