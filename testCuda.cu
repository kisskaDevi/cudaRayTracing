#include <iostream>
#include <math.h>
#include <chrono>
#include <fstream>
#include <string>
#include <curand_kernel.h>
#include "vec4.h"
#include "ray.h"
#include "triangle.h"
#include "sphere.h"
#include "hitableList.h"
#include "camera.h"
#include "material.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec4 color(ray r, size_t maxIterations, hitableList* list, curandState* local_rand_state) {
    vec4 color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
    bool lightFound = false;
    for (; maxIterations > 0; maxIterations--) {
        hitRecord rec;
        if (!lightFound && r.getDirection().length2() != 0.0f && list->hit(r, 0.001f, FLT_MAX, rec)) {
            color = min(rec.mat->getAlbedo(), color);
            r = ray(rec.point, rec.mat->scatter(r, rec.normal, local_rand_state));
            lightFound |= rec.mat->lightFound();
        } else {
            break;
        }
    }
    return lightFound ? color : vec4(0.0f, 0.0f, 0.0f, 0.0f);
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec4* frameBuffer, size_t max_x, size_t max_y, size_t hitCount, size_t raysCount, size_t samplesCount, camera* cam, hitableList* list, curandState* rand_state){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i < max_x) && (j < max_y)) {
        int pixel_index = j * max_x + i;
        curandState local_rand_state = rand_state[pixel_index];
        for (size_t sampleIndex = 0; sampleIndex < samplesCount; sampleIndex++) {
            float u = 2.0f * float(i) / float(max_x) - 1.0f;
            float v = 2.0f * float(j) / float(max_y) - 1.0f;

            size_t hitCounter = 0;
            vec4 col = vec4(0.0f, 0.0f, 0.0f, 0.0f);
            for (size_t rayIndex = 0 ; rayIndex < raysCount; rayIndex++) {
                col += color(cam->getPixelRay(u, v, &local_rand_state), hitCount, list, &local_rand_state);
                hitCounter += col.length2() > 0.0f ? 1 : 0;
            }
            col /= hitCounter;
            frameBuffer[pixel_index] += col;
        }
        frameBuffer[pixel_index] /= float(samplesCount);
    }
}

__global__ void create_world(camera* cam, int width, int height, hitableList* d_list) {
    d_list->add(new sphere(vec4(0.0f, 0.0f, 0.5f, 1.0f), 0.5, new lambertian(vec4(0.8, 0.3, 0.3, 1.0f))));
    d_list->add(new sphere(vec4(0.0f, 1.0f, 0.5f, 1.0f), 0.5, new metal(vec4(0.8f, 0.8f, 0.8f, 1.0f), 3.0f)));
    d_list->add(new sphere(vec4(0.0f, -1.0f, 0.5f, 1.0f), 0.5, new glass(vec4(0.9f, 0.9f, 0.9f, 1.0f), 1.5f, 0.96f)));

    d_list->add(
        //down
        new triangle(   vec4(-3.0f, -3.0f, 0.0f, 1.0f), vec4(-3.0f, 3.0f, 0.0f, 1.0f), vec4(3.0f, -3.0f, 0.0f, 1.0f),
                        vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f),
                        new lambertian(vec4(0.6f, 0.6f, 0.8f, 1.0f))),
        new triangle(   vec4(3.0f, 3.0f, 0.0f, 1.0f), vec4(-3.0f, 3.0f, 0.0f, 1.0f), vec4(3.0f, -3.0f, 0.0f, 1.0f),
                        vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f),
                        new lambertian(vec4(0.6f, 0.6f, 0.8f, 1.0f))),
        //top
        new triangle(   vec4(-3.0f, -3.0f, 3.0f, 1.0f), vec4(-3.0f, 3.0f, 3.0f, 1.0f), vec4(3.0f, -3.0f, 3.0f, 1.0f),
                        vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f),
                        new emitter(vec4(0.8f, 0.8f, 0.8f, 1.0f))),
        new triangle(   vec4(3.0f, 3.0f, 3.0f, 1.0f), vec4(-3.0f, 3.0f, 3.0f, 1.0f), vec4(3.0f, -3.0f, 3.0f, 1.0f),
                        vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f),
                        new emitter(vec4(0.8f, 0.8f, 0.8f, 1.0f))),
        //back
        new triangle(   vec4(-3.0f, -3.0f, 0.0f, 1.0f), vec4(-3.0f, 3.0f, 0.0f, 1.0f), vec4(-3.0f, 3.0f, 3.0f, 1.0f),
                        vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f),
                        new metal(vec4(0.8f, 0.8f, 0.8f, 1.0f), 3.0f)),
        new triangle(   vec4(-3.0f, -3.0f, 3.0f, 1.0f), vec4(-3.0f, 3.0f, 3.0f, 1.0f), vec4(-3.0f, -3.0f, 0.0f, 1.0f),
                        vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f),
                        new metal(vec4(0.8f, 0.8f, 0.8f, 1.0f), 3.0f)),
        //front
        new triangle(   vec4(3.0f, -3.0f, 0.0f, 1.0f), vec4(3.0f, 3.0f, 0.0f, 1.0f), vec4(3.0f, 3.0f, 3.0f, 1.0f),
                        vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f),
                        new metal(vec4(0.8f, 0.8f, 0.8f, 1.0f), 3.0f)),
        new triangle(   vec4(3.0f, -3.0f, 3.0f, 1.0f), vec4(3.0f, 3.0f, 3.0f, 1.0f), vec4(3.0f, -3.0f, 0.0f, 1.0f),
                        vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f),
                        new metal(vec4(0.8f, 0.8f, 0.8f, 1.0f), 3.0f)),
        //left
        new triangle(   vec4(-3.0f, -3.0f, 0.0f, 1.0f), vec4(3.0f, -3.0f, 0.0f, 1.0f), vec4(3.0f, -3.0f, 3.0f, 1.0f),
                        vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f),
                        new lambertian(vec4(0.8f, 0.5f, 0.0f, 1.0f))),
        new triangle(   vec4(-3.0f, -3.0f, 0.0f, 1.0f), vec4(-3.0f, -3.0f, 3.0f, 1.0f), vec4(3.0f, -3.0f, 3.0f, 1.0f),
                        vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f),
                        new lambertian(vec4(0.8f, 0.5f, 0.0f, 1.0f))),
        //right
        new triangle(   vec4(-3.0f, 3.0f, 0.0f, 1.0f), vec4(3.0f, 3.0f, 0.0f, 1.0f), vec4(3.0f, 3.0f, 3.0f, 1.0f),
                        vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f),
                        new lambertian(vec4(0.0f, 0.4f, 0.8f, 1.0f))),
        new triangle(   vec4(-3.0f, 3.0f, 0.0f, 1.0f), vec4(-3.0f, 3.0f, 3.0f, 1.0f), vec4(3.0f, 3.0f, 3.0f, 1.0f),
                        vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f),
                        new lambertian(vec4(0.0f, 0.4f, 0.8f, 1.0f)))
    );


    d_list->add(new sphere(vec4( 1.5f, -1.5f, 0.5f, 1.0f), 0.5, new emitter(vec4(0.99f, 0.8f, 0.2f, 1.0f))));
    d_list->add(new sphere(vec4( 1.5f,  1.5f, 0.5f, 1.0f), 0.5, new emitter(vec4(0.2f, 0.8f, 0.99f, 1.0f))));
    d_list->add(new sphere(vec4(-1.5f, -2.5f, 0.5f, 1.0f), 0.5, new emitter(vec4(0.99f, 0.4f, 0.85f, 1.0f))));
    d_list->add(new sphere(vec4(-1.5f,  2.5f, 0.5f, 1.0f), 0.5, new emitter(vec4(0.4f, 0.99f, 0.5f, 1.0f))));

    d_list->add(new sphere(vec4(-0.5f, -0.5f, 1.5f, 1.0f), 0.5, new emitter(vec4(1.0f, 1.0f, 1.0f, 1.0f))));
    d_list->add(new sphere(vec4(-0.5f, 0.5f, 1.5f, 1.0f), 0.5, new emitter(vec4(0.0f, 1.0f, 1.0f, 1.0f))));
    *cam = camera(ray(vec4(2.0f, 0.0f, 0.5f, 1.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f)), float(width) / float(height));
}

__global__ void free_world(hitableList* d_list) {
    delete d_list;
}

void outImage(vec4* frameBuffer, size_t width, size_t height, const std::string& filename) {
    std::ofstream image(filename);
    image << "P3\n" << width << " " << height << "\n255\n";
    for (int j = 0; j < height; j++) {
        for (int i = width - 1; i >= 0; i--) {
            size_t pixel_index = j * width + i;
            image   << static_cast<uint32_t>(255.99f * frameBuffer[pixel_index].r()) << " "
                    << static_cast<uint32_t>(255.99f * frameBuffer[pixel_index].g()) << " "
                    << static_cast<uint32_t>(255.99f * frameBuffer[pixel_index].b()) << "\n";
        }
    }
    image.close();
}

int testCuda()
{
    size_t width = 1920;
    size_t height = 1080;
    size_t frameBufferSize = 4 * width * height * sizeof(vec4);

    size_t xThreads = 4;
    size_t yThreads = 4;

    vec4* frameBuffer;
    checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, width * height * sizeof(curandState)));
    camera* cam;
    checkCudaErrors(cudaMalloc((void**)&cam, sizeof(camera)));
    hitableList* d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, sizeof(hitableList)));

    create_world<<<1, 1>>>(cam, width, height, d_list);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(width / xThreads + 1, height / yThreads + 1);
    dim3 threads(xThreads, yThreads);

    render_init<<<blocks, threads>>>(width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();

    render<<<blocks, threads>>>(frameBuffer, width, height, 10, 100, 10, cam, d_list, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "render time = " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

    outImage(frameBuffer, width, height, "image.ppm");

    checkCudaErrors(cudaDeviceSynchronize());

    free_world<<<1, 1>>>(d_list);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(frameBuffer));

    return 0;
}
