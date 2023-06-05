#include <iostream>
#include <math.h>
#include <chrono>
#include <fstream>
#include <string>
#include <curand_kernel.h>

#include "triangle.h"
#include "sphere.h"
#include "hitableList.h"
#include "camera.h"
#include "emitter.h"
#include "glass.h"
#include "lambertian.h"
#include "metal.h"

#include "rayTracingGraphics.h"
#include "graphicsManager.h"

__global__ void createWorld(hitableList* list) {
    list->add(
        new sphere(     vec4(0.0f, 0.0f, 0.5f, 1.0f),
                        0.5,
                        new lambertian(vec4(0.8, 0.3, 0.3, 1.0f))),
        new sphere(     vec4(0.0f, 1.0f, 0.5f, 1.0f),
                        0.5,
                        new metal(vec4(0.8f, 0.8f, 0.8f, 1.0f), 3.0f)),
        new sphere(     vec4(0.0f, -1.0f, 0.5f, 1.0f),
                        0.5,
                        new glass(vec4(0.9f, 0.9f, 0.9f, 1.0f), 1.5f, 0.96f))
    );

    list->add(
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
                        new emitter(vec4(0.9f, 0.9f, 0.9f, 1.0f))),
        new triangle(   vec4(3.0f, 3.0f, 3.0f, 1.0f), vec4(-3.0f, 3.0f, 3.0f, 1.0f), vec4(3.0f, -3.0f, 3.0f, 1.0f),
                        vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f),
                        new emitter(vec4(0.9f, 0.9f, 0.9f, 1.0f))),
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

    //d_list->add(new sphere(vec4( 1.5f, -1.5f, 0.2f, 1.0f), 0.2, new emitter(vec4(0.99f, 0.8f, 0.2f, 1.0f))));
    //d_list->add(new sphere(vec4( 1.5f,  1.5f, 0.2f, 1.0f), 0.2, new emitter(vec4(0.2f, 0.8f, 0.99f, 1.0f))));
    //d_list->add(new sphere(vec4(-1.5f, -1.5f, 0.2f, 1.0f), 0.2, new emitter(vec4(0.99f, 0.4f, 0.85f, 1.0f))));
    //d_list->add(new sphere(vec4(-1.5f,  1.5f, 0.2f, 1.0f), 0.2, new emitter(vec4(0.4f, 0.99f, 0.5f, 1.0f))));
    //d_list->add(new sphere(vec4(-0.5f, -0.5f, 0.2f, 1.0f), 0.2, new emitter(vec4(0.65f, 0.0f, 0.91f, 1.0f))));
    //d_list->add(new sphere(vec4( 0.5f,  0.5f, 0.2f, 1.0f), 0.2, new emitter(vec4(0.8f, 0.7f, 0.99f, 1.0f))));
    //d_list->add(new sphere(vec4(-0.5f,  0.5f, 0.2f, 1.0f), 0.2, new emitter(vec4(0.59f, 0.5f, 0.9f, 1.0f))));
    //d_list->add(new sphere(vec4( 0.5f, -0.5f, 0.2f, 1.0f), 0.2, new emitter(vec4(0.9f, 0.99f, 0.5f, 1.0f))));
    //d_list->add(new sphere(vec4(-1.0f, -1.0f, 0.2f, 1.0f), 0.2, new emitter(vec4(0.65f, 0.0f, 0.91f, 1.0f))));
    //d_list->add(new sphere(vec4( 1.0f,  1.0f, 0.2f, 1.0f), 0.2, new emitter(vec4(0.8f, 0.9f, 0.9f, 1.0f))));
    //d_list->add(new sphere(vec4(-1.0f,  1.0f, 0.2f, 1.0f), 0.2, new emitter(vec4(0.9f, 0.5f, 0.5f, 1.0f))));
    //d_list->add(new sphere(vec4( 1.0f, -1.0f, 0.2f, 1.0f), 0.2, new emitter(vec4(0.5f, 0.59f, 0.9f, 1.0f))));
}

int testCuda()
{
    size_t width = 1920;
    size_t height = 1080;

    unsigned int xThreads = 4;
    unsigned int yThreads = 4;

    dim3 blocks((unsigned int)width / xThreads + 1, height / yThreads + 1);
    dim3 threads(xThreads, yThreads);

    camera* cam = camera::create(ray(vec4(2.0f, 0.0f, 0.5f, 1.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f)), float(width) / float(height));
    hitableList* list = hitableList::create();
    createWorld << <1, 1 >> > (list);

    rayTracingGraphics graphics(width, height, cam);
    graphicsManager manager;
    manager.createInstance(width, height, &graphics);

    auto start = std::chrono::high_resolution_clock::now();

    manager.drawFrame(list);

    std::cout << "render time = " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

    checkCudaErrors(cudaDeviceSynchronize());

    manager.destroy();
    graphics.destroy();
    checkCudaErrors(cudaFree(list));
    checkCudaErrors(cudaFree(cam));

    return 0;
}
