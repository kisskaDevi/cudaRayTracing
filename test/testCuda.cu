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
                        vec4(0.8, 0.3, 0.3, 1.0f),
                        new lambertian(pi)),
        new sphere(     vec4(0.0f, 1.0f, 0.5f, 1.0f),
                        0.5,
                        vec4(0.8f, 0.8f, 0.8f, 1.0f),
                        new metal(3.0f, 0.005*pi)),
        new sphere(     vec4(0.0f, -1.0f, 0.5f, 1.0f),
                        0.5,
                        vec4(0.9f, 0.9f, 0.9f, 1.0f),
                        new glass(1.5f, 0.96f))
    );

    list->add(
        //down
        new triangle(   vertex(vec4(-3.0f, -3.0f,  0.0f,  1.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.6f, 0.6f, 0.8f, 1.0f)),
                        vertex(vec4(-3.0f,  3.0f,  0.0f,  1.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.6f, 0.6f, 0.8f, 1.0f)),
                        vertex(vec4( 3.0f, -3.0f,  0.0f,  1.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.6f, 0.6f, 0.8f, 1.0f)),
                        new lambertian(pi)),
        new triangle(   vertex(vec4( 3.0f,  3.0f,  0.0f,  1.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.6f, 0.6f, 0.8f, 1.0f)),
                        vertex(vec4(-3.0f,  3.0f,  0.0f,  1.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.6f, 0.6f, 0.8f, 1.0f)),
                        vertex(vec4( 3.0f, -3.0f,  0.0f,  1.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.6f, 0.6f, 0.8f, 1.0f)),
                        new lambertian(pi)),
        //top
        new triangle(   vertex(vec4(-3.0f, -3.0f,  3.0f,  1.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.9f, 0.9f, 0.9f, 1.0f)),
                        vertex(vec4(-3.0f,  3.0f,  3.0f,  1.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.9f, 0.9f, 0.9f, 1.0f)),
                        vertex(vec4( 3.0f, -3.0f,  3.0f,  1.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.9f, 0.9f, 0.9f, 1.0f)),
                        new emitter()),
        new triangle(   vertex(vec4( 3.0f,  3.0f,  3.0f,  1.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.9f, 0.9f, 0.9f, 1.0f)),
                        vertex(vec4(-3.0f,  3.0f,  3.0f,  1.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.9f, 0.9f, 0.9f, 1.0f)),
                        vertex(vec4( 3.0f, -3.0f,  3.0f,  1.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.9f, 0.9f, 0.9f, 1.0f)),
                        new emitter()),
        //back
        new triangle(   vertex(vec4(-3.0f, -3.0f,  0.0f,  1.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.2f, 0.2f, 1.0f)),
                        vertex(vec4(-3.0f,  3.0f,  0.0f,  1.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(0.2f, 0.8f, 0.2f, 1.0f)),
                        vertex(vec4(-3.0f,  3.0f,  3.0f,  1.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(0.2f, 0.2f, 0.8f, 1.0f)),
                        new metal(3.0f, pi)),
        new triangle(   vertex(vec4(-3.0f, -3.0f,  3.0f,  1.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.2f, 0.8f, 1.0f)),
                        vertex(vec4(-3.0f,  3.0f,  3.0f,  1.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(0.2f, 0.2f, 0.8f, 1.0f)),
                        vertex(vec4(-3.0f, -3.0f,  0.0f,  1.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.2f, 0.2f, 1.0f)),
                        new metal(3.0f, pi)),
        //front
        new triangle(   vertex(vec4( 3.0f, -3.0f,  0.0f,  1.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.8f, 0.8f, 1.0f)),
                        vertex(vec4( 3.0f,  3.0f,  0.0f,  1.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.8f, 0.8f, 1.0f)),
                        vertex(vec4( 3.0f,  3.0f,  3.0f,  1.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.8f, 0.8f, 1.0f)),
                        new metal(3.0f, 0.3f * pi)),
        new triangle(   vertex(vec4( 3.0f, -3.0f,  3.0f,  1.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.8f, 0.8f, 1.0f)),
                        vertex(vec4( 3.0f,  3.0f,  3.0f,  1.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.8f, 0.8f, 1.0f)),
                        vertex(vec4( 3.0f, -3.0f,  0.0f,  1.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.8f, 0.8f, 1.0f)),
                        new metal(3.0f, 0.3f * pi)),
        //left
        new triangle(   vertex(vec4(-3.0f, -3.0f,  0.0f,  1.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.8f, 0.5f, 0.0f, 1.0f)),
                        vertex(vec4( 3.0f, -3.0f,  0.0f,  1.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.8f, 0.5f, 0.0f, 1.0f)),
                        vertex(vec4( 3.0f, -3.0f,  3.0f,  1.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.8f, 0.5f, 0.0f, 1.0f)),
                        new lambertian(pi)),
        new triangle(   vertex(vec4(-3.0f, -3.0f,  0.0f,  1.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.8f, 0.5f, 0.0f, 1.0f)),
                        vertex(vec4(-3.0f, -3.0f,  3.0f,  1.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.8f, 0.5f, 0.0f, 1.0f)),
                        vertex(vec4( 3.0f, -3.0f,  3.0f,  1.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.8f, 0.5f, 0.0f, 1.0f)),
                        new lambertian(pi)),
        //right
        new triangle(   vertex(vec4(-3.0f,  3.0f,  0.0f,  1.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, 0.4f, 0.8f, 1.0f)),
                        vertex(vec4( 3.0f,  3.0f,  0.0f,  1.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, 0.4f, 0.8f, 1.0f)),
                        vertex(vec4( 3.0f,  3.0f,  3.0f,  1.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, 0.4f, 0.8f, 1.0f)),
                        new lambertian(0.3f * pi)),
        new triangle(   vertex(vec4(-3.0f,  3.0f,  0.0f,  1.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, 0.4f, 0.8f, 1.0f)),
                        vertex(vec4(-3.0f,  3.0f,  3.0f,  1.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, 0.4f, 0.8f, 1.0f)),
                        vertex(vec4( 3.0f,  3.0f,  3.0f,  1.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, 0.4f, 0.8f, 1.0f)),
                        new lambertian(0.3f * pi))
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
    size_t width = 1920, height = 1080;

    camera* cam = camera::create(ray(vec4(2.0f, 0.0f, 0.5f, 1.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f)), float(width) / float(height));
    hitableList* list = hitableList::create();

    createWorld<<<1, 1>>>(list);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    rayTracingGraphics graphics(width, height, cam);
    graphicsManager manager(&graphics);

    auto start = std::chrono::high_resolution_clock::now();

    size_t steps = 10;
    for (size_t i = 0; i < steps; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        manager.drawFrame(list);

        Image::outPPM(graphics.getSwapChain(), width, height, "image" + std::to_string(i + 1) + ".ppm");
        std::cout << (i + 1) * 100 / steps << " %\t"
        << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;
    }
    std::cout << "render time = " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

    checkCudaErrors(cudaDeviceSynchronize());
    manager.destroy();
    graphics.destroy();
    checkCudaErrors(cudaFree(list));
    checkCudaErrors(cudaFree(cam));

    return 0;
}
