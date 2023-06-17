#include <iostream>
#include <math.h>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
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

__global__ void createWorld(vertex* vertexBuffer, hitableList* list) {
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
                        new glass(1.5f, 0.96f)),
        new sphere(     vec4(0.0f, -1.0f, 0.5f, 1.0f),
                        0.45,
                        vec4(0.9f, 0.9f, 0.9f, 1.0f),
                        new glass(1.0f / 1.5f, 0.96f))
    );

    list->add(
        /*down*/ new triangle( 0, 1, 2, vertexBuffer, new lambertian(pi)), new triangle(3, 1, 2, vertexBuffer, new lambertian(pi)),
        /*top*/ new triangle( 4, 5, 6, vertexBuffer, new metal(3.0f, pi)), new triangle( 7, 5, 6, vertexBuffer, new metal(3.0f, pi)),
        /*back*/ new triangle( 8, 9, 10, vertexBuffer, new metal(3.0f, pi)), new triangle( 11, 10, 8, vertexBuffer, new metal(3.0f, pi)),
        /*front*/ new triangle( 12, 13, 14, vertexBuffer, new metal(3.0f, 0.3f * pi)), new triangle( 15, 14, 12, vertexBuffer, new metal(3.0f, 0.3f * pi)),
        /*left*/ new triangle( 16, 17, 18, vertexBuffer, new lambertian(pi)), new triangle( 16, 19, 18, vertexBuffer, new lambertian(pi)),
        /*right*/ new triangle( 20, 21, 22, vertexBuffer, new lambertian(0.3f * pi)), new triangle( 20, 23, 22, vertexBuffer, new lambertian(0.3f * pi))
    );

    list->add(
        new sphere(vec4( 1.5f, -1.5f,  0.2f,  1.0f), 0.2, vec4(0.99f, 0.80f, 0.20f, 1.00f), new emitter()),
        new sphere(vec4( 1.5f,  1.5f,  0.2f,  1.0f), 0.2, vec4(0.20f, 0.80f, 0.99f, 1.00f), new emitter()),
        new sphere(vec4(-1.5f, -1.5f,  0.2f,  1.0f), 0.2, vec4(0.99f, 0.40f, 0.85f, 1.00f), new emitter()),
        new sphere(vec4(-1.5f,  1.5f,  0.2f,  1.0f), 0.2, vec4(0.40f, 0.99f, 0.50f, 1.00f), new emitter()),
        new sphere(vec4(-0.5f, -0.5f,  0.2f,  1.0f), 0.2, vec4(0.65f, 0.00f, 0.91f, 1.00f), new emitter()),
        new sphere(vec4( 0.5f,  0.5f,  0.2f,  1.0f), 0.2, vec4(0.80f, 0.70f, 0.99f, 1.00f), new emitter()),
        new sphere(vec4(-0.5f,  0.5f,  0.2f,  1.0f), 0.2, vec4(0.59f, 0.50f, 0.90f, 1.00f), new emitter()),
        new sphere(vec4( 0.5f, -0.5f,  0.2f,  1.0f), 0.2, vec4(0.90f, 0.99f, 0.50f, 1.00f), new emitter()),
        new sphere(vec4(-1.0f, -1.0f,  0.2f,  1.0f), 0.2, vec4(0.65f, 0.00f, 0.91f, 1.00f), new emitter()),
        new sphere(vec4( 1.0f,  1.0f,  0.2f,  1.0f), 0.2, vec4(0.80f, 0.90f, 0.90f, 1.00f), new emitter()),
        new sphere(vec4(-1.0f,  1.0f,  0.2f,  1.0f), 0.2, vec4(0.90f, 0.50f, 0.50f, 1.00f), new emitter()),
        new sphere(vec4( 1.0f, -1.0f,  0.2f,  1.0f), 0.2, vec4(0.50f, 0.59f, 0.90f, 1.00f), new emitter())
    );
}

std::vector<vertex> createBox() {
    std::vector<vertex> vertexBuffer(24);

    vertexBuffer[0] = vertex(vec4(-3.0f, -3.0f, 0.0f, 1.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.5f, 0.5f, 0.5f, 1.0f));
    vertexBuffer[1] = vertex(vec4(-3.0f, 3.0f, 0.0f, 1.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.5f, 0.5f, 0.5f, 1.0f));
    vertexBuffer[2] = vertex(vec4(3.0f, -3.0f, 0.0f, 1.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.5f, 0.5f, 0.5f, 1.0f));
    vertexBuffer[3] = vertex(vec4(3.0f, 3.0f, 0.0f, 1.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.5f, 0.5f, 0.5f, 1.0f));

    vertexBuffer[4] = vertex(vec4(-3.0f, -3.0f, 3.0f, 1.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.9f, 0.9f, 0.9f, 1.0f));
    vertexBuffer[5] = vertex(vec4(-3.0f, 3.0f, 3.0f, 1.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.9f, 0.9f, 0.9f, 1.0f));
    vertexBuffer[6] = vertex(vec4(3.0f, -3.0f, 3.0f, 1.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.9f, 0.9f, 0.9f, 1.0f));
    vertexBuffer[7] = vertex(vec4(3.0f, 3.0f, 3.0f, 1.0f), vec4(0.0f, 0.0f, -1.0f, 0.0f), vec4(0.9f, 0.9f, 0.9f, 1.0f));

    vertexBuffer[8] = vertex(vec4(-3.0f, -3.0f, 0.0f, 1.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.2f, 0.2f, 1.0f));
    vertexBuffer[9] = vertex(vec4(-3.0f, 3.0f, 0.0f, 1.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(0.2f, 0.8f, 0.2f, 1.0f));
    vertexBuffer[10] = vertex(vec4(-3.0f, 3.0f, 3.0f, 1.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(0.2f, 0.2f, 0.8f, 1.0f));
    vertexBuffer[11] = vertex(vec4(-3.0f, -3.0f, 3.0f, 1.0f), vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.2f, 0.8f, 1.0f));

    vertexBuffer[12] = vertex(vec4(3.0f, -3.0f, 0.0f, 1.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.8f, 0.8f, 1.0f));
    vertexBuffer[13] = vertex(vec4(3.0f, 3.0f, 0.0f, 1.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.8f, 0.8f, 1.0f));
    vertexBuffer[14] = vertex(vec4(3.0f, 3.0f, 3.0f, 1.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.8f, 0.8f, 1.0f));
    vertexBuffer[15] = vertex(vec4(3.0f, -3.0f, 3.0f, 1.0f), vec4(-1.0f, 0.0f, 0.0f, 0.0f), vec4(0.8f, 0.8f, 0.8f, 1.0f));

    vertexBuffer[16] = vertex(vec4(-3.0f, -3.0f, 0.0f, 1.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.8f, 0.5f, 0.0f, 1.0f));
    vertexBuffer[17] = vertex(vec4(3.0f, -3.0f, 0.0f, 1.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.8f, 0.5f, 0.0f, 1.0f));
    vertexBuffer[18] = vertex(vec4(3.0f, -3.0f, 3.0f, 1.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.8f, 0.5f, 0.0f, 1.0f));
    vertexBuffer[19] = vertex(vec4(-3.0f, -3.0f, 3.0f, 1.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.8f, 0.5f, 0.0f, 1.0f));

    vertexBuffer[20] = vertex(vec4(-3.0f, 3.0f, 0.0f, 1.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, 0.4f, 0.8f, 1.0f));
    vertexBuffer[21] = vertex(vec4(3.0f, 3.0f, 0.0f, 1.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, 0.4f, 0.8f, 1.0f));
    vertexBuffer[22] = vertex(vec4(3.0f, 3.0f, 3.0f, 1.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, 0.4f, 0.8f, 1.0f));
    vertexBuffer[23] = vertex(vec4(-3.0f, 3.0f, 3.0f, 1.0f), vec4(0.0f, -1.0f, 0.0f, 0.0f), vec4(0.0f, 0.4f, 0.8f, 1.0f));

    return vertexBuffer;
}

int testCuda()
{
    size_t width = 1920, height = 1080;

    buffer<vertex> vertexBuffer = buffer<vertex>(24, createBox().data());

    camera* cam = camera::create(ray(vec4(2.0f, -2.0f, 0.5f, 1.0f), vec4(-1.0f, 1.0f, 0.0f, 0.0f)), float(width) / float(height));
    hitableList* list = hitableList::create();

    createWorld<<<1, 1>>>(vertexBuffer.get(), list);
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
    vertexBuffer.destroy();
    checkCudaErrors(cudaFree(list));
    checkCudaErrors(cudaFree(cam));

    return 0;
}
