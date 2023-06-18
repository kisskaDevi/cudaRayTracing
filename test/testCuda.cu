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

void createWorld(vertex* vertexBuffer, hitableList* list) {

    addInList( list,
        sphere::create( vec4( 0.0f,  0.0f,  0.5f,  1.0f), 0.50f, vec4(0.8f, 0.3f, 0.3f, 1.0f), lambertian::create(pi)),
        sphere::create( vec4( 0.0f,  1.0f,  0.5f,  1.0f), 0.50f, vec4(0.8f, 0.8f, 0.8f, 1.0f), metal::create(3.0f, 0.005 * pi)),
        sphere::create( vec4( 0.0f, -1.0f,  0.5f,  1.0f), 0.50f, vec4(0.9f, 0.9f, 0.9f, 1.0f), glass::create(1.5f, 0.96f)),
        sphere::create( vec4( 0.0f, -1.0f,  0.5f,  1.0f), 0.45f, vec4(0.9f, 0.9f, 0.9f, 1.0f), glass::create(1.0f / 1.5f, 0.96f)),
        sphere::create( vec4(-1.5f,  0.0f,  0.5f,  1.0f), 0.50f, vec4(1.0f, 0.9f, 0.7f, 1.0f), emitter::create())
    );

    addInList( list,
        /*down*/  triangle::create(  0,  1,  2, vertexBuffer, lambertian::create(pi)),          triangle::create(  3,  1,  2, vertexBuffer, lambertian::create(pi)),
        /*top*/   triangle::create(  4,  5,  6, vertexBuffer, metal::create(3.0f, pi)),         triangle::create(  7,  5,  6, vertexBuffer, metal::create(3.0f, pi)),
        /*back*/  triangle::create(  8,  9, 10, vertexBuffer, metal::create(3.0f, pi)),         triangle::create( 11, 10,  8, vertexBuffer, metal::create(3.0f, pi)),
        /*front*/ triangle::create( 12, 13, 14, vertexBuffer, metal::create(3.0f, 0.3f * pi)),  triangle::create( 15, 14, 12, vertexBuffer, metal::create(3.0f, 0.3f * pi)),
        /*left*/  triangle::create( 16, 17, 18, vertexBuffer, lambertian::create(pi)),          triangle::create( 16, 19, 18, vertexBuffer, lambertian::create(pi)),
        /*right*/ triangle::create( 20, 21, 22, vertexBuffer, lambertian::create(0.3f * pi)),   triangle::create( 20, 23, 22, vertexBuffer, lambertian::create(0.3f * pi))
    );

    addInList(list,
        sphere::create(vec4( 1.5f, -1.5f,  0.2f,  1.0f), 0.2, vec4(0.99f, 0.80f, 0.20f, 1.00f), emitter::create()),
        sphere::create(vec4( 1.5f,  1.5f,  0.2f,  1.0f), 0.2, vec4(0.20f, 0.80f, 0.99f, 1.00f), emitter::create()),
        sphere::create(vec4(-1.5f, -1.5f,  0.2f,  1.0f), 0.2, vec4(0.99f, 0.40f, 0.85f, 1.00f), emitter::create()),
        sphere::create(vec4(-1.5f,  1.5f,  0.2f,  1.0f), 0.2, vec4(0.40f, 0.99f, 0.50f, 1.00f), emitter::create()),
        sphere::create(vec4(-0.5f, -0.5f,  0.2f,  1.0f), 0.2, vec4(0.65f, 0.00f, 0.91f, 1.00f), emitter::create()),
        sphere::create(vec4( 0.5f,  0.5f,  0.2f,  1.0f), 0.2, vec4(0.80f, 0.70f, 0.99f, 1.00f), emitter::create()),
        sphere::create(vec4(-0.5f,  0.5f,  0.2f,  1.0f), 0.2, vec4(0.59f, 0.50f, 0.90f, 1.00f), emitter::create()),
        sphere::create(vec4( 0.5f, -0.5f,  0.2f,  1.0f), 0.2, vec4(0.90f, 0.99f, 0.50f, 1.00f), emitter::create()),
        sphere::create(vec4(-1.0f, -1.0f,  0.2f,  1.0f), 0.2, vec4(0.65f, 0.00f, 0.91f, 1.00f), emitter::create()),
        sphere::create(vec4( 1.0f,  1.0f,  0.2f,  1.0f), 0.2, vec4(0.80f, 0.90f, 0.90f, 1.00f), emitter::create()),
        sphere::create(vec4(-1.0f,  1.0f,  0.2f,  1.0f), 0.2, vec4(0.90f, 0.50f, 0.50f, 1.00f), emitter::create()),
        sphere::create(vec4( 1.0f, -1.0f,  0.2f,  1.0f), 0.2, vec4(0.50f, 0.59f, 0.90f, 1.00f), emitter::create())
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

    hitableList* list = hitableList::create();
    buffer<vertex> vertexBuffer = buffer<vertex>(24, createBox().data());


    createWorld(vertexBuffer.get(), list);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    camera* cam = camera::create(ray(vec4(2.0f, -2.0f, 0.5f, 1.0f), vec4(-1.0f, 1.0f, 0.0f, 0.0f)), float(width) / float(height));
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
