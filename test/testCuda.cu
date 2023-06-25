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
#include "baseMaterial.h"
#include "emitter.h"

#include "rayTracingGraphics.h"
#include "graphicsManager.h"

struct primitive {
    buffer<vertex> vertexBuffer;
    std::vector<uint32_t> indexBuffer;
    size_t firstIndex{ 0 };
    size_t size{ 0 };
    material* mat{ nullptr };

    primitive(
        std::vector<vertex>& vertexBuffer,
        std::vector<uint32_t>& indexBuffer,
        size_t firstIndex,
        material* mat
    ) : vertexBuffer(std::move(buffer<vertex>(vertexBuffer.size(), vertexBuffer.data()))), indexBuffer(std::move(indexBuffer)), firstIndex(firstIndex), size(this->indexBuffer.size()), mat(mat) {}

    void moveToList(hitableList* list) {
        for (size_t index = firstIndex; index < size; index += 3) {
            addInList( list, triangle::create( indexBuffer[index + 0], indexBuffer[index + 1], indexBuffer[index + 2], vertexBuffer.get(), mat));
        }
    }
};

enum sign
{
    minus,
    plus
};

std::vector<vertex> createBoxVertexBuffer(vec4 scale, vec4 translate, sign normalSign, properties props, std::vector<vec4> colors) {
    float plus = normalSign == sign::plus ? 1.0f : -1.0f, minus = -plus;
    vec4 v[8] =
    {
        scale * vec4(-1.0f, -1.0f, -1.0f, 1.0f) + translate,
        scale * vec4(-1.0f,  1.0f, -1.0f, 1.0f) + translate,
        scale * vec4(1.0f, -1.0f, -1.0f, 1.0f) + translate,
        scale * vec4(1.0f,  1.0f, -1.0f, 1.0f) + translate,
        scale * vec4(-1.0f, -1.0f,  1.0f, 1.0f) + translate,
        scale * vec4(-1.0f,  1.0f,  1.0f, 1.0f) + translate,
        scale * vec4(1.0f, -1.0f,  1.0f, 1.0f) + translate,
        scale * vec4(1.0f,  1.0f,  1.0f, 1.0f) + translate
    };
    vec4 n[6] =
    {
        vec4(0.0f, 0.0f, minus, 0.0f), vec4(0.0f, 0.0f, plus, 0.0f), vec4(minus, 0.0f, 0.0f, 0.0f),
        vec4(plus, 0.0f, 0.0f, 0.0f), vec4(0.0f, minus, 0.0f, 0.0f), vec4(0.0f, plus, 0.0f, 0.0f)
    };
    size_t indices[6][4] = { {0,1,2,3}, {4,5,6,7}, {0,1,4,5}, {2,3,6,7}, {0,2,4,6}, {1,3,5,7} };

    std::vector<vertex> vertexBuffer;
    for (size_t i = 0; i < 6; i++) {
        for (size_t j = 0; j < 4; j++) {
            vertexBuffer.push_back(vertex(v[indices[i][j]], n[i], colors[i], props));
        }
    }
    return vertexBuffer;
}

std::vector<uint32_t> createBoxIndexBuffer() {
    return std::vector<uint32_t>{
        0, 1, 2, 3, 1, 2,
        4, 5, 6, 7, 5, 6,
        8, 9, 11, 10, 11, 8,
        12, 13, 15, 14, 15, 12,
        16, 17, 19, 16, 18, 19,
        20, 21, 23, 20, 22, 23
    };
}

void createWorld(std::vector<primitive>& primitives, hitableList* list, baseMaterial* baseMat, emitter* emit) {

    addInList( list,
        sphere::create( vec4( 0.0f,  0.0f,  0.5f,  1.0f), 0.50f, vec4(0.8f, 0.3f, 0.3f, 1.0f), { 1.0f, 0.0f, 0.0f, pi }, baseMat),
        sphere::create( vec4( 0.0f,  1.0f,  0.5f,  1.0f), 0.50f, vec4(0.8f, 0.8f, 0.8f, 1.0f), { 1.0f, 0.0f, 3.0f, 0.05f * pi }, baseMat),
        sphere::create( vec4( 0.0f, -1.0f,  0.5f,  1.0f), 0.50f, vec4(0.9f, 0.9f, 0.9f, 1.0f), { 1.5f, 0.96f, 0.001f, 0.0f }, baseMat),
        sphere::create( vec4( 0.0f, -1.0f,  0.5f,  1.0f), 0.45f, vec4(0.9f, 0.9f, 0.9f, 1.0f), { 1.0f / 1.5f, 0.96f, 0.001f, 0.0f }, baseMat),
        sphere::create( vec4(-1.5f,  0.0f,  0.5f,  1.0f), 0.50f, vec4(1.00f, 0.90f, 0.70f, 1.00f), {}, emit),
        sphere::create( vec4( 1.5f, -1.5f,  0.2f,  1.0f), 0.20f, vec4(0.99f, 0.80f, 0.20f, 1.00f), {}, emit),
        sphere::create( vec4( 1.5f,  1.5f,  0.2f,  1.0f), 0.20f, vec4(0.20f, 0.80f, 0.99f, 1.00f), {}, emit),
        sphere::create( vec4(-1.5f, -1.5f,  0.2f,  1.0f), 0.20f, vec4(0.99f, 0.40f, 0.85f, 1.00f), {}, emit),
        sphere::create( vec4(-1.5f,  1.5f,  0.2f,  1.0f), 0.20f, vec4(0.40f, 0.99f, 0.50f, 1.00f), {}, emit),
        sphere::create( vec4(-0.5f, -0.5f,  0.2f,  1.0f), 0.20f, vec4(0.65f, 0.00f, 0.91f, 1.00f), {}, emit),
        sphere::create( vec4( 0.5f,  0.5f,  0.2f,  1.0f), 0.20f, vec4(0.80f, 0.70f, 0.99f, 1.00f), {}, emit),
        sphere::create( vec4(-0.5f,  0.5f,  0.2f,  1.0f), 0.20f, vec4(0.59f, 0.50f, 0.90f, 1.00f), {}, emit),
        sphere::create( vec4( 0.5f, -0.5f,  0.2f,  1.0f), 0.20f, vec4(0.90f, 0.99f, 0.50f, 1.00f), {}, emit),
        sphere::create( vec4(-1.0f, -1.0f,  0.2f,  1.0f), 0.20f, vec4(0.65f, 0.00f, 0.91f, 1.00f), {}, emit),
        sphere::create( vec4( 1.0f,  1.0f,  0.2f,  1.0f), 0.20f, vec4(0.80f, 0.90f, 0.90f, 1.00f), {}, emit),
        sphere::create( vec4(-1.0f,  1.0f,  0.2f,  1.0f), 0.20f, vec4(0.90f, 0.50f, 0.50f, 1.00f), {}, emit),
        sphere::create( vec4( 1.0f, -1.0f,  0.2f,  1.0f), 0.20f, vec4(0.50f, 0.59f, 0.90f, 1.00f), {}, emit)
    );

    primitives.emplace_back(
        createBoxVertexBuffer(vec4(3.0f, 3.0f, 1.5f, 1.0f), vec4(0.0f, 0.0f, 1.5f, 0.0f), sign::minus, { 1.0f, 0.0f, 0.0f, pi },
            { vec4(0.5f, 0.5f, 0.5f, 1.0f), vec4(0.5f, 0.5f, 0.5f, 1.0f), vec4(0.8f, 0.4f, 0.8f, 1.0f), vec4(0.4f, 0.4f, 0.4f, 1.0f), vec4(0.9f, 0.5f, 0.0f, 1.0f), vec4(0.1f, 0.4f, 0.9f, 1.0f) }),
        createBoxIndexBuffer(),
        0,
        baseMat
    );
    primitives.back().moveToList(list);

    primitives.emplace_back(
        createBoxVertexBuffer(vec4(0.4f, 0.4f, 0.4f, 1.0f), vec4(1.5f, 0.0f, 0.4f, 0.0f), sign::plus, { 2.0f, 0.96f, 0.01f, 0.01f * pi },
            std::vector<vec4>(6, vec4(1.0f))),
        createBoxIndexBuffer(),
        0,
        baseMat
    );
    primitives.back().moveToList(list);

    primitives.emplace_back(
        createBoxVertexBuffer(vec4(0.3f, 0.3f, 0.3f, 1.0f), vec4(1.5f, 0.0f, 0.4f, 0.0f), sign::plus, { 0.5f, 0.96f, 0.01f, 0.01f * pi },
            std::vector<vec4>(6, vec4(1.0f))),
        createBoxIndexBuffer(),
        0,
        baseMat
    );
    primitives.back().moveToList(list);

    for (int i = 0; i < 0; i++) {
        float phi = 2.0f * pi * float(i) / 50.0f;
        primitives.emplace_back(
            createBoxVertexBuffer(vec4(0.1f, 0.1f, 0.1f, 1.0f), vec4(2.8 * std::cos(phi), 2.8 * std::sin(phi), 0.1f, 0.0f), sign::plus, { std::cos(phi), 0.96f, std::sin(phi), std::abs(std::sin(phi) * std::cos(phi)) * pi },
                std::vector<vec4>(6, vec4(std::abs(std::cos(phi)), std::abs(std::sin(phi)), std::abs(std::sin(phi) * std::cos(phi)), 1.0f))),
            createBoxIndexBuffer(),
            0,
            baseMat
        );
        primitives.back().moveToList(list);
    }
}

int testCuda()
{
    size_t width = 1920, height = 1080;

    camera* cam = camera::create(ray(vec4(2.0f, 2.0f, 2.0f, 1.0f), vec4(-1.0f, -1.0f, -1.0f, 0.0f)), float(width) / float(height));
    hitableList* list = hitableList::create();
    baseMaterial* baseMat = baseMaterial::create();
    emitter* emit = emitter::create();

    std::vector<primitive> primitives;

    createWorld(primitives, list, baseMat, emit);
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
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "render time = " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

    manager.destroy();
    graphics.destroy();
    hitableList::destroy(list);
    camera::destroy(cam);
    baseMaterial::destroy(baseMat);
    emitter::destroy(emit);

    return 0;
}
