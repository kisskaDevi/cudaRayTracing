#include "operations.h"

#include <fstream>
#include <iostream>

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

namespace Buffer {
    vec4* Buffer::create(size_t size) {
        vec4* buffer;
        checkCudaErrors(cudaMalloc((void**)&buffer, size));
        return buffer;
    }
}

namespace Image {

    __global__ void Image::clear(vec4* image, size_t width, size_t height) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        if ((i < width) && (j < height)) {
            int pixel_index = j * width + i;
            image[pixel_index] = vec4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    __global__ void Image::combine(vec4* dst, vec4* src, size_t width, size_t height) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        if ((i < width) && (j < height)) {
            int pixel_index = j * width + i;
            float a = src[pixel_index].a() == 0.0f ? 1.0f : src[pixel_index].a();
            dst[pixel_index] = vec4(dst[pixel_index].r() + src[pixel_index].r() / a,
                                    dst[pixel_index].g() + src[pixel_index].g() / a,
                                    dst[pixel_index].b() + src[pixel_index].b() / a, 1.0f);
        }
    }

    __global__ void Image::normalize(vec4* frameBuffer, size_t width, size_t height) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        auto max = [](const float& a, const float& b) {
            return a >= b ? a : b;
        };

        if ((i < width) && (j < height)) {
            int pixel_index = j * width + i;
            float maximum = max(frameBuffer[pixel_index].r(), max(frameBuffer[pixel_index].g(), frameBuffer[pixel_index].b()));
            if (maximum > 1.0f) {
                frameBuffer[pixel_index] /= maximum;
            }
        }
    }

    void Image::outPPM(vec4* frameBuffer, size_t width, size_t height, const std::string& filename) {
        vec4* hostFrameBuffer = new vec4[width * height];
        cudaMemcpy(hostFrameBuffer, frameBuffer, width * height * sizeof(vec4), cudaMemcpyDeviceToHost);

        std::ofstream image(filename);
        image << "P3\n" << width << " " << height << "\n255\n";
        for (int j = 0; j < height; j++) {
            for (int i = width - 1; i >= 0; i--) {
                size_t pixel_index = j * width + i;
                image   << static_cast<uint32_t>(255.99f * hostFrameBuffer[pixel_index].r()) << " "
                        << static_cast<uint32_t>(255.99f * hostFrameBuffer[pixel_index].g()) << " "
                        << static_cast<uint32_t>(255.99f * hostFrameBuffer[pixel_index].b()) << "\n";
            }
        }
        image.close();
        delete[] hostFrameBuffer;
    }

    void Image::outPGM(vec4* frameBuffer, size_t width, size_t height, const std::string& filename) {
        vec4* hostFrameBuffer = new vec4[width * height];
        cudaMemcpy(hostFrameBuffer, frameBuffer, width * height * sizeof(vec4), cudaMemcpyDeviceToHost);

        std::ofstream image(filename);
        image << "P2\n" << width << " " << height << "\n255\n";
        for (int j = 0; j < height; j++) {
            for (int i = width - 1; i >= 0; i--) {
                size_t pixel_index = j * width + i;
                image << static_cast<uint32_t>(255.99f * (hostFrameBuffer[pixel_index].r() + hostFrameBuffer[pixel_index].g() + hostFrameBuffer[pixel_index].b()) / 3) << "\n";
            }
        }
        image.close();
        delete[] hostFrameBuffer;
    }
}