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
        checkCudaErrors(cudaMallocManaged((void**)&buffer, size));
        return buffer;
    }
}

namespace Image {

    void Image::clear(vec4* image, size_t size) {
        for (size_t i = 0; i < size; i++) {
            image[i] = vec4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    void Image::combine(vec4* dst, vec4* src, size_t size) {
        for (size_t i = 0; i < size; i++) {
            float a = src[i].a() == 0.0f ? 1.0f : src[i].a();
            dst[i] = vec4(dst[i].r() + src[i].r() / a, dst[i].g() + src[i].g() / a, dst[i].b() + src[i].b() / a, 1.0f);
        }
    }

    void Image::normalize(vec4* frameBuffer, size_t size) {
        for (size_t i = 0; i < size; i++) {
            float max = std::max(frameBuffer[i].r(), std::max(frameBuffer[i].g(), frameBuffer[i].b()));
            if (max > 1.0f) {
                frameBuffer[i] /= max;
            }
        }
    }

    void Image::outPPM(vec4* frameBuffer, size_t width, size_t height, const std::string& filename) {
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

    void Image::outPGM(vec4* frameBuffer, size_t width, size_t height, const std::string& filename) {
        std::ofstream image(filename);
        image << "P2\n" << width << " " << height << "\n255\n";
        for (int j = 0; j < height; j++) {
            for (int i = width - 1; i >= 0; i--) {
                size_t pixel_index = j * width + i;
                image << static_cast<uint32_t>(255.99f * (frameBuffer[pixel_index].r() + frameBuffer[pixel_index].g() + frameBuffer[pixel_index].b()) / 3) << "\n";
            }
        }
        image.close();
    }
}