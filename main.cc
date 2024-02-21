#include <iostream>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Metal/Metal.hpp"

// Benchmark parameters.
#define WIDTH 4096
#define HEIGHT 4096
#define GROUP_WIDTH 16
#define GROUP_HEIGHT 16
#define ITERATIONS 5000

// List of kernel functions to benchmark.
const char *kBenchmarks[] = {
    "texture_float",
    "texture_half",
};

using namespace std;

namespace std {
ostream &operator<<(ostream &stream, NS::String *str) {
  stream << str->utf8String();
  return stream;
}
ostream &operator<<(ostream &stream, NS::Error *err) {
  stream << err->localizedDescription();
  return stream;
}
} // namespace std

namespace {
NS::String *String(const char *str) {
  return NS::String::string(str, NS::StringEncoding::UTF8StringEncoding);
}
} // namespace

class Context {
private:
  MTL::Device *device_;
  MTL::CommandQueue *queue_;
  MTL::Buffer *output_buffer_;
  MTL::Texture *texture_;
  MTL::Library *library_;
  MTL::ComputePipelineState *copy_pipeline_;

  const uint16_t kInputValueHalf[4] = {0x5140, 0xC000, 0x6400, 0xE212};
  const float kInputValueFloat[4] = {42, -2, 1024, -777};

public:
  /// Initialize the common Metal objects and resources.
  /// @param device_index the index of the device to use
  /// @returns true on success, false on failure
  bool Init(uint32_t device_index) {
    // Get the Metal device at the requested index.
    auto devices = MTL::CopyAllDevices();
    if (devices->count() == 0) {
      cerr << "No Metal devices found\n";
      return false;
    }
    if (device_index >= devices->count()) {
      cerr << "Device index '" << device_index << "' is out of bounds ("
           << devices->count() << " devices total)\n";
      return false;
    }
    device_ = devices->object<MTL::Device>(device_index);
    if (!device_) {
      cerr << "Failed to create device\n";
      return false;
    }
    cout << "Device: " << device_->name() << "\n";

    // Create a queue.
    queue_ = device_->newCommandQueue();
    if (!queue_) {
      cerr << "Failed to create a queue\n";
      return false;
    }

    // Load and create the shader library.
    NS::Error *error = NS::Error::alloc();
    NS::String *path = String("shaders.metallib");
    library_ = device_->newLibrary(path, &error);
    if (!library_) {
      cerr << "Failed to load the shader library: " << error << "\n";
      return false;
    }

    // Create the copy pipeline used to check the results.
    copy_pipeline_ = MakePipeline("copy");
    if (!copy_pipeline_) {
      return false;
    }

    return InitResources();
  }

  /// Initialize the common Metal resources.
  /// @returns true on success, false on failure
  bool InitResources() {
    // Create the output buffer used to transfer the output for verification.
    output_buffer_ =
        device_->newBuffer(WIDTH * HEIGHT * 16, MTL::ResourceStorageModeShared);
    if (!output_buffer_) {
      cerr << "Failed to create output buffer\n";
      return false;
    }
    memset(output_buffer_->contents(), 0, WIDTH * HEIGHT * 16);

    // Create an f16 texture.
    auto *descriptor = MTL::TextureDescriptor::alloc();
    descriptor->init();
    descriptor->setTextureType(MTL::TextureType2D);
    descriptor->setPixelFormat(MTL::PixelFormatRGBA16Float);
    descriptor->setWidth(WIDTH);
    descriptor->setHeight(HEIGHT);
    descriptor->setUsage(MTL::TextureUsageShaderRead |
                         MTL::TextureUsageShaderWrite);
    texture_ = device_->newTexture(descriptor);
    if (!texture_) {
      cerr << "Failed to create the texture\n";
      return false;
    }

    return true;
  }

  /// Create a pipeline with the given function name.
  /// @param func_name the name of the kernel function
  /// @returns the pipeline, or nullptr on error
  MTL::ComputePipelineState *MakePipeline(const char *func_name) {
    auto *function = library_->newFunction(String(func_name));
    if (!function) {
      cerr << "Failed to create function '" << func_name << "'\n";
      return nullptr;
    }

    NS::Error *error = NS::Error::alloc();
    auto *pipeline = device_->newComputePipelineState(function, &error);
    if (!pipeline) {
      cerr << "Failed to create pipeline for '" << func_name << "': " << error
           << "\n";
      return nullptr;
    }

    return pipeline;
  }

  /// Benchmark a kernel function.
  /// @param func_name the name of the kernel function
  /// @returns true on success, false on failure
  bool RunBenchmark(const char *func_name) {
    cout << "\nRunning '" << func_name << "'\n";

    auto *pipeline = MakePipeline(func_name);
    if (!pipeline) {
      return false;
    }

    auto *command_buffer = queue_->commandBuffer();
    if (!command_buffer) {
      cerr << "Failed to create a command buffer\n";
      return false;
    }

    auto *compute_encoder = command_buffer->computeCommandEncoder();
    if (!compute_encoder) {
      cerr << "Failed to create a compute command encoder\n";
      return false;
    }

    // Dispatch ITERATIONS instances of the kernel function.
    auto grid_size = MTL::Size(WIDTH, HEIGHT, 1);
    auto group_size = MTL::Size(GROUP_WIDTH, GROUP_HEIGHT, 1);
    compute_encoder->setComputePipelineState(pipeline);
    compute_encoder->setBytes(kInputValueHalf, 8, 0);
    compute_encoder->setTexture(texture_, 0);
    for (uint32_t i = 0; i < ITERATIONS; i++) {
      compute_encoder->dispatchThreads(grid_size, group_size);
    }
    compute_encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    // Display the performance.
    double start = command_buffer->GPUStartTime();
    double end = command_buffer->GPUEndTime();
    double duration_total = end - start;
    double duration_per_call = duration_total / ITERATIONS;
    double total_bytes = WIDTH * HEIGHT * 8;
    cout << "  Total runtime = " << 1000 * duration_total << "ms\n";
    cout << "  Runtime per dispatch = " << (1000 * duration_per_call) << "ms\n";
    cout << "  Bandwidth = " << (total_bytes / duration_per_call) / 1e9
         << " GB/s\n";

    return Check();
  }

  /// Check the contents of texture.
  /// @returns true on success, false on verification failure
  bool Check() {
    auto *command_buffer = queue_->commandBuffer();
    if (!command_buffer) {
      cerr << "Failed to create a command buffer\n";
      return false;
    }

    auto *compute_encoder = command_buffer->computeCommandEncoder();
    if (!compute_encoder) {
      cerr << "Failed to create a compute command encoder\n";
      return false;
    }

    // Dispatch the function to copy texture data to a buffer.
    auto grid_size = MTL::Size(WIDTH, HEIGHT, 1);
    auto group_size = MTL::Size(GROUP_WIDTH, GROUP_HEIGHT, 1);
    compute_encoder->setComputePipelineState(copy_pipeline_);
    compute_encoder->setBuffer(output_buffer_, 0, 0);
    compute_encoder->setTexture(texture_, 0);
    compute_encoder->dispatchThreads(grid_size, group_size);
    compute_encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    // Check all entries in the buffer.
    bool correct = true;
    float *result = static_cast<float *>(output_buffer_->contents());
    for (uint32_t y = 0; y < HEIGHT && correct; y++) {
      for (uint32_t x = 0; x < WIDTH && correct; x++) {
        for (uint32_t c = 0; c < 4; c++) {
          uint32_t idx = (x + y * WIDTH) * 4 + c;
          if (result[idx] != kInputValueFloat[c]) {
            printf("output(%u, %u, %u) = %f\n", x, y, c, result[idx]);
            correct = false;
          }
        }
      }
    }
    return correct;
  }
};

int main(int argc, const char *argv[]) {
  Context context;
  if (!context.Init(0)) {
    return 1;
  }

  for (auto *name : kBenchmarks) {
    if (!context.RunBenchmark(name)) {
      return 1;
    }
  }

  return 0;
}
