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

// A benchmark config.
struct Benchmark {
  enum ResourceType {
    kBytes,
    kBuffer,
    kTexture,
  };

  const char *name = nullptr;
  const char *func = nullptr;
  ResourceType input_type;
  ResourceType output_type;
};

// List of benchmarks.
Benchmark kBenchmarks[] = {
    {.name = "Write to buffer",
     .func = "write_buffer",
     .input_type = Benchmark::kBytes,
     .output_type = Benchmark::kBuffer},
    {.name = "Write to texture<float>",
     .func = "write_texture_float",
     .input_type = Benchmark::kBytes,
     .output_type = Benchmark::kTexture},
    {.name = "Write to texture<half>",
     .func = "write_texture_half",
     .input_type = Benchmark::kBytes,
     .output_type = Benchmark::kTexture},

    {.name = "Read from buffer (copy to buffer)",
     .func = "read_buffer",
     .input_type = Benchmark::kBuffer,
     .output_type = Benchmark::kBuffer},
    {.name = "Read from texture<float> (copy to buffer)",
     .func = "read_texture_float",
     .input_type = Benchmark::kTexture,
     .output_type = Benchmark::kBuffer},
    {.name = "Read from texture<half> (copy to buffer)",
     .func = "read_texture_half",
     .input_type = Benchmark::kTexture,
     .output_type = Benchmark::kBuffer},
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
  MTL::Buffer *staging_buffer_;
  MTL::Buffer *input_buffer_;
  MTL::Buffer *output_buffer_;
  MTL::Texture *input_texture_;
  MTL::Texture *output_texture_;
  MTL::Library *library_;
  MTL::ComputePipelineState *copy_from_buffer_;
  MTL::ComputePipelineState *copy_from_texture_;

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

    // Create the copy pipelines used to check the results.
    copy_from_buffer_ = MakePipeline("copy_from_buffer");
    if (!copy_from_buffer_) {
      return false;
    }
    copy_from_texture_ = MakePipeline("copy_from_texture");
    if (!copy_from_texture_) {
      return false;
    }

    return InitResources();
  }

  /// Initialize the common Metal resources.
  /// @returns true on success, false on failure
  bool InitResources() {
    // Create the output buffer used to transfer the output for verification.
    staging_buffer_ =
        device_->newBuffer(WIDTH * HEIGHT * 16, MTL::ResourceStorageModeShared);
    if (!staging_buffer_) {
      cerr << "Failed to create output buffer\n";
      return false;
    }
    memset(staging_buffer_->contents(), 0, WIDTH * HEIGHT * 16);

    // Create the input buffer used for benchmarks that read from buffers.
    input_buffer_ =
        device_->newBuffer(WIDTH * HEIGHT * 8, MTL::ResourceStorageModePrivate);
    if (!input_buffer_) {
      cerr << "Failed to create input buffer\n";
      return false;
    }

    // Create the output buffer used for benchmarks that write to buffers.
    output_buffer_ =
        device_->newBuffer(WIDTH * HEIGHT * 8, MTL::ResourceStorageModePrivate);
    if (!output_buffer_) {
      cerr << "Failed to create output buffer\n";
      return false;
    }

    // Create the textures.
    auto *descriptor = MTL::TextureDescriptor::alloc();
    descriptor->init();
    descriptor->setTextureType(MTL::TextureType2D);
    descriptor->setPixelFormat(MTL::PixelFormatRGBA16Float);
    descriptor->setWidth(WIDTH);
    descriptor->setHeight(HEIGHT);
    descriptor->setUsage(MTL::TextureUsageShaderRead |
                         MTL::TextureUsageShaderWrite);
    input_texture_ = device_->newTexture(descriptor);
    if (!input_texture_) {
      cerr << "Failed to create the input texture\n";
      return false;
    }
    output_texture_ = device_->newTexture(descriptor);
    if (!output_texture_) {
      cerr << "Failed to create the output texture\n";
      return false;
    }

    // Fill the input buffer.
    auto *fill_buffer_pipeline = MakePipeline("write_buffer");
    if (!fill_buffer_pipeline) {
      return false;
    }
    auto *command_buffer =
        RunPipeline(1, [&](MTL::ComputeCommandEncoder *encoder) {
          encoder->setComputePipelineState(fill_buffer_pipeline);
          encoder->setBytes(kInputValueHalf, 8, 0);
          encoder->setBuffer(input_buffer_, 0, 1);
          return true;
        });
    command_buffer->release();
    fill_buffer_pipeline->release();

    // Fill the input texture.
    auto *fill_texture_pipeline = MakePipeline("write_texture_half");
    if (!fill_texture_pipeline) {
      return false;
    }
    command_buffer = RunPipeline(1, [&](MTL::ComputeCommandEncoder *encoder) {
      encoder->setComputePipelineState(fill_texture_pipeline);
      encoder->setBytes(kInputValueHalf, 8, 0);
      encoder->setTexture(input_texture_, 0);
      return true;
    });
    command_buffer->release();
    fill_texture_pipeline->release();

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

  /// Run a pipeline.
  /// @param iterations the number of iterations to run
  /// @param set_resources a callback that sets the resources on the encoder
  /// @returns true on success, false on failure
  MTL::CommandBuffer *
  RunPipeline(uint32_t iterations,
              const std::function<bool(MTL::ComputeCommandEncoder *encoder)>
                  &set_resources) {
    auto *command_buffer = queue_->commandBuffer();
    if (!command_buffer) {
      cerr << "Failed to create a command buffer\n";
      return nullptr;
    }

    auto *compute_encoder = command_buffer->computeCommandEncoder();
    if (!compute_encoder) {
      cerr << "Failed to create a compute command encoder\n";
      return nullptr;
    }

    // Set up the dispatch resources.
    if (!set_resources(compute_encoder)) {
      return nullptr;
    }

    // Dispatch `iterations` instances of the pipeline.
    auto grid_size = MTL::Size(WIDTH, HEIGHT, 1);
    auto group_size = MTL::Size(GROUP_WIDTH, GROUP_HEIGHT, 1);
    for (uint32_t i = 0; i < iterations; i++) {
      compute_encoder->dispatchThreads(grid_size, group_size);
    }

    compute_encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    compute_encoder->release();

    return command_buffer;
  }

  /// Run a benchmark.
  /// @param benchmark the benchmark config to run
  /// @returns true on success, false on failure
  bool RunBenchmark(const Benchmark &benchmark) {
    cout << "\nRunning '" << benchmark.name << "'\n";

    auto *pipeline = MakePipeline(benchmark.func);
    if (!pipeline) {
      return false;
    }

    auto *command_buffer =
        RunPipeline(ITERATIONS, [&](MTL::ComputeCommandEncoder *encoder) {
          encoder->setComputePipelineState(pipeline);

          uint32_t buffer_index = 0;
          uint32_t texture_index = 0;

          switch (benchmark.input_type) {
          case Benchmark::kBuffer:
            encoder->setBuffer(input_buffer_, 0, buffer_index++);
            break;
          case Benchmark::kTexture:
            encoder->setTexture(input_texture_, texture_index++);
            break;
          case Benchmark::kBytes:
            encoder->setBytes(kInputValueHalf, 8, buffer_index++);
            break;
          }

          switch (benchmark.output_type) {
          case Benchmark::kBuffer:
            encoder->setBuffer(output_buffer_, 0, buffer_index++);
            break;
          case Benchmark::kTexture:
            encoder->setTexture(output_texture_, texture_index++);
            break;
          case Benchmark::kBytes:
            cerr << "Invalid output type 'bytes'\n";
            return false;
          }

          return true;
        });

    // Display the performance.
    double start = command_buffer->GPUStartTime();
    double end = command_buffer->GPUEndTime();
    double duration_total = end - start;
    double duration_per_call = duration_total / ITERATIONS;
    double total_bytes = WIDTH * HEIGHT * 8;
    if (benchmark.input_type != Benchmark::kBytes) {
      total_bytes *= 2;
    }
    cout << "  Total runtime = " << 1000 * duration_total << "ms\n";
    cout << "  Runtime per dispatch = " << (1000 * duration_per_call) << "ms\n";
    cout << "  Bandwidth = " << (total_bytes / duration_per_call) / 1e9
         << " GB/s\n";

    command_buffer->release();
    pipeline->release();

    return Check(benchmark.output_type);
  }

  /// Check the contents of texture.
  /// @returns true on success, false on verification failure
  bool Check(Benchmark::ResourceType output_type) {
    auto *command_buffer =
        RunPipeline(1, [&](MTL::ComputeCommandEncoder *encoder) {
          switch (output_type) {
          case Benchmark::kBuffer:
            encoder->setComputePipelineState(copy_from_buffer_);
            encoder->setBuffer(output_buffer_, 0, 0);
            encoder->setBuffer(staging_buffer_, 0, 1);
            return true;
          case Benchmark::kTexture:
            encoder->setComputePipelineState(copy_from_texture_);
            encoder->setBuffer(staging_buffer_, 0, 0);
            encoder->setTexture(output_texture_, 0);
            return true;
          case Benchmark::kBytes:
            cerr << "Invalid output type 'bytes'\n";
            return false;
          }
        });
    command_buffer->release();

    // Check all entries in the buffer.
    bool correct = true;
    float *result = static_cast<float *>(staging_buffer_->contents());
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

  for (auto &benchmark : kBenchmarks) {
    if (!context.RunBenchmark(benchmark)) {
      return 1;
    }
  }

  return 0;
}
