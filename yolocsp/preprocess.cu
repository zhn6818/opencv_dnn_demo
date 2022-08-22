#include "preprocess.h"

using namespace std;

// preprocess (NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1](Normalize))
__global__ void preprocess_kernel(float* output, uint8_t* input,
    const int batchSize, const int height, const int width, const int channel,
    const int thread_count)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= thread_count) return;

    const int w_idx = index % width;
    int idx = index / width;
    const int h_idx = idx % height;
    idx /= height;
    const int c_idx = idx % channel;
    const int b_idx = idx / channel;

    int g_idx = b_idx * height * width * channel + h_idx * width * channel + w_idx * channel + 2 - c_idx;

    output[index] = input[g_idx] / 255.f;
}

void preprocess(float* output, uint8_t*input, int batchSize, int height, int width, int channel, cudaStream_t stream)
{
    int thread_count = batchSize * height * width * channel;
    int block = 512;
    int grid = (thread_count - 1) / block + 1;

    preprocess_kernel << <grid, block, 0, stream >> > (output, input, batchSize, height, width, channel, thread_count);
}

#include "preprocess.h"

namespace nvinfer1
{
    int PreprocessPluginV2::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) noexcept
    {
        uint8_t* input = (uint8_t*)inputs[0];
        float* output = (float*)outputs[0];

        const int H = mPreprocess.H;
        const int W = mPreprocess.W;
        const int C = mPreprocess.C;

        preprocess(output, input, batchSize, H, W, C, stream);

        return 0;
    }
}

// namespace nvinfer1
// {
//     __global__ void preprocess_kernel(float* output, uint8_t* input,
//         const int batchSize, const int height, const int width, const int channel,
//         const int thread_count)
//     {
//         int index = threadIdx.x + blockIdx.x * blockDim.x;
//         if (index >= thread_count) return;

//         const int w_idx = index % width;
//         int idx = index / width;
//         const int h_idx = idx % height;
//         idx /= height;
//         const int c_idx = idx % channel;
//         const int b_idx = idx / channel;

//         int g_idx = b_idx * height * width * channel + h_idx * width * channel + w_idx * channel + 2 - c_idx;

//         output[index] = input[g_idx] / 255.f;
//     }

//     void preprocess(float* output, uint8_t*input, int batchSize, int height, int width, int channel, cudaStream_t stream)
//     {
//         int thread_count = batchSize * height * width * channel;
//         int block = 512;
//         int grid = (thread_count - 1) / block + 1;

//         preprocess_kernel << <grid, block, 0, stream >> > (output, input, batchSize, height, width, channel, thread_count);
//     }
//     PreprocessPluginV2::PreprocessPluginV2(const Preprocess &arg)
//     {
//         mPreprocess = arg;
//     }

//     PreprocessPluginV2::PreprocessPluginV2(const void *data, size_t length)
//     {
//         const char *d = static_cast<const char *>(data);
//         const char *const a = d;
//         mPreprocess = read_a<Preprocess>(d);
//         assert(d == a + length);
//     }
//     PreprocessPluginV2::~PreprocessPluginV2()
//     {
//     }
//     Dims PreprocessPluginV2::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
//     {
//         return Dims3(mPreprocess.C, mPreprocess.H, mPreprocess.W);
//     }
//     int PreprocessPluginV2::initialize()
//     {
//         return 0;
//     }

//     int PreprocessPluginV2::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
//     {
//         uint8_t* input = (uint8_t*)inputs[0];
//         float* output = (float*)outputs[0];

//         const int H = mPreprocess.H;
//         const int W = mPreprocess.W;
//         const int C = mPreprocess.C;

//         preprocess(output, input, batchSize, H, W, C, stream);

//         return 0;
//     }
//     DataType PreprocessPluginV2::getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const
//     {
//         assert(inputTypes && nbInputs == 1);
//         return DataType::kFLOAT; //
//     }
//     void PreprocessPluginV2::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator)
//     {
//     }
//     void PreprocessPluginV2::detachFromContext()
//     {

//     }
//     IPluginV2Ext *PreprocessPluginV2::clone() const
//     {
//         PreprocessPluginV2 *plugin = new PreprocessPluginV2(*this);
//         return plugin;
//     }

//     IPluginV2 *PreprocessPluginV2Creator::createPlugin(const char *name, const PluginFieldCollection *fc)
//     {
//         PreprocessPluginV2 *plugin = new PreprocessPluginV2(*(Preprocess *)fc);
//         mPluginName = name;
//         return plugin;
//     }

//     IPluginV2 *PreprocessPluginV2Creator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
//     {
//         auto plugin = new PreprocessPluginV2(serialData, serialLength);
//         mPluginName = name;
//         return plugin;
//     }
// }


// // #include "preprocess.h"

// // using namespace std;

// // namespace nvinfer1
// // {
// // // preprocess (NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1](Normalize))
// //     __global__ void preprocess_kernel(float* output, uint8_t* input,
// //         const int batchSize, const int height, const int width, const int channel,
// //         const int thread_count)
// //     {
// //         int index = threadIdx.x + blockIdx.x * blockDim.x;
// //         if (index >= thread_count) return;

// //         const int w_idx = index % width;
// //         int idx = index / width;
// //         const int h_idx = idx % height;
// //         idx /= height;
// //         const int c_idx = idx % channel;
// //         const int b_idx = idx / channel;

// //         int g_idx = b_idx * height * width * channel + h_idx * width * channel + w_idx * channel + 2 - c_idx;

// //         output[index] = input[g_idx] / 255.f;
// //     }

// //     void preprocess(float* output, uint8_t*input, int batchSize, int height, int width, int channel, cudaStream_t stream)
// //     {
// //         int thread_count = batchSize * height * width * channel;
// //         int block = 512;
// //         int grid = (thread_count - 1) / block + 1;

// //         preprocess_kernel << <grid, block, 0, stream >> > (output, input, batchSize, height, width, channel, thread_count);
// //     }


// //     int PreprocessPluginV2::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
// //     {
// //         uint8_t* input = (uint8_t*)inputs[0];
// //         float* output = (float*)outputs[0];

// //         const int H = mPreprocess.H;
// //         const int W = mPreprocess.W;
// //         const int C = mPreprocess.C;

// //         preprocess(output, input, batchSize, H, W, C, stream);

// //         return 0;
// //     }
// // }