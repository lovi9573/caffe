#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

#ifdef GPU_ENABLED  // Normal GPU + CPU Caffe.

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#ifdef USE_CUDNN  // cuDNN acceleration library.
#include "caffe/util/cudnn.hpp"
#endif

//
// CUDA macros
//

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace caffe {

// CUDA: library error reporting.
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
    const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
    const int CAFFE_CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

}  // namespace caffe

#elif defined FPGA_ENABLED

	//Todo: fill in fpga code

#else
	#define NO_DEVICE LOG(FATAL) << "No accelerator device support. Check mode or recompile with support."
#endif // Device specific code inclusion

#ifndef GPU_ENABLED

#include <vector>

// Stub out GPU calls as unavailable.

#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

#define STUB_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \
template <typename Dtype> \
void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \

#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#endif //NO GPU

#ifndef FPGA_ENABLED

#define NO_FPGA LOG(FATAL) << "Cannot use FPGA. Check mode, or recompile with fpga support."

#define STUB_FPGA(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_fpga(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_FPGA; } \
template <typename Dtype> \
void classname<Dtype>::Backward_fpga(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_FPGA; } \

#define STUB_FPGA_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##fpga(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_FPGA; } \

#define STUB_FPGA_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##fpga(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_FPGA; } \

#endif  //NO FPGA

#endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_
