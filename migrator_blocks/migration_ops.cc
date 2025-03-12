#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>	// for at::cuda::getCurrentCUDAStream()

#define CUDA_CHECK(expr)                                                         \
	do {                                                                          \
		cudaError_t status = (expr);                                               \
		if (status != cudaSuccess) {                                               \
			std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at "      \
					  << __FILE__ << ":" << __LINE__ << std::endl;                    \
			throw std::runtime_error("CUDA error");                                   \
		}                                                                           \
	} while (0)

/*
 * Convert a cudaIpcMemHandle_t to a vector of bytes.
 */
static std::vector<int64_t> cudaIpcMemHandle2Bytes(const cudaIpcMemHandle_t &handle) {
	std::vector<int64_t> result;
	for (size_t i = 0; i < sizeof(handle); ++i) {
		result.push_back(((uint8_t*) &handle)[i]);
	}
	return result;
}

/*
 * Convert a vector of bytes to a cudaIpcMemHandle_t.
 */
static cudaIpcMemHandle_t bytes2CudaIpcMemHandle(const std::vector<int64_t> &bytes) {
	TORCH_CHECK(bytes.size() == sizeof(cudaIpcMemHandle_t), 
			   "Invalid byte array size for cudaIpcMemHandle_t");
	cudaIpcMemHandle_t result;
	for (size_t i = 0; i < sizeof(result); ++i) {
		((uint8_t*) &result)[i] = bytes[i];
	}
	return result;
}

/*
 * Get the IPC mem handle of a pytorch tensor.
 */
std::vector<int64_t> get_ipc_mem_handle(torch::Tensor tensor) {
	TORCH_CHECK(tensor.device().is_cuda(), "Input tensor must be a CUDA tensor");
	TORCH_CHECK(tensor.is_contiguous(), "Input tensor must be contiguous");
	
	cudaIpcMemHandle_t handle;
	CUDA_CHECK(cudaIpcGetMemHandle(&handle, tensor.data_ptr()));
	return cudaIpcMemHandle2Bytes(handle);
}


static cudaIpcMemHandle_t ipc_mem_handlers[4096];
static void* ipc_mem_ptrs[4096];

bool register_ipc_mem_handler(
	int index,
	const std::vector<int64_t>& handle_bytes
){
	if (index >= 4096) {
		throw std::runtime_error("Index out of bounds");
	}

	cudaIpcMemHandle_t handle = bytes2CudaIpcMemHandle(handle_bytes);
	
	void* ptr;
	cudaError_t err = cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess);
	if (err != cudaSuccess) {
		std::cerr << "Failed to open IPC memory handle: " << err << " " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	
	ipc_mem_handlers[index] = handle;
	ipc_mem_ptrs[index] = ptr;
	return true;
}


/*
 * Open an IPC memory handle and return a tensor that shares the memory.
 */
torch::Tensor open_ipc_mem_handle(const std::vector<int64_t>& handle_bytes, 
								 const std::vector<int64_t>& sizes,
								 torch::ScalarType dtype) {
	cudaIpcMemHandle_t handle = bytes2CudaIpcMemHandle(handle_bytes);
	void* ptr;
	CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
	
	return torch::from_blob(ptr, 
						  sizes,
						  [ptr](void*) { 
							  cudaIpcCloseMemHandle(ptr);
						  },
						  torch::TensorOptions()
							  .device(torch::kCUDA)
							  .dtype(dtype));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("cudaIpcMemHandle2Bytes", &cudaIpcMemHandle2Bytes, "Convert CUDA IPC memory handle to bytes");
	m.def("bytes2CudaIpcMemHandle", &bytes2CudaIpcMemHandle, "Convert bytes to CUDA IPC memory handle");
	m.def("get_ipc_mem_handle", &get_ipc_mem_handle, "Get CUDA IPC memory handle for tensor");
	m.def("open_ipc_mem_handle", &open_ipc_mem_handle, "Open CUDA IPC memory handle and create tensor");
	m.def("register_ipc_mem_handler", &register_ipc_mem_handler, "Register CUDA IPC memory handler");
}

