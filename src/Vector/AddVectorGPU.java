package Vector;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;
import jcuda.utils.KernelLauncher;

public class AddVectorGPU {

	public static void main(String[] args) {
		jcuda.driver.JCudaDriver.cuInit(0);
		String sourceCode = "extern \"C\"" + "\n" + "__global__ void add(int *result, int *a, int *b, int *N)" + "\n" + "{"
				+ "\n" + "int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;"
				+ "\n"
				+ "int i = blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;"
				+ "\n" + "if(i<*N) result[i]=a[i]+b[i];" + "\n" + "}";

		// Prepare the kernel
		KernelLauncher kernelLauncher = KernelLauncher.compile(sourceCode, "add");
		
		int result[] = new int[1048576];
		int a[] = new int[1048576];
		int b[] = new int[1048576];
		int size[] = new int[1];
		size[0]=1048576;

		for (int i = 0; i < 1048576; i++) {
			a[i] = b[i] = i;
		}
		//create pointer in device - GPU
		CUdeviceptr N = new CUdeviceptr();
		cuMemAlloc(N, Sizeof.INT);
		cuMemcpyHtoD(N, Pointer.to(size), Sizeof.INT);
		CUdeviceptr dResult = new CUdeviceptr();
		//allocate memory for the pointer
		cuMemAlloc(dResult, 1048576 * Sizeof.INT);
		CUdeviceptr dA = new CUdeviceptr();
		cuMemAlloc(dA, 1048576 * Sizeof.INT);
		//copy the memory from host (CPU) to device
		cuMemcpyHtoD(dA, Pointer.to(a), 1048576 * Sizeof.INT);
		CUdeviceptr dB = new CUdeviceptr();
		cuMemAlloc(dB, 1048576 * Sizeof.INT);
		cuMemcpyHtoD(dB, Pointer.to(b), 1048576 * Sizeof.INT);

		//set block size and grid size
		kernelLauncher.setBlockSize(32, 32, 1);
		kernelLauncher.setGridSize(32, 32, 1);
		
		long startTime = System.currentTimeMillis();
		kernelLauncher.call(dResult, dA, dB, N);
		JCuda.cudaDeviceSynchronize();

		
		cuMemcpyDtoH(Pointer.to(result), dResult, 1048576 * Sizeof.INT);
		long stopTime = System.currentTimeMillis();
		
		long elapsedTime = stopTime - startTime;
		System.out.println("Execute time in milisecond: " + elapsedTime);

		cuMemFree(dA);
		cuMemFree(dB);
		cuMemFree(dResult);
	}
}
