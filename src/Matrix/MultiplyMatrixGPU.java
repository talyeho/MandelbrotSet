package Matrix;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;
import jcuda.utils.KernelLauncher;

public class MultiplyMatrixGPU {
	public static void main(String[] args) {
		String sourceCode = "extern \"C\"" + "\n" + "__global__ void add(int *result, int *a, int *b, int* size)" + "\n"
				+ "{" + "\n" + "int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;"
				+ "\n"
				+ "int i = blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;"
				+ "\n" + "int row = i/(*size);" + "\n" + "int colum = i%(*size); " + "\n" + "\n"
				+ "if(row<*size && colum < *size){" + "\n" + "int sum=0;" + "\n" + "for (int j = 0; j < (*size); j++) {"
				+ "\n" + "sum+=a[row*(*size)+j] * b[j*(*size)+colum];" + "\n" + "}" + "\n"
				+ "result[row*(*size)+colum]=sum;" + "\n" + "}" + "\n" + "}";

		// Prepare the kernel
		KernelLauncher kernelLauncher = KernelLauncher.compile(sourceCode, "add");

		int size = (int) Math.pow(2, 10);
		int[] a = new int[size * size];
		int[] b = new int[size * size];
		int[] res = new int[size * size];

		for (int i = 0; i < (size); i++) {
			for (int j = 0; j < (size); j++) {
				a[i*size + j] = b[i*size + j] = j + size * i;
			}
		}
		
		int temp[] = new int[1];
		temp[0] = size;
		CUdeviceptr GPUsize = new CUdeviceptr();
		cuMemAlloc(GPUsize, Sizeof.INT);
		cuMemcpyHtoD(GPUsize, Pointer.to(temp), Sizeof.INT);
		CUdeviceptr dResult = new CUdeviceptr();
		cuMemAlloc(dResult, size * size * Sizeof.INT);
		CUdeviceptr dA = new CUdeviceptr();
		cuMemAlloc(dA, size * size * Sizeof.POINTER);

		CUdeviceptr dB = new CUdeviceptr();
		cuMemAlloc(dB, size * size * Sizeof.POINTER);

		cuMemcpyHtoD(dA, Pointer.to(a), size * size * Sizeof.POINTER);
		cuMemcpyHtoD(dB, Pointer.to(b), size * size * Sizeof.POINTER);
		// set block size and grid size
		kernelLauncher.setBlockSize(size, 1, 1);
		kernelLauncher.setGridSize(size, 1, 1);

		long startTime = System.currentTimeMillis();
		kernelLauncher.call(dResult, dA, dB, GPUsize);
		JCuda.cudaDeviceSynchronize();

		cuMemcpyDtoH(Pointer.to(res), dResult, size * size * Sizeof.INT);
		long stopTime = System.currentTimeMillis();

		long elapsedTime = stopTime - startTime;
		System.out.println("Execute time in milisecond: " + elapsedTime);
		cuMemFree(dA);
		cuMemFree(dB);
		cuMemFree(GPUsize);
		cuMemFree(dResult);

	}
}
