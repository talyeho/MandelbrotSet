package Mandelbrot;

import static jcuda.driver.JCudaDriver.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import jcuda.utils.KernelLauncher;

public class GPUMandelbrot {
	public static void main(String args[]) throws IOException {
		JCudaDriver.setExceptionsEnabled(true);

		String sourceCode = "extern \"C\"" + "\n"
				+ "__global__ void add(float *result, int *iter, float *height, float *width, float* zoom)" + "\n" + "{" + "\n"
				+ "int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;" + "\n"
				+ "int i = blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;"
				+ "\n" + "float x_re = (i%((int)height[0]) - (height[0])/2.0)*zoom[0];" + "\n"
				+ "float y_im = (i/((int)width[0]) - (width[0])/2.0)*zoom[0];" + "\n" + "float x = 0 , y = 0, temp;" + "\n"
				+ "int iterations=0;" + "\n" + "while(x*x + y*y < 4 && iterations < iter[0]) {" + "\n"
				+ "temp = x*x - y*y + x_re;" + "\n" + "y = 2*x*y+y_im;" + "\n" + "x=temp;" + "\n" + "iterations++;"
				+ "\n" + "}" + "\n" + "result[i] = iterations/(iter[0]*1.0);" + "\n" + "}";

		// Prepare the kernel
		KernelLauncher kernelLauncher = KernelLauncher.compile(sourceCode, "add");

		// Create the input data
		int size = 2048;
		float result[] = new float[size * size];
		float a[] = new float[1];
		float b[] = new float[1];
		float c[] = new float[1];
		int iter[] = new int[1];

		a[0] = size;
		b[0] = size;
		c[0] = (float) (4 / (float) (size));
		iter[0]=32000;
		System.out.println(c[0]);
		// Allocate the device memory and copy the input
		// data to the device
		System.out.println("Initializing device memory...");
		CUdeviceptr dResult = new CUdeviceptr();
		cuMemAlloc(dResult, size * size * Sizeof.FLOAT);
		CUdeviceptr dA = new CUdeviceptr();
		cuMemAlloc(dA, 1 * Sizeof.FLOAT);
		cuMemcpyHtoD(dA, Pointer.to(a), 1 * Sizeof.FLOAT);
		CUdeviceptr dB = new CUdeviceptr();
		cuMemAlloc(dB, 1 * Sizeof.FLOAT);
		cuMemcpyHtoD(dB, Pointer.to(b), 1 * Sizeof.FLOAT);
		CUdeviceptr dC = new CUdeviceptr();
		cuMemAlloc(dC, 1 * Sizeof.FLOAT);
		cuMemcpyHtoD(dC, Pointer.to(c), 1 * Sizeof.FLOAT);
		CUdeviceptr dIter = new CUdeviceptr();
		cuMemAlloc(dIter, 1 * Sizeof.INT);
		cuMemcpyHtoD(dIter, Pointer.to(iter), 1 * Sizeof.INT);

		// Call the kernel
		System.out.println("Calling the kernel...");

		kernelLauncher.setBlockSize(4, 4, 8);
		kernelLauncher.setGridSize(size / 16, size / 8);
		long startTime = System.currentTimeMillis();
		kernelLauncher.call(dResult, dIter, dA, dB, dC);
		System.out.println("Hi");
		//JCuda.cudaDeviceSynchronize(); exist in kernelLauncher.call

		// Copy the result from the device to the host
		System.out.println("Obtaining results...");

		cuMemcpyDtoH(Pointer.to(result), dResult, size * size * Sizeof.FLOAT);

		// set BufferedImage to display mandelbrot image/set
		BufferedImage image = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
		int black = 0x00000, white=0xFFFFFF, step = 0x030303, light2 = 0x00005E, lightSteps2 = 0x01015C - light2, light1 = 0xFF3600, lightSteps1 = 0xFF3904-light1;
		long stopTime = System.currentTimeMillis();
		long elapsedTime = stopTime - startTime;
		System.out.println("Execute time in milisecond: " + elapsedTime);
		// System.out.println("Obtaining results...");
		int x, y;
		double temp;

		// set the image from result
		for (int i = 0; i < size * size; i++) {
			x = (int) (i % size);
			y = (int) (i / size);
			temp = white;
			if (result[i] > 0.9) {
				temp = white / step;
				temp = (1 - result[i]) * temp * step; // get place in RGB
			}
			if (result[i] > 0.0009 && result[i] < 0.9) {
				temp = ((white - light1) / lightSteps1);
				temp = (1 - result[i] / 0.9) * temp * lightSteps1 + light1;
			}
			if (result[i] > 0.0005 && result[i] < 0.0009) {
				temp = ((white - light2) / lightSteps2);
				temp = (1 - result[i] / 0.0009) * temp * lightSteps2 + light2;
			}
			image.setRGB(x, y, (int) (temp));
		}
		ImageIO.write(image, "png", new File("mandelbrot.png"));
		stopTime = System.currentTimeMillis();
		elapsedTime = stopTime - startTime;
		System.out.println("Execute after main time in milisecond: " + elapsedTime);

		// Clean up
		cuMemFree(dA);
		cuMemFree(dB);
		cuMemFree(dC);
		cuMemFree(dIter);
		cuMemFree(dResult);
	}
}
