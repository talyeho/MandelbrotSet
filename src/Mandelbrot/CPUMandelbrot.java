package Mandelbrot;

import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

public class CPUMandelbrot {
    public static void main(String[] args) throws Exception {
        int width = 2048, height = 2048, max = 32000;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int white = 0xFFFFFF, step = 0x030303, light2 = 0xf6ff00, lightSteps2 = 0x000003 - light2, light1 = 0xFf6500, lightSteps1 = 0x000006 - light1;
        int temp;
        float normalizeIterations;
        long startTime = System.currentTimeMillis();
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                double c_re = (col - width/2)*4.0/width;
                double c_im = (row - height/2)*4.0/width;
                double x = 0, y = 0;
                int iterations = 0;
                while (x*x+y*y < 4 && iterations < max) {
                    double x_new = x*x-y*y+c_re;
                    y = 2*x*y+c_im;
                    x = x_new;
                    iterations++;
                } 
                normalizeIterations=iterations/(float)(max);	//how close we were to be in the set
    			temp = (int) (white * normalizeIterations); // get place in RGB
    			temp = (int) (temp / step) * step;
    			if (normalizeIterations > 0.05 && normalizeIterations < 0.999) {
    				temp = (int) ((white - light1) / lightSteps1);
    				temp = (int) (normalizeIterations * (temp * lightSteps1 + light1));
    			}
//    			if (iterations > 0.09 && normalizeIterations < 0.1) {
//    				temp = (int) ((white - light2) / lightSteps2);
//    				temp = (int) (normalizeIterations * (temp * lightSteps2 + light2));
//    			}
    			image.setRGB(col, row, (int) (temp));
            }
        }
		long stopTime = System.currentTimeMillis();
		long elapsedTime = stopTime - startTime;
		System.out.println("Execute time in milisecond: " + elapsedTime);

        ImageIO.write(image, "png", new File("mandelbrotCPU.png"));
    }
    
    
    
    
    
    
    
    
    
    
    
    
}