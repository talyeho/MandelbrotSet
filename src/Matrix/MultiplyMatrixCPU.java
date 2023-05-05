package Matrix;


public class MultiplyMatrixCPU {
	

	public static void main(String[] args) {
		int size = (int) Math.pow(2, 12);
		int[][]a = new int[size][size];
		int[][] b = new int[size][size];
		int[][] res = new int[size][size];

		for (int i = 0; i < (size); i++) {
			for (int j = 0; j < (size); j++) {
				a[i][j] = b[i][j] = j + size * i;
			}
		}
		int temp=0;
		long startTime = System.currentTimeMillis();
		for (int k = 0; k < (size); k++) {
			for (int i = 0; i < (size); i++) {
				for (int j = 0; j < size; j++) {
					temp+=a[k][j]*b[j][i];
				}
				res[k][i]=temp;
				temp=0;
			}
		}
		long stopTime = System.currentTimeMillis();
		long elapsedTime = stopTime - startTime;
		System.out.println("Execute time in milisecond: " + elapsedTime);

	}
}
