package Vector;




public class AddVectorCPU {
	private static int a[] = new int[1048576];
	private static int b[] = new int[1048576];
	private static int c[] = new int[1048576];
	public static void main(String[] args) {
		
		for(int i=0; i<(1048576);i++) {
			a[i]=b[i]=i;
		}
		long startTime = System.currentTimeMillis();
		for(int i=0; i<(1048576);i++) {
			c[i]=a[i]+b[i];
		}
		long stopTime = System.currentTimeMillis();
		long elapsedTime = stopTime - startTime;
		System.out.println("Execute time in milisecond: " + elapsedTime);
	}
}
